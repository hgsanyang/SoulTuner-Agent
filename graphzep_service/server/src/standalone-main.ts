import { Hono } from 'hono';
import { serve } from '@hono/node-server';
import { zValidator } from '@hono/zod-validator';
import { getSettings, resolveApiKey } from './config/settings.js';
import {
  AddMessagesRequestSchema,
  type AddMessagesRequest
} from './dto/ingest.js';
import {
  SearchQuerySchema,
  GetMemoryRequestSchema,
  type SearchQuery,
  type GetMemoryRequest,
  type SearchResults,
  type GetMemoryResponse,
  type FactResult
} from './dto/retrieve.js';
import { type Message } from './dto/common.js';

// ---- 【V2 升级】引入真正的 Graphzep 库组件 ----
import { Graphzep } from '../../src/graphzep.js';
import { Neo4jDriver } from '../../src/drivers/neo4j.js';
import { OpenAIEmbedder } from '../../src/embedders/openai.js';
import { OpenAIClient } from '../../src/llm/openai.js';
import { ZepMemoryManager } from '../../src/zep/memory.js';
import { ZepRetrieval } from '../../src/zep/retrieval.js';

const app = new Hono();

// ---- 全局单例（在 initializeServer 中初始化） ----
let graphzep: Graphzep;
let neo4jDriver: Neo4jDriver;
let memoryManager: ZepMemoryManager;
let retrieval: ZepRetrieval;

// Simple async worker queue for background fact extraction
const jobQueue: (() => Promise<void>)[] = [];
let isProcessing = false;

async function processQueue() {
  if (isProcessing) return;
  isProcessing = true;

  while (jobQueue.length > 0) {
    const job = jobQueue.shift();
    if (job) {
      try {
        console.log(`[GraphZep] Processing background job (remaining: ${jobQueue.length})`);
        await job();
      } catch (error) {
        console.error('[GraphZep] Background job error:', error);
      }
    }
  }

  isProcessing = false;
}

// ============================================================
// 初始化：连接 Neo4j + 实例化 Graphzep 三大组件
// ============================================================
async function initializeServer() {
  const settings = getSettings();
  const apiKey = resolveApiKey(settings);

  console.log('[GraphZep] Initializing with Neo4j backend...');
  console.log('[GraphZep] Neo4j URI:', settings.NEO4J_URI);
  console.log('[GraphZep] LLM Model:', settings.MODEL_NAME);
  console.log('[GraphZep] Embedding Model:', settings.EMBEDDING_MODEL_NAME);

  // 1. 创建 Neo4j 驱动（与音乐图谱共用同一个 Neo4j 实例）
  neo4jDriver = new Neo4jDriver(
    settings.NEO4J_URI,
    settings.NEO4J_USER,
    settings.NEO4J_PASSWORD
  );

  // 验证连接
  await neo4jDriver.verifyConnectivity();
  console.log('[GraphZep] ✅ Neo4j connection verified');

  // 2. 创建 Embedding 客户端（使用 SiliconFlow / OpenAI 兼容接口）
  const embedder = new OpenAIEmbedder({
    apiKey,
    baseURL: settings.OPENAI_BASE_URL || 'https://api.siliconflow.cn/v1',
    model: settings.EMBEDDING_MODEL_NAME,
  });

  // 3. 创建 LLM 客户端（用于实体关系提取、事实提取）
  const llmClient = new OpenAIClient({
    apiKey,
    baseURL: settings.OPENAI_BASE_URL || 'https://api.siliconflow.cn/v1',
    model: settings.MODEL_NAME,
    temperature: 0.3,  // 事实提取需要低温度
    maxTokens: 2000,
  });

  // 4. 创建 Graphzep 核心实例
  graphzep = new Graphzep({
    driver: neo4jDriver,
    llmClient,
    embedder,
    groupId: 'music-agent-memory',
  });

  // 创建 Neo4j 索引（如果不存在）
  await graphzep.createIndexes();
  console.log('[GraphZep] ✅ Neo4j indexes ensured');

  // 5. 创建 Zep 记忆管理器（LLM 事实提取 + 持久化）
  memoryManager = new ZepMemoryManager(graphzep, llmClient, embedder, neo4jDriver);

  // 6. 创建 Zep 检索引擎（语义/关键词/混合/MMR 搜索）
  retrieval = new ZepRetrieval(embedder, neo4jDriver);

  console.log('[GraphZep] ✅ Server fully initialized with Neo4j + Zep library');
}

// ============================================================
// Health check
// ============================================================
app.get('/healthcheck', (c) => {
  return c.json({ status: 'healthy', backend: 'neo4j' });
});

// ============================================================
// POST /messages — 写入对话 + 异步 LLM 实体/事实提取
// ============================================================
app.post('/messages', zValidator('json', AddMessagesRequestSchema), async (c) => {
  const request: AddMessagesRequest = c.req.valid('json');
  const groupId = request.group_id;

  // 对每条消息：立即写入 Graphzep 事件节点 + 排队异步 LLM 提取
  for (const message of request.messages) {
    const content = `${message.role || ''}(${message.role_type}): ${message.content}`;

    // 异步执行：Graphzep.addEpisode 会自动做 LLM 实体提取 + 嵌入
    jobQueue.push(async () => {
      try {
        await graphzep.addEpisode({
          content,
          groupId,
          metadata: {
            role: message.role,
            role_type: message.role_type,
            timestamp: message.timestamp,
          },
        });

        // 同时用 ZepMemoryManager 做事实三元组提取
        await memoryManager.addMemory({
          content: message.content,
          sessionId: groupId,
          metadata: {
            role: message.role,
            role_type: message.role_type,
          },
        });

        console.log(`[GraphZep] ✅ Processed episode for group ${groupId}: ${message.content.substring(0, 60)}...`);
      } catch (error) {
        console.error(`[GraphZep] ❌ Failed to process episode: ${error}`);
      }
    });
  }

  // 启动后台队列处理
  processQueue().catch(console.error);

  return c.json({ message: 'Messages added to processing queue', success: true }, 202);
});

// ============================================================
// DELETE /group/:groupId — 删除某个 group 的所有记忆
// ============================================================
app.delete('/group/:groupId', async (c) => {
  const groupId = c.req.param('groupId');

  try {
    // 删除 Graphzep 图中该 group 的所有节点
    await neo4jDriver.executeQuery(
      'MATCH (n) WHERE n.groupId = $groupId DETACH DELETE n',
      { groupId }
    );
    // 删除 Zep 记忆节点
    await neo4jDriver.executeQuery(
      'MATCH (m:ZepMemory {sessionId: $groupId}) DETACH DELETE m',
      { groupId }
    );

    console.log(`[GraphZep] Deleted all data for group: ${groupId}`);
    return c.json({ message: 'Group deleted', success: true });
  } catch (error) {
    console.error(`[GraphZep] Delete error: ${error}`);
    return c.json({ error: `Failed to delete group: ${error}` }, 500);
  }
});

// ============================================================
// POST /clear — 清空所有 GraphZep 数据（不影响 Song/Artist 等音乐图谱节点）
// ============================================================
app.post('/clear', async (c) => {
  try {
    // 只清除 GraphZep 相关的节点标签，不删除音乐图谱数据
    await neo4jDriver.executeQuery(
      'MATCH (n) WHERE n:Entity OR n:Episodic OR n:Community OR n:ZepMemory OR n:ZepFact DETACH DELETE n'
    );
    console.log('[GraphZep] Cleared all GraphZep data (music graph untouched)');
    return c.json({ message: 'GraphZep data cleared', success: true });
  } catch (error) {
    console.error(`[GraphZep] Clear error: ${error}`);
    return c.json({ error: `Failed to clear: ${error}` }, 500);
  }
});

// ============================================================
// POST /search — 语义/混合搜索记忆
// ============================================================
app.post('/search', zValidator('json', SearchQuerySchema), async (c) => {
  const query: SearchQuery = c.req.valid('json');

  try {
    // 使用 Graphzep 的语义搜索（embedding cosine similarity）
    const graphNodes = await graphzep.search({
      query: query.query,
      groupId: query.group_ids?.[0],
      limit: query.max_facts,
      searchType: 'hybrid',
    });

    // 同时搜索 Zep 记忆中的事实三元组
    const zepResults = await retrieval.search({
      query: query.query,
      limit: query.max_facts,
      searchType: 'hybrid',
    });

    // 合并两路结果为统一的 FactResult 格式
    const facts: FactResult[] = [];

    // 从 Graphzep 语义搜索结果中提取
    for (const node of graphNodes) {
      const props = (node as any).properties || node;
      facts.push({
        uuid: props.uuid || `graph_${facts.length}`,
        name: props.name || `Node ${facts.length}`,
        fact: props.content || props.summary || props.name || '',
        valid_at: props.validAt || props.createdAt || new Date().toISOString(),
        invalid_at: null,
        created_at: props.createdAt || new Date().toISOString(),
        expired_at: null,
      });
    }

    // 从 Zep 记忆搜索中提取事实
    for (const result of zepResults) {
      const mem = result.memory;
      // 如果有 facts（SPO 三元组），展开它们
      if (mem.facts && mem.facts.length > 0) {
        for (const fact of mem.facts) {
          facts.push({
            uuid: fact.uuid,
            name: `${fact.subject} ${fact.predicate}`,
            fact: `${fact.subject} ${fact.predicate} ${fact.object}`,
            valid_at: fact.validFrom?.toISOString() || new Date().toISOString(),
            invalid_at: fact.validUntil?.toISOString() || null,
            created_at: mem.createdAt?.toISOString() || new Date().toISOString(),
            expired_at: null,
          });
        }
      } else {
        // 没有 facts 就返回原始内容
        facts.push({
          uuid: mem.uuid,
          name: `Memory ${mem.uuid.slice(-6)}`,
          fact: mem.content,
          valid_at: mem.validFrom?.toISOString() || new Date().toISOString(),
          invalid_at: null,
          created_at: mem.createdAt?.toISOString() || new Date().toISOString(),
          expired_at: null,
        });
      }
    }

    // 去重（按 fact 内容）并限制数量
    const uniqueFacts = Array.from(
      new Map(facts.map(f => [f.fact, f])).values()
    ).slice(0, query.max_facts);

    const results: SearchResults = { facts: uniqueFacts };

    console.log(`[GraphZep] Search "${query.query}" → ${uniqueFacts.length} facts (graph: ${graphNodes.length}, zep: ${zepResults.length})`);

    return c.json(results);
  } catch (error) {
    console.error(`[GraphZep] Search error: ${error}`);
    // Graceful 降级：返回空结果而非 500
    return c.json({ facts: [] } as SearchResults);
  }
});

// ============================================================
// GET /episodes/:groupId — 获取某个 group 的最近 episodes
// ============================================================
app.get('/episodes/:groupId', async (c) => {
  const groupId = c.req.param('groupId');
  const lastN = parseInt(c.req.query('last_n') || '10');

  try {
    const results = await neo4jDriver.executeQuery<any>(
      `MATCH (n:Episodic {groupId: $groupId})
       RETURN n
       ORDER BY n.createdAt DESC
       LIMIT $limit`,
      { groupId, limit: lastN }
    );

    const episodes = results.map((r: any) => {
      const props = r.n?.properties || r.n || {};
      return {
        uuid: props.uuid,
        group_id: groupId,
        content: props.content || '',
        timestamp: props.createdAt || new Date().toISOString(),
      };
    });

    console.log(`[GraphZep] Retrieved ${episodes.length} episodes for group ${groupId}`);
    return c.json(episodes);
  } catch (error) {
    console.error(`[GraphZep] Episodes error: ${error}`);
    return c.json([]);
  }
});

// ============================================================
// POST /get-memory — 基于消息上下文获取相关记忆
// ============================================================
app.post('/get-memory', zValidator('json', GetMemoryRequestSchema), async (c) => {
  const request: GetMemoryRequest = c.req.valid('json');

  try {
    // 将所有消息拼接为综合查询
    const combinedQuery = request.messages
      .map(m => m.content)
      .join(' ');

    // 使用 Zep retrieval 做混合搜索
    const zepResults = await retrieval.search({
      query: combinedQuery,
      limit: request.max_facts,
      searchType: 'hybrid',
    });

    const facts: FactResult[] = zepResults.map((result: any) => {
      const mem = result.memory;
      // 优先返回提取的 facts
      if (mem.facts && mem.facts.length > 0) {
        const fact = mem.facts[0]; // 取第一条事实
        return {
          uuid: fact.uuid,
          name: `${fact.subject} ${fact.predicate}`,
          fact: `${fact.subject} ${fact.predicate} ${fact.object}`,
          valid_at: fact.validFrom?.toISOString() || new Date().toISOString(),
          invalid_at: fact.validUntil?.toISOString() || null,
          created_at: mem.createdAt?.toISOString() || new Date().toISOString(),
          expired_at: null,
        };
      }
      return {
        uuid: mem.uuid,
        name: `Memory ${mem.uuid.slice(-6)}`,
        fact: mem.content,
        valid_at: mem.validFrom?.toISOString() || new Date().toISOString(),
        invalid_at: null,
        created_at: mem.createdAt?.toISOString() || new Date().toISOString(),
        expired_at: null,
      };
    });

    const response: GetMemoryResponse = { facts };

    console.log(`[GraphZep] Get memory for group ${request.group_id} → ${facts.length} facts`);
    return c.json(response);
  } catch (error) {
    console.error(`[GraphZep] Get memory error: ${error}`);
    return c.json({ facts: [] } as GetMemoryResponse);
  }
});

// ============================================================
// Error handler
// ============================================================
app.onError((err, c) => {
  console.error('[GraphZep] Server error:', err);
  return c.json({ error: err.message }, 500);
});

// Graceful shutdown
process.on('SIGINT', async () => {
  console.log('[GraphZep] Shutting down...');
  if (neo4jDriver) {
    await neo4jDriver.close();
    console.log('[GraphZep] Neo4j connection closed');
  }
  process.exit(0);
});

// ============================================================
// Start server
// ============================================================
async function startServer() {
  try {
    await initializeServer();

    const settings = getSettings();
    const port = settings.PORT || 3100;

    console.log(`\n[GraphZep] 🚀 Server running on port ${port} (Neo4j backend)`);
    console.log('[GraphZep] Available endpoints:');
    console.log('  GET  /healthcheck');
    console.log('  POST /messages       ← 写入对话 + 异步 LLM 实体提取');
    console.log('  POST /search         ← 语义/混合搜索记忆');
    console.log('  POST /get-memory     ← 基于上下文获取记忆');
    console.log('  GET  /episodes/:gid  ← 获取最近 episodes');
    console.log('  DELETE /group/:gid   ← 删除 group 数据');
    console.log('  POST /clear          ← 清空 GraphZep 数据\n');

    serve({
      fetch: app.fetch,
      port,
    });
  } catch (error) {
    console.error('[GraphZep] ❌ Failed to start server:', error);
    process.exit(1);
  }
}

startServer().catch(console.error);