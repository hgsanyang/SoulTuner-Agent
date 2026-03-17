import asyncio
import os
import aiohttp
import json
from typing import List, Dict, Any
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv(override=True)

from config.logging_config import get_logger
logger = get_logger(__name__)

from config.settings import settings

# SearxNG 自建元搜索引擎地址（docker 启动后使用）
SEARXNG_BASE_URL = settings.searxng_base_url

async def fetch_zhipu_search(query: str, session: aiohttp.ClientSession) -> List[Dict[str, str]]:
    """
    调用智谱 search-std 专用搜索模型。
    相比 glm-4-flash + web_search tool，search-std 原生返回结构化搜索结果列表，
    更快、更省 token、额度独立计算。
    响应中 message.web_search 包含结构化条目，message.content 包含 AI 摘要。
    """
    api_key = os.getenv("ZHIPU_API_KEY") or getattr(settings, "zhipu_api_key", None)
    if not api_key:
        logger.warning("ZHIPU_API_KEY not found in env or settings.")
        return []

    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # search-std 直接搜索，无需 tools 参数
    payload = {
        "model": "search-std",
        "messages": [
            {"role": "user", "content": query}
        ],
    }

    try:
        logger.info(f"🚀 发送智谱 search-std 请求... Query: {query}")
        async with session.post(url, headers=headers, json=payload, timeout=settings.web_search_timeout) as response:
            resp_text = await response.text()
            if response.status != 200:
                logger.error(f"❌ Zhipu search-std 失败 status={response.status}: {resp_text[:200]}")
                return []

            data = json.loads(resp_text)
            msg = data["choices"][0]["message"]
            results: List[Dict[str, str]] = []

            # ① 结构化搜索条目（search-std 专有字段）
            for item in msg.get("web_search", []):
                results.append({
                    "title": item.get("title", ""),
                    "content": item.get("content", "") or item.get("snippet", ""),
                    "url": item.get("link", item.get("url", "")),
                    "source": "Zhipu_search-std",
                })

            # ② AI 总结摘要（无论是否有上面的结构化结果都附加）
            summary = msg.get("content", "")
            if summary:
                results.append({
                    "title": f"智谱 AI 摘要: {query}",
                    "content": summary,
                    "url": "https://zhipu.cn",
                    "source": "Zhipu_AI_Summary",
                })

            logger.info(f"✅ Zhipu search-std 成功，共 {len(results)} 条（含摘要）")
            return results

    except (KeyError, IndexError) as e:
        logger.warning(f"⚠️ Zhipu search-std 解析异常: {e}")
    except Exception as e:
        logger.warning(f"❌ Zhipu search-std 请求异常: {e}")

    return []


async def fetch_searxng_search(query: str, session: aiohttp.ClientSession) -> List[Dict[str, str]]:
    """
    调用本地 SearxNG 元搜索引擎（Docker 自建），聚合 Google/Bing/百度/bilibili 等多源结果。
    SearxNG 未启动时静默返回 []，不影响其他搜索引擎。
    """
    url = f"{SEARXNG_BASE_URL}/search"
    params = {
        "q": query,
        "format": "json",       # 需要 settings.yml 中开启 json 格式
        "language": "zh-CN",
        "safesearch": "0",
        "categories": "general,music",
    }
    try:
        logger.info(f"🚀 发送 SearxNG 请求... Query: {query}")
        async with session.get(url, params=params, timeout=settings.searxng_timeout) as response:
            if response.status != 200:
                logger.debug(f"⚠️ SearxNG 无法使用 (状态码 {response.status})，正常降级跳过")
                return []
            data = await response.json(content_type=None)
            results = []
            for item in data.get("results", [])[:6]:  # 最多取 6 条
                results.append({
                    "title": item.get("title", ""),
                    "content": item.get("content", "") or item.get("snippet", ""),
                    "url": item.get("url", ""),
                    "source": f"SearxNG({item.get('engine', 'unknown')})",
                })
            logger.info(f"✅ SearxNG 搜索成功，找到 {len(results)} 条结果")
            return results
    except aiohttp.ClientConnectorError:
        logger.debug("[SearxNG] 服务未启动（连接被拒绝），跳过")
        return []
    except Exception as e:
        logger.warning(f"❌ SearxNG 搜索异常: {e}")
        return []


async def fetch_tavily_search(query: str, session: aiohttp.ClientSession) -> List[Dict[str, str]]:
    """调用 Tavily 的高级 Search API 进行极速搜索"""
    api_key = os.getenv("TAVILY_API_KEY") or getattr(settings, "tavily_api_key", None)
    if not api_key:
        logger.warning("TAVILY_API_KEY not found in env or settings.")
        return []
        
    url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key
    }
    payload = {
        "api_key": api_key,          # 与官方示例保持一致，key 同时放入 body
        "query": query,
        "search_depth": "basic",     # basic 免费 key 可用；advanced 需要付费计划
        "include_answer": True,
        "max_results": settings.web_search_max_results
    }
    
    try:
        logger.info(f"🚀 发送 Tavily Search 请求... Query: {query}")
        async with session.post(url, headers=headers, json=payload, timeout=settings.web_search_timeout) as response:
            resp_text = await response.text()
            if response.status != 200:
                logger.error(f"❌ Tavily Search failed with status {response.status}. Resp: {resp_text[:200]}")
                return []
                
            data = json.loads(resp_text)
            results = []
            
            if "answer" in data and data["answer"]:
                results.append({"title": "Tavily AI 实时解答", "content": data["answer"], "url": "https://tavily.com", "source": "Tavily_AI_Answer"})
                
            for item in data.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "url": item.get("url", ""),
                    "source": "Tavily_Web_Search"
                })
            
            logger.info(f"✅ Tavily Search success. Found {len(results)} items.")
            return results
            
    except Exception as e:
        logger.warning(f"❌ Tavily Search exception: {e}")
        
    return []

async def _federated_search_async(query: str) -> str:
    """异步并发请求所有的搜索引擎"""
    logger.info(f"🌐 Initiating Federated Web Search for: {query}")
    
    async with aiohttp.ClientSession() as session:
        # 并发派发任务（SearxNG 未启动时自动跳过，不阻塞）
        tasks = [
            fetch_zhipu_search(query, session),
            fetch_tavily_search(query, session),
            fetch_searxng_search(query, session),   # 自建 SearxNG 聚合搜索
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
    # 结果拼装去重
    combined_docs = []
    seen_urls = set()
    
    for res in results:
        if isinstance(res, list):
            for item in res:
                url = item.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    combined_docs.append(item)
                elif not url: # 如果没有url也加进去（如纯文本总结）
                    combined_docs.append(item)
                    
    if not combined_docs:
        return "网络搜索未能找到相关有效信息。"
        
    # 格式化输出为纯文本给大模型食用
    formatted_str = "[网络搜索结果聚合]\n"
    for idx, doc in enumerate(combined_docs):
        formatted_str += f"{idx+1}. [{doc['source']}] {doc['title']}\n摘要: {doc['content']}\n"
        
    logger.info(f"✅ Federated Search found {len(combined_docs)} snippets")
    return formatted_str


@tool("multi_source_web_search")
def multi_source_web_search(query: str) -> str:
    """
    Multi-source Federated Web Search Tool (Internet Search).
    Use this tool when the user asks about the latest music trends, new album releases, 
    artist news, concerts, or any modern/real-world information that might not be in the static database.
    It concurrently searches multiple engines (e.g. Zhipu and Tavily) and returns aggregated summaries.
    
    Args:
        query: The natural language search query (e.g. "林俊杰最近一场演唱会在哪", "Taylor swift 2026 new song")
    """
    # 由于该 tool 处于同步环境中执行（由 langchain invoke 触发）
    # 我们需要在内部起一个 event loop 来跑异步网络请求
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    if loop.is_running():
        # 如果当前环境就在一个循环里 (如 jupyter 或 fastapi main thread)
        # 用 nest_asyncio 解决或是新开线程
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(_federated_search_async(query))
    else:
        return loop.run_until_complete(_federated_search_async(query))
