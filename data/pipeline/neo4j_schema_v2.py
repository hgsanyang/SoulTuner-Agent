# ============================================================
# 【V2 升级】Neo4j 图数据库 Schema 迁移脚本
# 来源：V2 架构重构方案 → Phase 3
#
# 本脚本完成以下工作：
#   1. 为 Song 节点创建 embedding (Float[]) 属性的向量索引
#   2. 清理旧 MTG/Kaggle 数据（用户确认可以丢弃）
#   3. 验证索引创建是否成功
#
# 使用方法：
#   python data_pipeline/neo4j_schema_v2.py
#   python data_pipeline/neo4j_schema_v2.py --clean-old-data
# ============================================================

import os
import sys
import argparse
import logging

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.neo4j_client import get_neo4j_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 【V2 升级】M2D-CLAP 实测输出 768 维（非早期文档的 512 维），OMAR/MERT 输出 768 维
M2D2_EMBEDDING_DIM = 768
OMAR_EMBEDDING_DIM = 768


def create_vector_indexes():
    """
    【V2 升级】创建 Neo4j 原生向量索引
    
    创建两个索引：    1. song_m2d2_index: 基于 M2D2 跨模态向量的索引（文本搜音频用）
    2. song_omar_index: 基于 OMAR 纯音频向量的索引（相似听感搜索用）    """
    client = get_neo4j_client()
    
    # ---- 创建 M2D2 向量索引 (跨模态检索) ----
    logger.info(f"[Schema V2] 创建 M2D2 向量索引 (dim={M2D2_EMBEDDING_DIM})...")
    try:
        client.execute_query(f"""
        CREATE VECTOR INDEX song_m2d2_index IF NOT EXISTS
        FOR (s:Song) ON (s.m2d2_embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {M2D2_EMBEDDING_DIM},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """)
        logger.info("[Schema V2] ✅ song_m2d2_index 创建成功")
    except Exception as e:
        if "already exists" in str(e).lower() or "equivalent index" in str(e).lower():
            logger.info("[Schema V2] song_m2d2_index 已存在，跳过")
        else:
            logger.error(f"[Schema V2] 创建 M2D2 索引失败: {e}")
    
    # ---- 创建 OMAR 向量索引 (纯音频相似度) ----
    logger.info(f"[Schema V2] 创建 OMAR 向量索引 (dim={OMAR_EMBEDDING_DIM})...")
    try:
        client.execute_query(f"""
        CREATE VECTOR INDEX song_omar_index IF NOT EXISTS
        FOR (s:Song) ON (s.omar_embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {OMAR_EMBEDDING_DIM},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """)
        logger.info("[Schema V2] ✅ song_omar_index 创建成功")
    except Exception as e:
        if "already exists" in str(e).lower() or "equivalent index" in str(e).lower():
            logger.info("[Schema V2] song_omar_index 已存在，跳过")
        else:
            logger.error(f"[Schema V2] 创建 OMAR 索引失败: {e}")


def verify_indexes():
    """验证向量索引状态"""
    client = get_neo4j_client()
    
    result = client.execute_query("SHOW INDEXES YIELD name, type, state WHERE type = 'VECTOR'")
    
    if result:
        logger.info("[Schema V2] 当前向量索引状态")
        for idx in result:
            logger.info(f"  📌 {idx.get('name', 'N/A')} | Type: {idx.get('type', 'N/A')} | State: {idx.get('state', 'N/A')}")
    else:
        logger.warning("[Schema V2] 未发现任何向量索引。请确认 Neo4j 版本 >= 5.11")


def clean_old_data():
    """
    【V2 升级】清理旧版 MTG/Kaggle 数据
    
    用户已确认：旧数据集（MTG-Jamendo, Kaggle 等）可以全部丢弃。
    保留 User 节点和 Artist/Genre 元数据结构。
    """
    client = get_neo4j_client()
    
    # 统计当前数据量
    count_result = client.execute_query("MATCH (s:Song) RETURN count(s) AS song_count")
    song_count = count_result[0].get("song_count", 0) if count_result else 0
    logger.info(f"[Schema V2] 当前 Song 节点数量: {song_count}")
    
    if song_count == 0:
        logger.info("[Schema V2] 数据库为空，无需清理。")
        return
    
    # 删除所有 Song 节点及其关联边
    logger.info(f"[Schema V2] 正在清理 {song_count} 首旧歌曲数据...")
    client.execute_query("""
    MATCH (s:Song)
    DETACH DELETE s
    """)
    
    # 清理孤立的 Genre/Artist 节点（没有关联边的）
    client.execute_query("""
    MATCH (g:Genre) WHERE NOT (g)<-[:BELONGS_TO_GENRE]-()
    DELETE g
    """)
    client.execute_query("""
    MATCH (a:Artist) WHERE NOT (a)<-[:PERFORMED_BY]-()
    DELETE a
    """)
    
    logger.info("[Schema V2] ✅ 旧数据清理完毕！数据库已重置为空白状态。")


def add_song_with_embeddings(
    title: str,
    artist: str,
    genre: str = "",
    m2d2_embedding: list = None,
    omar_embedding: list = None,
    auto_tags: dict = None,
    preview_url: str = ""
):
    """
    【V2 升级】写入带向量的 Song 节点（示例方法，供 data_flywheel 使用）    
    创建 Song 节点并附带双模型 Embedding，同时建立与 Artist/Genre 的关联边。    """
    client = get_neo4j_client()
    
    params = {
        "title": title,
        "artist_name": artist,
        "genre_name": genre,
        "preview_url": preview_url,
        "auto_tags": auto_tags or {},
    }
    
    # 动态构建 SET 子句
    set_parts = ["s.title = $title", "s.preview_url = $preview_url", "s.auto_tags = $auto_tags"]
    
    if m2d2_embedding:
        params["m2d2_embedding"] = m2d2_embedding
        set_parts.append("s.m2d2_embedding = $m2d2_embedding")
    
    if omar_embedding:
        params["omar_embedding"] = omar_embedding
        set_parts.append("s.omar_embedding = $omar_embedding")
    
    set_clause = ", ".join(set_parts)
    
    query = f"""
    MERGE (s:Song {{title: $title, artist: $artist_name}})
    SET {set_clause}, s.updated_at = timestamp()
    
    MERGE (a:Artist {{name: $artist_name}})
    MERGE (s)-[:PERFORMED_BY]->(a)
    
    WITH s
    FOREACH (_ IN CASE WHEN $genre_name <> '' THEN [1] ELSE [] END |
        MERGE (g:Genre {{name: $genre_name}})
        MERGE (s)-[:BELONGS_TO_GENRE]->(g)
    )
    """
    
    client.execute_query(query, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neo4j V2 Schema 迁移工具")
    parser.add_argument("--clean-old-data", action="store_true", help="清理旧版 MTG/Kaggle 数据")
    parser.add_argument("--verify", action="store_true", help="仅验证索引状态")
    args = parser.parse_args()
    
    if args.verify:
        verify_indexes()
    elif args.clean_old_data:
        logger.info("=" * 60")
        logger.info("⚠️  即将清理所有旧版歌曲数据！")
        logger.info("=" * 60")
        confirm = input("确认清理？输入 YES 继续: ")
        if confirm == "YES":
            clean_old_data()
            create_vector_indexes()
            verify_indexes()
        else:
            logger.info("已取消。")
    else:
        create_vector_indexes()
        verify_indexes()
