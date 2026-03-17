"""
一次性迁移脚本：为所有 Song 节点批量补充 cover_url 和 lrc_url

原理：  - 每个 Song 节点已有 audio_url（如 /static/audio/清白之年 - 朴树.mp3）  - cover 文件命名规则：{basename}_cover.jpg → /static/covers/{basename}_cover.jpg
  - lrc 文件命名规则：{basename}.lrc → /static/lyrics/{basename}.lrc
  - 本脚本用一条 Cypher 批量更新，秒级完成，无需重新入库

用法：  python data_pipeline/migrate_cover_lrc_urls.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.neo4j_client import get_neo4j_client


def migrate():
    client = get_neo4j_client()

    # 先检查有多少 Song 节点缺少 cover_url
    check_query = """
    MATCH (s:Song)
    WHERE s.audio_url IS NOT NULL
    RETURN count(s) AS total,
           count(s.cover_url) AS has_cover,
           count(s.lrc_url) AS has_lrc
    """
    result = client.execute_query(check_query)
    if result:
        row = result[0]
        total = row.get("total", 0)
        has_cover = row.get("has_cover", 0)
        has_lrc = row.get("has_lrc", 0)
        print(f"📊 当前状态: 共 {total} 首歌, 已有 cover_url: {has_cover}, 已有 lrc_url: {has_lrc}")

        # 检查已有的 cover_url 是否为空字符串
        empty_check = """
        MATCH (s:Song)
        WHERE s.cover_url IS NOT NULL AND s.cover_url <> '' AND s.cover_url <> '/static/covers/_cover.jpg'
        RETURN count(s) AS valid_cover
        """
        empty_result = client.execute_query(empty_check)
        if empty_result:
            valid = empty_result[0].get("valid_cover", 0)
            print(f"   其中有效（非空）cover_url: {valid}")

    # 核心迁移 Cypher：从 audio_url 推导 cover_url 和 lrc_url
    # audio_url 格式: /static/audio/清白之年 - 朴树.mp3
    # 如 basename: 清白之年 - 朴树
    # 则 cover_url: /static/covers/清白之年 - 朴树_cover.jpg
    # 则 lrc_url: /static/lyrics/清白之年 - 朴树.lrc
    migrate_query = """
    MATCH (s:Song)
    WHERE s.audio_url IS NOT NULL AND s.audio_url STARTS WITH '/static/audio/'
    WITH s,
         substring(s.audio_url, 14) AS filename
    WITH s,
         left(filename, size(filename) - size(last(split(filename, '.')))) AS basename_with_dot
    WITH s,
         left(basename_with_dot, size(basename_with_dot) - 1) AS basename
    SET s.cover_url = '/static/covers/' + basename + '_cover.jpg',
        s.lrc_url = '/static/lyrics/' + basename + '.lrc'
    RETURN count(s) AS updated
    """

    print("🔄 正在批量更新 cover_url 和 lrc_url ...")
    update_result = client.execute_query(migrate_query)
    if update_result:
        updated = update_result[0].get("updated", 0)
        print(f"✅ 成功更新 {updated} 首歌的 cover_url 和 lrc_url")

    # 验证结果: 随机取 3 首查看
    sample_query = """
    MATCH (s:Song)
    WHERE s.cover_url IS NOT NULL AND s.cover_url <> ''
    RETURN s.title AS title, s.audio_url AS audio, s.cover_url AS cover, s.lrc_url AS lrc
    LIMIT 3
    """
    samples = client.execute_query(sample_query)
    if samples:
        print("\n📋 验证样本:")
        for row in samples:
            print(f"  🎵 {row.get('title', '?')}")
            print(f"     audio:  {row.get('audio', '')}")
            print(f"     cover:  {row.get('cover', '')}")
            print(f"     lrc:    {row.get('lrc', '')}")


if __name__ == "__main__":
    migrate()
