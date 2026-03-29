"""
将 gemini_result.json 的标签批量写入 Neo4j 图谱关系。

运行方式：
  conda activate music_agent
  python ingest_tags_to_neo4j.py

效果：
  - 为每首歌创建 Mood / Genre / Theme / Scenario 节点（MERGE 去重）
  - 建立关系：HAS_MOOD, BELONGS_TO_GENRE, HAS_THEME, FITS_SCENARIO
  - 更新 Song 节点的 language / region / vibe 属性
"""

import sys, json, time
sys.path.insert(0, r"c:\Users\sanyang\sanyangworkspace\music_recommendation\Muisc-Research")

from retrieval.neo4j_client import get_neo4j_client


def main():
    print("=" * 60)
    print("  📀 gemini_result.json → Neo4j 标签入库")
    print("=" * 60)

    client = get_neo4j_client()
    data = json.load(open(
        r"data\pipeline\gemini_prompts\gemini_result.json",
        encoding="utf-8",
    ))
    print(f"✅ 加载 {len(data)} 条标注数据\n")

    # 统计计数
    stats = {
        "matched": 0,
        "unmatched": 0,
        "mood_created": 0,
        "genre_created": 0,
        "theme_created": 0,
        "scenario_created": 0,
        "props_updated": 0,
    }

    t0 = time.time()

    for i, item in enumerate(data):
        fn = item.get("filename", "")
        # 从 filename 提取 title: "The World - Richard Cheese，xxx.lrc" → "The World"
        title = fn.replace(".lrc", "").split(" - ", 1)[0].strip()
        if not title:
            stats["unmatched"] += 1
            continue

        # 检查 Song 是否存在
        check = client.execute_query(
            "MATCH (s:Song {title: $title}) RETURN s.title AS t LIMIT 1",
            {"title": title},
        )
        if not check:
            stats["unmatched"] += 1
            continue

        stats["matched"] += 1

        # 1. 创建 Mood 关系
        for mood in item.get("moods", []):
            if mood:
                client.execute_query(
                    "MATCH (s:Song {title: $title}) "
                    "MERGE (m:Mood {name: $mood}) "
                    "MERGE (s)-[:HAS_MOOD]->(m)",
                    {"title": title, "mood": mood},
                )
                stats["mood_created"] += 1

        # 2. 创建 Genre 关系
        for genre in item.get("genre", []):
            if genre:
                client.execute_query(
                    "MATCH (s:Song {title: $title}) "
                    "MERGE (g:Genre {name: $genre}) "
                    "MERGE (s)-[:BELONGS_TO_GENRE]->(g)",
                    {"title": title, "genre": genre},
                )
                stats["genre_created"] += 1

        # 3. 创建 Theme 关系
        for theme in item.get("themes", []):
            if theme:
                client.execute_query(
                    "MATCH (s:Song {title: $title}) "
                    "MERGE (t:Theme {name: $theme}) "
                    "MERGE (s)-[:HAS_THEME]->(t)",
                    {"title": title, "theme": theme},
                )
                stats["theme_created"] += 1

        # 4. 创建 Scenario 关系
        for scenario in item.get("scenarios", []):
            if scenario:
                client.execute_query(
                    "MATCH (s:Song {title: $title}) "
                    "MERGE (sc:Scenario {name: $scenario}) "
                    "MERGE (s)-[:FITS_SCENARIO]->(sc)",
                    {"title": title, "scenario": scenario},
                )
                stats["scenario_created"] += 1

        # 5. 更新 Song 节点属性
        props = {}
        if item.get("language"):
            props["language"] = item["language"]
        if item.get("region"):
            props["region"] = item["region"]
        if item.get("vibe"):
            props["vibe"] = item["vibe"]

        if props:
            set_clauses = ", ".join([f"s.{k} = ${k}" for k in props])
            client.execute_query(
                f"MATCH (s:Song {{title: $title}}) SET {set_clauses}",
                {"title": title, **props},
            )
            stats["props_updated"] += 1

        # 进度
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(data)}] {elapsed:.1f}s | matched={stats['matched']}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  ✅ 完成！耗时 {elapsed:.1f}s")
    print(f"  匹配成功: {stats['matched']} / {len(data)}")
    print(f"  未匹配:   {stats['unmatched']}")
    print(f"  Mood 关系:     {stats['mood_created']}")
    print(f"  Genre 关系:    {stats['genre_created']}")
    print(f"  Theme 关系:    {stats['theme_created']}")
    print(f"  Scenario 关系: {stats['scenario_created']}")
    print(f"  属性更新:      {stats['props_updated']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
