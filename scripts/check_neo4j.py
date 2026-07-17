"""检查 Running Up That Hill 的完整属性和关联"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.neo4j_client import get_neo4j_client

client = get_neo4j_client()

# 1. 完整属性
print("=" * 60)
print("完整 Song 节点属性:")
print("=" * 60)
results = client.execute_query(
    "MATCH (s:Song {title: 'Running Up That Hill'}) RETURN s",
    {}
)
for r in results:
    s = r.get('s', {})
    for k, v in sorted(s.items()):
        if k.endswith('_embedding'):
            print(f"  {k}: <{len(v)} dims>" if isinstance(v, list) else f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

# 2. 关联关系
print("\n" + "=" * 60)
print("关联的节点和关系:")
print("=" * 60)
results2 = client.execute_query(
    """MATCH (s:Song {title: 'Running Up That Hill'})-[r]->(n)
    RETURN type(r) AS rel_type, labels(n) AS node_labels,
           CASE WHEN 'name' IN keys(n) THEN n.name ELSE n.title END AS node_name
    """,
    {}
)
if results2:
    for r in results2:
        print(f"  -[{r['rel_type']}]-> {r['node_labels']}: {r['node_name']}")
else:
    print("  ❌ 没有任何关联关系 (缺少 PERFORMED_BY, HAS_MOOD, HAS_THEME 等)")

# 3. 检查是否有嵌入向量
print("\n" + "=" * 60)
print("向量状态:")
print("=" * 60)
results3 = client.execute_query(
    """MATCH (s:Song {title: 'Running Up That Hill'})
    RETURN s.m2d2_embedding IS NOT NULL AS has_m2d2,
           s.omar_embedding IS NOT NULL AS has_omar,
           s.vibe AS vibe
    """,
    {}
)
for r in results3:
    print(f"  M2D-CLAP 向量: {'✅ 已有' if r.get('has_m2d2') else '❌ 缺失'}")
    print(f"  OMAR 向量:     {'✅ 已有' if r.get('has_omar') else '❌ 缺失'}")
    print(f"  Vibe 标签:     {r.get('vibe') or '❌ 缺失'}")
