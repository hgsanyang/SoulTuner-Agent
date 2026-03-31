"""
端到端验证：Like / Save / Dislike / Unlike / Unsave 的 Neo4j 状态一致性测试
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.user_memory import UserMemoryManager
from retrieval.neo4j_client import get_neo4j_client

UID = "local_admin"
TEST_SONG = "Test_Verification_Song"
TEST_ARTIST = "Test_Bot"

client = get_neo4j_client()
mem = UserMemoryManager()
mem.ensure_user_exists(UID)

def count_rels(rel_type: str) -> int:
    q = f"MATCH (u:User {{id: $uid}})-[r:{rel_type}]->(s:Song {{title: $t}}) RETURN count(r) AS c"
    r = client.execute_query(q, {"uid": UID, "t": TEST_SONG})
    return r[0]["c"] if r else 0

def show_state(label: str):
    likes = count_rels("LIKES")
    saves = count_rels("SAVES")
    dislikes = count_rels("DISLIKES")
    print(f"  [{label}] LIKES={likes}  SAVES={saves}  DISLIKES={dislikes}")
    return likes, saves, dislikes

def assert_state(label, expected_likes, expected_saves, expected_dislikes):
    l, s, d = show_state(label)
    ok = (l == expected_likes and s == expected_saves and d == expected_dislikes)
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {status}: expected L={expected_likes} S={expected_saves} D={expected_dislikes}")
    return ok

print("\n" + "="*60)
print("🧪 Neo4j 行为事件状态一致性验证")
print("="*60)

# 清理测试歌曲
client.execute_query(
    "MATCH (u:User {id: $uid})-[r]->(s:Song {title: $t}) DELETE r",
    {"uid": UID, "t": TEST_SONG}
)

results = []

# Test 1: 初始状态
print("\n① 初始状态（应全部为 0）")
results.append(assert_state("初始", 0, 0, 0))

# Test 2: 点赞
print("\n② 点赞 → LIKES=1")
mem.record_liked_song(UID, TEST_SONG, TEST_ARTIST)
results.append(assert_state("点赞后", 1, 0, 0))

# Test 3: 收藏
print("\n③ 再收藏 → LIKES=1, SAVES=1")
mem.record_saved_song(UID, TEST_SONG, TEST_ARTIST)
results.append(assert_state("收藏后", 1, 1, 0))

# Test 4: 不喜欢（应同时清除 LIKES + SAVES）
print("\n④ 不喜欢 → LIKES=0, SAVES=0, DISLIKES=1")
mem.record_dislike(UID, TEST_SONG, TEST_ARTIST)
results.append(assert_state("不喜欢后", 0, 0, 1))

# Test 5: 取消不喜欢（通过 DELETE API 模拟）
print("\n⑤ 撤销不喜欢 → 全部为 0")
client.execute_query(
    "MATCH (u:User {id: $uid})-[r:DISLIKES]->(s:Song {title: $t}) DELETE r",
    {"uid": UID, "t": TEST_SONG}
)
results.append(assert_state("撤销不喜欢后", 0, 0, 0))

# Test 6: 再次点赞，然后取消点赞
print("\n⑥ 点赞 → 取消点赞（unlike）")
mem.record_liked_song(UID, TEST_SONG, TEST_ARTIST)
mem.remove_like(UID, TEST_SONG, TEST_ARTIST)
results.append(assert_state("取消点赞后", 0, 0, 0))

# Test 7: 收藏 → 取消收藏
print("\n⑦ 收藏 → 取消收藏（unsave）")
mem.record_saved_song(UID, TEST_SONG, TEST_ARTIST)
mem.remove_save(UID, TEST_SONG, TEST_ARTIST)
results.append(assert_state("取消收藏后", 0, 0, 0))

# 清理
client.execute_query(
    "MATCH (s:Song {title: $t}) DETACH DELETE s", {"t": TEST_SONG}
)

print("\n" + "="*60)
passed = sum(results)
total = len(results)
print(f"📊 测试结果: {passed}/{total} 通过")
if passed == total:
    print("🎉 全部通过！状态一致性验证成功。")
else:
    print("⚠️ 存在失败用例，请检查上方输出。")
print("="*60 + "\n")
