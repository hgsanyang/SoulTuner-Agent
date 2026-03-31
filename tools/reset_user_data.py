"""
🧹 用户行为数据清理工具
用途：清除 Neo4j 中 local_admin 用户的点赞/收藏/不喜欢关系 + 用户画像偏好。
运行: conda activate music_agent && python tools/reset_user_data.py

支持参数:
  --all         清理所有（点赞+收藏+不喜欢+跳过+画像），默认行为
  --likes       仅清理点赞(LIKES)
  --saves       仅清理收藏(SAVES)
  --dislikes    仅清理不喜欢(DISLIKES)
  --skips       仅清理跳过记录(SKIPPED)
  --profile     仅重置用户画像偏好
  --dry-run     仅预览不执行
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from retrieval.neo4j_client import get_neo4j_client
from config.logging_config import get_logger

logger = get_logger(__name__)
USER_ID = "local_admin"


def count_relationships(client, rel_type: str) -> int:
    """统计某类关系的数量"""
    query = f"MATCH (u:User {{id: $uid}})-[r:{rel_type}]->() RETURN count(r) AS cnt"
    result = client.execute_query(query, {"uid": USER_ID})
    return result[0]["cnt"] if result else 0


def delete_relationships(client, rel_type: str, dry_run: bool = False) -> int:
    """删除某类关系"""
    cnt = count_relationships(client, rel_type)
    if cnt == 0:
        print(f"  ✓ {rel_type}: 无记录，跳过")
        return 0
    if dry_run:
        print(f"  🔍 {rel_type}: 发现 {cnt} 条记录（dry-run，不执行删除）")
        return cnt
    query = f"MATCH (u:User {{id: $uid}})-[r:{rel_type}]->() DELETE r"
    client.execute_query(query, {"uid": USER_ID})
    print(f"  🗑️  {rel_type}: 已删除 {cnt} 条记录")
    return cnt


def reset_profile(client, dry_run: bool = False):
    """重置用户画像偏好属性"""
    query = """
    MATCH (u:User {id: $uid})
    RETURN u.favorite_genres AS genres, u.mood_tendency AS mood,
           u.preferred_scenarios AS scenarios, u.language_preferences AS langs
    """
    result = client.execute_query(query, {"uid": USER_ID})
    if not result:
        print("  ✓ 用户节点不存在，跳过")
        return

    profile = result[0]
    has_profile = any(v for v in profile.values() if v)
    if not has_profile:
        print("  ✓ 用户画像: 已是空白，跳过")
        return

    if dry_run:
        print(f"  🔍 用户画像: 当前偏好 = {dict(profile)}（dry-run，不执行清除）")
        return

    clear_query = """
    MATCH (u:User {id: $uid})
    REMOVE u.favorite_genres, u.mood_tendency,
           u.preferred_scenarios, u.language_preferences,
           u.other_preferences
    """
    client.execute_query(clear_query, {"uid": USER_ID})
    print(f"  🗑️  用户画像: 已清除所有偏好设置")


def main():
    parser = argparse.ArgumentParser(description="清理 Neo4j 用户行为数据")
    parser.add_argument("--all", action="store_true", help="清理全部（默认）")
    parser.add_argument("--likes", action="store_true", help="仅清理点赞")
    parser.add_argument("--saves", action="store_true", help="仅清理收藏")
    parser.add_argument("--dislikes", action="store_true", help="仅清理不喜欢")
    parser.add_argument("--skips", action="store_true", help="仅清理跳过")
    parser.add_argument("--profile", action="store_true", help="仅重置画像")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不执行")
    args = parser.parse_args()

    # 如果没指定任何具体选项，默认 --all
    specific = args.likes or args.saves or args.dislikes or args.skips or args.profile
    if not specific:
        args.all = True

    client = get_neo4j_client()
    dry_label = " [DRY-RUN 预览模式]" if args.dry_run else ""
    print(f"\n{'='*50}")
    print(f"🧹 用户数据清理工具{dry_label}")
    print(f"   目标用户: {USER_ID}")
    print(f"{'='*50}\n")

    total = 0
    if args.all or args.likes:
        total += delete_relationships(client, "LIKES", args.dry_run)
    if args.all or args.saves:
        total += delete_relationships(client, "SAVES", args.dry_run)
    if args.all or args.dislikes:
        total += delete_relationships(client, "DISLIKES", args.dry_run)
    if args.all or args.skips:
        total += delete_relationships(client, "SKIPPED", args.dry_run)
        delete_relationships(client, "LISTENED_TO", args.dry_run)
    if args.all or args.profile:
        reset_profile(client, args.dry_run)

    print(f"\n{'='*50}")
    if args.dry_run:
        print(f"📋 预览完成，共发现 {total} 条待清理记录")
        print(f"   去掉 --dry-run 参数可真正执行清理")
    else:
        print(f"✅ 清理完成，共删除 {total} 条记录")
        print(f"   前端 localStorage 需手动清除（浏览器 F12 → Application → Clear site data）")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
