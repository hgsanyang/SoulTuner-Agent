"""
用户画像 API 路由
- GET  /api/user-profile  读取偏好
- POST /api/user-profile  保存偏好 → Neo4j + GraphZep
"""
import asyncio
import json
import logging
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from config.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


class UserProfileRequest(BaseModel):
    """用户画像保存请求"""
    user_id: str = "local_admin"
    preferred_genres: List[str] = []       # 流派偏好
    preferred_moods: List[str] = []        # 情绪偏向
    preferred_scenarios: List[str] = []    # 常听场景
    preferred_languages: List[str] = []    # 语言偏好
    free_text: str = ""                    # 自由描述


# ──── 预设选项 ────
PRESET_GENRES = [
    "摇滚", "电子", "爵士", "嘻哈", "民谣", "古典", "R&B", "后摇",
    "金属", "流行", "独立", "氛围", "Lo-fi", "朋克", "Soul"
]
PRESET_MOODS = ["开心", "悲伤", "放松", "热血", "浪漫", "治愈", "怀旧", "深沉"]
PRESET_SCENARIOS = ["学习", "跑步", "开车", "睡前", "派对", "冥想"]
PRESET_LANGUAGES = ["中文", "英文", "日语", "韩语", "纯音乐"]


@router.get("/api/user-profile")
async def get_user_profile(user_id: str = "local_admin"):
    """
    从 Neo4j User 节点读取已保存的偏好。
    """
    try:
        from retrieval.neo4j_client import get_neo4j_client
        client = get_neo4j_client()

        result = client.execute_query("""
        MATCH (u:User {id: $user_id})
        RETURN u.preferred_genres AS preferred_genres,
               u.preferred_moods AS preferred_moods,
               u.preferred_scenarios AS preferred_scenarios,
               u.preferred_languages AS preferred_languages,
               u.profile_free_text AS free_text
        """, {"user_id": user_id})

        if result and result[0]:
            row = result[0]
            return {
                "success": True,
                "profile": {
                    "user_id": user_id,
                    "preferred_genres": json.loads(row.get("preferred_genres") or "[]"),
                    "preferred_moods": json.loads(row.get("preferred_moods") or "[]"),
                    "preferred_scenarios": json.loads(row.get("preferred_scenarios") or "[]"),
                    "preferred_languages": json.loads(row.get("preferred_languages") or "[]"),
                    "free_text": row.get("free_text") or "",
                },
                "presets": {
                    "genres": PRESET_GENRES,
                    "moods": PRESET_MOODS,
                    "scenarios": PRESET_SCENARIOS,
                    "languages": PRESET_LANGUAGES,
                },
            }
        else:
            # 用户节点存在但没有偏好属性
            return {
                "success": True,
                "profile": {
                    "user_id": user_id,
                    "preferred_genres": [],
                    "preferred_moods": [],
                    "preferred_scenarios": [],
                    "preferred_languages": [],
                    "free_text": "",
                },
                "presets": {
                    "genres": PRESET_GENRES,
                    "moods": PRESET_MOODS,
                    "scenarios": PRESET_SCENARIOS,
                    "languages": PRESET_LANGUAGES,
                },
            }

    except Exception as e:
        logger.error(f"[UserProfile] 读取偏好失败: {e}")
        return {"success": False, "error": str(e)}


@router.post("/api/user-profile")
async def save_user_profile(request: UserProfileRequest):
    """
    保存偏好到 Neo4j User 节点属性 + GraphZep 长期记忆。
    """
    try:
        from retrieval.neo4j_client import get_neo4j_client
        client = get_neo4j_client()

        # ① 确保 User 节点存在
        client.execute_query("""
        MERGE (u:User {id: $user_id})
        """, {"user_id": request.user_id})

        # ② 保存偏好属性到 User 节点（JSON 字符串）
        client.execute_query("""
        MATCH (u:User {id: $user_id})
        SET u.preferred_genres = $genres,
            u.preferred_moods = $moods,
            u.preferred_scenarios = $scenarios,
            u.preferred_languages = $languages,
            u.profile_free_text = $free_text,
            u.profile_updated_at = datetime()
        """, {
            "user_id": request.user_id,
            "genres": json.dumps(request.preferred_genres, ensure_ascii=False),
            "moods": json.dumps(request.preferred_moods, ensure_ascii=False),
            "scenarios": json.dumps(request.preferred_scenarios, ensure_ascii=False),
            "languages": json.dumps(request.preferred_languages, ensure_ascii=False),
            "free_text": request.free_text,
        })

        logger.info(
            f"[UserProfile] Neo4j 偏好已更新: "
            f"genres={request.preferred_genres}, moods={request.preferred_moods}, "
            f"scenarios={request.preferred_scenarios}, langs={request.preferred_languages}"
        )

        # ③ 生成自然语言描述 → 写入 GraphZep 长期记忆
        desc_parts = []
        if request.preferred_genres:
            desc_parts.append(f"喜欢的音乐流派: {', '.join(request.preferred_genres)}")
        if request.preferred_moods:
            desc_parts.append(f"听歌时的情绪偏好: {', '.join(request.preferred_moods)}")
        if request.preferred_scenarios:
            desc_parts.append(f"常听音乐的场景: {', '.join(request.preferred_scenarios)}")
        if request.preferred_languages:
            desc_parts.append(f"语言偏好: {', '.join(request.preferred_languages)}")
        if request.free_text:
            desc_parts.append(f"额外偏好描述: {request.free_text}")

        if desc_parts:
            description = "用户主动设置了音乐画像偏好 —— " + "；".join(desc_parts)
            try:
                from services.graphzep_client import get_graphzep_client
                gz = get_graphzep_client()
                asyncio.create_task(gz.add_user_event(event_description=description))
                logger.info(f"[UserProfile] GraphZep 偏好已投递: {description[:80]}...")
            except Exception as gz_err:
                logger.warning(f"[UserProfile] GraphZep 写入失败（不影响主流程）: {gz_err}")

        return {
            "success": True,
            "message": "音乐偏好已保存",
            "saved": {
                "preferred_genres": request.preferred_genres,
                "preferred_moods": request.preferred_moods,
                "preferred_scenarios": request.preferred_scenarios,
                "preferred_languages": request.preferred_languages,
                "free_text": request.free_text,
            },
        }

    except Exception as e:
        logger.error(f"[UserProfile] 保存偏好失败: {e}")
        return {"success": False, "error": str(e)}
