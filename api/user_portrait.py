"""
用户画像 API 路由
- GET  /api/user-portrait      获取当前动态画像
- POST /api/user-portrait/refresh  手动触发画像刷新
"""
import logging
from typing import Optional

from fastapi import APIRouter
from config.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/api/user-portrait")
async def get_user_portrait(user_id: str = "local_admin"):
    """
    获取当前用户的动态画像。
    优先返回内存缓存，其次从 Neo4j 加载，最后提示无画像。
    """
    try:
        from services.profile_synthesizer import get_profile_synthesizer
        synth = get_profile_synthesizer(user_id)

        # 优先读缓存
        portrait = synth.get_cached_portrait()
        source = "cache"

        # 缓存无 → 从 Neo4j 加载
        if portrait is None:
            portrait = await synth.load_portrait()
            source = "neo4j"

        if portrait is None:
            return {
                "success": True,
                "portrait": None,
                "source": "none",
                "message": "尚未生成用户画像，请对话几轮后刷新",
            }

        return {
            "success": True,
            "portrait": portrait.model_dump(),
            "source": source,
            "summary": portrait.one_line_summary,
        }
    except Exception as e:
        logger.error(f"[UserPortrait] 获取画像失败: {e}")
        return {"success": False, "error": str(e)}


@router.post("/api/user-portrait/refresh")
async def refresh_user_portrait(user_id: str = "local_admin"):
    """
    手动触发画像刷新。
    收集 GraphZep + Neo4j 数据 → LLM 聚合 → 保存 → 返回新画像。
    """
    try:
        from services.profile_synthesizer import trigger_portrait_refresh
        portrait = await trigger_portrait_refresh(user_id)

        if portrait is None:
            return {
                "success": False,
                "message": "画像刷新失败",
            }

        return {
            "success": True,
            "portrait": portrait.model_dump(),
            "summary": portrait.one_line_summary,
            "confidence": portrait.confidence,
            "message": "画像已刷新",
        }
    except Exception as e:
        logger.error(f"[UserPortrait] 画像刷新失败: {e}")
        return {"success": False, "error": str(e)}
