from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from pathlib import Path
import os

class GlobalSettings(BaseSettings):
    """
    全局配置中心
    ============
    所有可调参数通过此文件统一管理。
    支持从 .env 文件或环境变量自动映射。

    使用方式：
      from config.settings import settings
      settings.semantic_search_limit
    """

    # ================================================================
    # 1. LLM API Keys & 端点
    # ================================================================
    siliconflow_api_key: str = Field("", validation_alias="SILICONFLOW_API_KEY")
    siliconflow_base_url: str = Field("https://api.siliconflow.cn/v1", validation_alias="SILICONFLOW_BASE_URL")
    zhipu_api_key: str = Field("", validation_alias="ZHIPU_API_KEY")
    dashscope_api_key: str = Field("", validation_alias="DASHSCOPE_API_KEY")
    google_api_key: str = Field("", validation_alias="GOOGLE_API_KEY")

    # ================================================================
    # 2. LLM 推理参数
    # ================================================================
    llm_temperature: float = Field(default=0.7, description="LLM 默认温度（0-1，越高越随机）")
    llm_default_provider: str = Field(default="siliconflow", description="默认 LLM 提供商")
    llm_default_model: str = Field(
        default="deepseek-ai/DeepSeek-V3.2",
        validation_alias="MODEL_NAME",
        description="默认模型名称",
    )
    intent_llm_provider: str = Field(
        default="siliconflow",
        validation_alias="INTENT_LLM_PROVIDER",
        description="意图分析专用 LLM 提供商（可独立于主模型配置更快/更小的模型）",
    )
    intent_llm_model: str = Field(
        default="",
        validation_alias="INTENT_LLM_MODEL",
        description="意图分析专用模型名（空则用该 provider 默认模型）",
    )
    llm_timeout: int = Field(
        default=80,
        validation_alias="LLM_TIMEOUT",
        description="LLM API 调用超时（秒），防止请求无限挂起",
    )

    # ================================================================
    # 3. 本地大模型环境
    # ================================================================
    ollama_api_key: str = Field("fake_key", validation_alias="OLLAMA_API_KEY")
    ollama_base_url: str = Field("http://localhost:11434/v1", validation_alias="OLLAMA_BASE_URL")
    vllm_api_key: str = Field("fake_key", validation_alias="VLLM_API_KEY")
    vllm_base_url: str = Field("http://localhost:8000/v1", validation_alias="VLLM_BASE_URL")

    # ================================================================
    # 4. 服务端口 & URL
    # ================================================================
    api_base_url: str = Field(
        default="http://localhost:8501",
        validation_alias="MUSIC_API_BASE_URL",
        description="后端 API 基地址，前端和工具引用音频/封面地址时使用",
    )
    api_port: int = Field(default=8501, description="后端 API 服务端口")
    frontend_port: int = Field(default=3000, description="前端 dev server 端口")
    netease_api_base: str = Field("http://localhost:3000", validation_alias="NETEASE_API_BASE")
    searxng_base_url: str = Field("http://localhost:8888", validation_alias="SEARXNG_BASE_URL")
    graphzep_base_url: str = Field("http://localhost:3100", validation_alias="GRAPHZEP_BASE_URL")

    # ================================================================
    # 5. 路径配置
    # ================================================================
    audio_data_dir: str = Field(
        default=str(Path("data/processed_audio/audio")),
        validation_alias="MUSIC_AUDIO_DATA_DIR",
    )
    online_acquired_dir: str = Field(
        default="data/online_acquired",
        description="联网获取音乐的存储根目录",
    )

    # ================================================================
    # 6. 检索 & 推荐参数（★ 核心调参区）
    # ================================================================
    semantic_search_limit: int = Field(
        default=12,
        description="语义向量搜索默认返回条数（Neo4j KNN）",
    )
    graph_search_limit: int = Field(
        default=12,
        description="GraphRAG 图谱搜索默认返回条数",
    )
    hybrid_retrieval_limit: int = Field(
        default=12,
        description="混合检索合并后最终返回条数（传给 LLM 推荐解释）",
    )
    web_search_max_results: int = Field(
        default=5,
        description="联网搜索每个引擎最大返回条数",
    )
    netease_search_limit: int = Field(
        default=3,
        description="网易云 API 搜歌时每次搜索返回候选数",
    )
    user_preference_limit: int = Field(
        default=20,
        description="从 Neo4j 读取用户偏好时的最大条数",
    )
    enhanced_recommend_limit: int = Field(
        default=5,
        description="增强推荐节点返回条数",
    )

    # ================================================================
    # 7. Agent 内存与上下文
    # ================================================================
    memory_retain_rounds: int = Field(default=5, description="上下文管理器保留的最近聊天轮数")
    max_context_tokens: int = Field(default=4000, description="允许的最大 Token 数")
    default_user_id: str = Field(default="local_admin", description="默认用户 ID（单用户模式）")

    # ================================================================
    # 8. 网络请求超时（秒）
    # ================================================================
    web_search_timeout: int = Field(default=12, description="联网搜索 HTTP 超时（智谱/Tavily）")
    searxng_timeout: int = Field(default=8, description="SearxNG 搜索超时")
    netease_api_timeout: int = Field(default=10, description="网易云 API 请求超时")
    audio_download_timeout: int = Field(default=60, description="音频文件下载超时")

    # ================================================================
    # 9. 兼容旧字段（保留但不推荐直接使用）
    # ================================================================
    default_search_limit: int = Field(default=10, description="（旧）检索默认条数")
    default_recommend_limit: int = Field(default=5, description="（旧）推荐歌曲时的默认返回条数")
    tavily_api_key: str = Field("", validation_alias="TAVILY_API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # 忽略无法匹配的环境变量


# 暴露单例给全量代码使用
settings = GlobalSettings()
