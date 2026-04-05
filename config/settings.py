from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from pathlib import Path
import os

class GlobalSettings(BaseSettings):
    """
    全局配置中心
    ============
    配置分工原则：
      .env              → API 密钥、机器相关服务地址、模型部署选择（与 Key 绑定的）
      config/settings.py → 所有功能开关、调参旋钮、检索参数（直接改 default= 即可）
      前端设置面板       → 运行时即时调整（无需重启，关闭面板丢弃）

    修改 settings.py 后重启后端生效：python startup_all.py --no-web
    使用方式：
      from config.settings import settings
      settings.reranker_enabled
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

    # --- 主 LLM ---
    # ★ 这里的 default= 就是启动时的实际值，可直接改；前端⚙️ 设置面板可运行时覆盖
    # 可选提供商: siliconflow | dashscope | google | sglang | vllm | ollama
    llm_default_provider: str = Field(
        default="siliconflow",
        validation_alias="MAIN_LLM_PROVIDER",
        description="主 LLM 提供商（siliconflow / volcengine / dashscope / google / sglang / vllm / ollama）",
    )
    llm_default_model: str = Field(
        default="deepseek-ai/DeepSeek-V3.2",
        validation_alias="MODEL_NAME",
        description="主 LLM 模型名称（云端 API 时为模型全名；本地时为 SGLang 部署的模型标识）",
    )

    # --- 意图分析专用 LLM（Planner）---
    # ★ 切换本地 vs API 就改这两行，或直接在前端⚙️ 设置面板修改
    #
    # 用 API 大模型（融合版）——一次调用输出: 意图 + 实体 + 标签 + 内联 HyDE
    #   intent_llm_provider = "siliconflow"
    #   intent_llm_model    = "deepseek-ai/DeepSeek-V3.2"
    #
    # 用本地小模型（精简版）——只输出: 意图 + 实体，HyDE 由下游独立生成
    #   intent_llm_provider = "sglang"
    #   intent_llm_model    = "local-planner-qwen3-4b-fp8"
    intent_llm_provider: str = Field(
        default="siliconflow",
        validation_alias="INTENT_LLM_PROVIDER",
        description="意图分析专用 LLM 提供商（siliconflow / sglang 等）",
    )
    intent_llm_model: str = Field(
        default="deepseek-ai/DeepSeek-V3.2",
        validation_alias="INTENT_LLM_MODEL",
        description="意图分析专用模型名（空则复用主模型）",
    )

    # --- HyDE 声学描述生成专用 LLM（本地模式专用；API 模式已内联，此字段留空）---
    hyde_llm_provider: str = Field(
        default="",
        validation_alias="HYDE_LLM_PROVIDER",
        description="HyDE 声学描述生成专用 LLM 提供商（空则复用主模型）",
    )
    hyde_llm_model: str = Field(
        default="",
        validation_alias="HYDE_LLM_MODEL",
        description="HyDE 声学描述生成专用模型名（空则复用主模型）",
    )

    finetuned_model_path: str = Field(
        default="",
        validation_alias="FINETUNED_MODEL_PATH",
        description="本地微调模型路径（vLLM 加载用）",
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
    sglang_api_key: str = Field("fake_key", validation_alias="SGLANG_API_KEY")
    sglang_base_url: str = Field("http://localhost:8000/v1", validation_alias="SGLANG_BASE_URL")

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
    graphzep_base_url: str = Field("http://localhost:8350", validation_alias="GRAPHZEP_BASE_URL")

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
    mtg_audio_dir: str = Field(
        default="data/mtg_sample/audio",
        validation_alias="MTG_AUDIO_DIR",
        description="MTG 数据集音频目录",
    )

    # ================================================================
    # 6. 检索 & 推荐参数（★ 核心调参区）
    # ================================================================
    semantic_search_limit: int = Field(
        default=15,
        description="语义向量搜索默认返回条数（Neo4j KNN）",
    )
    graph_search_limit: int = Field(
        default=15,
        description="GraphRAG 图谱搜索默认返回条数",
    )
    hybrid_retrieval_limit: int = Field(
        default=15,
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
    # 6b. 精排管线参数（双锚精排 + 图距离 + 多样性）
    # ================================================================

    # --- 双锚精排（M2D-CLAP 语义锚 + OMAR-RQ 声学锚）---
    dual_anchor_weight_semantic: float = Field(
        default=0.6,
        description="双锚精排中 M2D-CLAP 语义锚（text embedding cosine）的权重",
    )
    dual_anchor_weight_acoustic: float = Field(
        default=0.4,
        description="双锚精排中 OMAR-RQ 声学锚（centroid cosine）的权重",
    )

    # --- Graph Affinity（图距离 + 用户画像 Jaccard）---
    graph_affinity_enabled: bool = Field(
        default=True,
        description="是否启用 Neo4j 图距离亲和力 + 用户画像 Jaccard 加权",
    )
    graph_affinity_weight: float = Field(
        default=0.15,
        description="Graph Affinity 分数在精排前微调排序中的权重",
    )
    graph_affinity_max_hops: int = Field(
        default=4,
        description="图距离计算最大跳数",
    )

    # --- Artist 多样性 & MMR ---
    max_songs_per_artist: int = Field(
        default=2,
        description="多样性过滤：每个艺术家最多占的歌曲数",
    )
    mmr_lambda: float = Field(
        default=0.7,
        description="MMR 多样性重排序中 relevance vs diversity 的平衡系数（越高越偏向相关性）",
    )

    # --- recommend_by_favorites 智能推荐 ---
    favorites_seed_limit: int = Field(
        default=5,
        description="recommend_by_favorites: 种子展示数量（来自用户收藏）",
    )
    favorites_discovery_limit: int = Field(
        default=10,
        description="recommend_by_favorites: 新歌发现数量（来自向量检索扩展）",
    )

    # ================================================================
    # 6c. Cross-Encoder 精排层
    # ================================================================
    # ★ 直接修改 default= 来开关此功能，不需要在 .env 设置
    # True  = 启用（需要加载 bge-reranker-v2-m3，占用显存，速度变慢）
    # False = 关闭（推荐，RRF + Graph Affinity 已足够精准）
    reranker_enabled: bool = Field(
        default=False,
        description="是否启用 Cross-Encoder 精排层（bge-reranker-v2-m3）",
    )
    reranker_model_name: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Cross-Encoder 精排模型名称",
    )
    reranker_top_k: int = Field(
        default=10,
        description="精排后保留的 Top K 结果",
    )
    reranker_device: str = Field(
        default="cuda",
        description="Cross-Encoder 推理设备（cpu / cuda）",
    )
    reranker_batch_size: int = Field(
        default=16,
        description="Cross-Encoder 批推理大小",
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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # 忽略无法匹配的环境变量


# 暴露单例给全量代码使用
settings = GlobalSettings()
