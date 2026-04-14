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

    # --- 上下文压缩专用 LLM（GSSC Compressor）---
    # ★ 用于多轮对话时压缩过长的历史记录，建议用快速稳定的模型（如 DeepSeek-V3.2）
    compress_llm_provider: str = Field(
        default="",
        validation_alias="COMPRESS_LLM_PROVIDER",
        description="上下文压缩专用 LLM 提供商（空则复用主模型）",
    )
    compress_llm_model: str = Field(
        default="",
        validation_alias="COMPRESS_LLM_MODEL",
        description="上下文压缩专用模型名（空则复用主模型）",
    )

    # --- 解释生成专用 LLM（generate_explanation 节点）---
    # ★ 负责生成推荐理由和最终呈现给用户的解释文本（流式输出）
    # ★ 建议使用表达能力强的模型（如 DeepSeek-V3.2 / Qwen3.5-35B）
    explain_llm_provider: str = Field(
        default="",
        validation_alias="EXPLAIN_LLM_PROVIDER",
        description="解释生成专用 LLM 提供商（空则复用主模型）",
    )
    explain_llm_model: str = Field(
        default="",
        validation_alias="EXPLAIN_LLM_MODEL",
        description="解释生成专用模型名（空则复用主模型）",
    )

    # --- 上下文窗口预算 ---
    context_total_budget: int = Field(
        default=8000,
        description="GSSC 上下文总 Token 预算（不含 system prompt），越大保留越多历史，但增加 LLM 成本",
    )

    # --- 意图分析最大输出 Token ---
    intent_max_tokens: int = Field(
        default=2048,
        description="意图分析 LLM 最大输出 Token 数（json_mode 下 ~200 tokens 足够，2048 留足余量）",
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
        default=24,
        description="仅向量检索时的返回条数（Neo4j KNN）",
    )
    graph_search_limit: int = Field(
        default=24,
        description="仅图谱检索时的返回条数（GraphRAG）",
    )
    mixed_retrieval_limit: int = Field(
        default=24,
        description="混合检索时每个引擎各返回的条数（双引擎时，各自返回此数量）",
    )
    hybrid_retrieval_limit: int = Field(
        default=15,
        description="FinalCut 最终输出条数（传给 LLM 推荐解释）",
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
    # 6b. 精排管线参数（粗排 + 三锚精排 + 探索 + 多样性）
    # ================================================================

    # --- 粗排 & 探索（Graph Affinity + Thompson Sampling）---
    graph_affinity_enabled: bool = Field(
        default=True,
        description="是否启用 Graph Affinity 粗排（图距离+Jaccard 筛选）+ TS 探索槽",
    )
    graph_affinity_max_hops: int = Field(
        default=4,
        description="图距离计算最大跳数",
    )
    coarse_cut_ratio: float = Field(
        default=0.65,
        description="粗排保留比例（如 0.65 = 保留 65% 的候选歌曲）",
    )
    exploration_ratio: float = Field(
        default=0.15,
        description="小众歌曲曝光度（从尾部按此比例捞回冷门歌进入精排）",
    )

    # --- 三锚精排权重（语义 + 声学 + 个性化，自动归一化）---
    tri_anchor_w_semantic: float = Field(
        default=0.45,
        description="三锚精排: M2D-CLAP 语义相关性权重",
    )
    tri_anchor_w_acoustic: float = Field(
        default=0.30,
        description="三锚精排: OMAR-RQ 声学风格一致性权重",
    )
    tri_anchor_w_personal: float = Field(
        default=0.25,
        description="三锚精排: 个性化偏好（图距离+Jaccard）权重",
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
    max_context_tokens: int = Field(default=8000, description="允许的最大 Token 数")
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


# ---- 用户设置持久化（JSON 文件） ----
import json as _json

_USER_SETTINGS_FILE = Path(__file__).parent / "user_settings.json"

def _load_user_overrides(s: GlobalSettings) -> list[str]:
    """启动时从 user_settings.json 加载用户上次保存的设置覆盖"""
    if not _USER_SETTINGS_FILE.exists():
        return []
    try:
        with open(_USER_SETTINGS_FILE, "r", encoding="utf-8") as f:
            overrides = _json.load(f)
        applied = []
        for key, val in overrides.items():
            if hasattr(s, key):
                setattr(s, key, val)
                applied.append(key)
        return applied
    except Exception:
        return []

def save_user_settings(s: GlobalSettings, keys: list[str] | None = None):
    """将指定字段（或全部非敏感字段）持久化到 JSON 文件"""
    # 可持久化的字段白名单（排除 API key 等敏感信息）
    _PERSISTABLE = {
        "llm_default_provider", "llm_default_model",
        "intent_llm_provider", "intent_llm_model",
        "hyde_llm_provider", "hyde_llm_model",
        "compress_llm_provider", "compress_llm_model",
        "intent_max_tokens",
        "context_total_budget",
        "finetuned_model_path", "llm_timeout", "llm_temperature",
        "audio_data_dir", "mtg_audio_dir", "online_acquired_dir",
        "graph_search_limit", "semantic_search_limit",
        "mixed_retrieval_limit", "hybrid_retrieval_limit", "web_search_max_results",
        "graph_affinity_enabled", "graph_affinity_max_hops",
        "coarse_cut_ratio", "exploration_ratio",
        "tri_anchor_w_semantic", "tri_anchor_w_acoustic", "tri_anchor_w_personal",
        "max_songs_per_artist", "mmr_lambda",
        "memory_retain_rounds", "default_user_id",
    }
    # 读取已有文件
    existing = {}
    if _USER_SETTINGS_FILE.exists():
        try:
            with open(_USER_SETTINGS_FILE, "r", encoding="utf-8") as f:
                existing = _json.load(f)
        except Exception:
            pass
    # 合并更新
    fields_to_save = keys if keys else list(_PERSISTABLE)
    for key in fields_to_save:
        if key in _PERSISTABLE and hasattr(s, key):
            existing[key] = getattr(s, key)
    # 写入
    with open(_USER_SETTINGS_FILE, "w", encoding="utf-8") as f:
        _json.dump(existing, f, indent=2, ensure_ascii=False)

def clear_user_settings():
    """删除持久化文件（还原默认时调用）"""
    if _USER_SETTINGS_FILE.exists():
        _USER_SETTINGS_FILE.unlink()


# 暴露单例给全量代码使用
settings = GlobalSettings()

# 启动时自动加载用户上次保存的设置
_applied = _load_user_overrides(settings)
if _applied:
    import logging as _logging
    _logging.getLogger(__name__).info(f"[Settings] 从 user_settings.json 恢复了 {len(_applied)} 项设置: {_applied}")
