"""
多模型聚合引擎 (基于 LiteLLM)
提供开箱即用的多厂商大模型自动切换功能。
支持厂商：OpenAI, DeepSeek, 智谱(Zhipu), Minimax, 阿里百炼(DashScope), SiliconFlow
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# 加载 .env 文件（覆盖模式，确保能读到最新的 Key）
load_dotenv(override=True)

try:
    from config.settings import settings
except Exception:
    pass

from .base import BaseLLM

# ---- 统一模型配置字典 ----
# 预设各大厂商的默认调用前缀和推荐选型（你可以在这里随时修改默认模型）
MODEL_REGISTRY = {
    "openai": {
        "prefix": "openai/",
        "default_model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL"
    },
    "deepseek": {
        "prefix": "deepseek/",
        "default_model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL"
    },
    "zhipu": {
        "prefix": "zhipu/",
        "default_model": "glm-4",
        "api_key_env": "ZHIPU_API_KEY",
        "base_url_env": "ZHIPU_BASE_URL"
    },
    "minimax": {
        "prefix": "minimax/",
        "default_model": "abab6.5s-chat",
        "api_key_env": "MINIMAX_API_KEY",
        "base_url_env": "MINIMAX_BASE_URL"
    },
    "dashscope": { # 阿里百炼
        "prefix": "dashscope/",
        "default_model": "qwen3.5-flash",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url_env": "DASHSCOPE_BASE_URL"
    },
    "siliconflow": {
        "prefix": "openai/",  # SiliconFlow 完全兼容 OpenAI，所以让 LiteLLM 走 openai 协议最稳
        "default_model": "deepseek-ai/DeepSeek-V3.2",
        #MiniMax-M2，
        "api_key_env": ["SiliconFlow_API_KEY", "SILICONFLOW_API_KEY"],
        "base_url_env": "SILICONFLOW_BASE_URL",
        "default_base_url": "https://api.siliconflow.cn/v1"
    },
    "ollama": {
        "prefix": "openai/",  # 使用 OpenAI 兼容模式
        "default_model": "qwen2.5:7b", # 本地默认推断模型
        "api_key_env": ["OLLAMA_API_KEY", "LLM_API_KEY"],
        "base_url_env": "OLLAMA_BASE_URL",
        "default_base_url": "http://localhost:11434/v1" # Ollama 的 OpenAI 兼容接口路径
    },
    "vllm": {
        "prefix": "openai/",  # 使用 OpenAI 兼容模式
        "default_model": "Qwen/Qwen2.5-7B-Instruct",
        "api_key_env": ["VLLM_API_KEY", "LLM_API_KEY"],
        "base_url_env": "VLLM_BASE_URL",
        "default_base_url": "http://localhost:8000/v1" # vLLM 默认路径
    },
    "sglang": {
        "prefix": "openai/",  # SGLang 也提供 OpenAI 兼容 API
        "default_model": "local-planner-qwen3-4b-fp8",
        "api_key_env": ["SGLANG_API_KEY", "LLM_API_KEY"],
        "base_url_env": "SGLANG_BASE_URL",
        "default_base_url": "http://localhost:8000/v1"
    },
    "google": {  # Google Gemini
        "prefix": "gemini/",
        "default_model": "gemini-3-flash-preview",
        "api_key_env": "GOOGLE_API_KEY",
        "base_url_env": ""
    },
    "volcengine": {  # 火山引擎 / 豆包（字节跳动）
        "prefix": "openai/",  # 火山引擎 Ark 完全兼容 OpenAI 协议
        "default_model": "ep-20260405142751-x4jm6",
        "api_key_env": "VOLCENGINE_API_KEY",
        "base_url_env": "VOLCENGINE_BASE_URL",
        "default_base_url": "https://ark.cn-beijing.volces.com/api/v3"
    }
}

def _get_env_val(env_keys, default=None):
    """辅助函数：按优先级获取环境变量配置"""
    if isinstance(env_keys, str):
        env_keys = [env_keys]
    for key in env_keys:
        val = os.getenv(key)
        if val:
            return val
    return default


class MultiLLM(BaseLLM):
    """
    轻量级原生调用器 (继承自 BaseLLM)
    适用场景: mcp_adapter.py 处理简单清洗、写歌单描述等不需要 LangChain 的底层独立模块
    """
    def __init__(self, provider: str = "siliconflow", model_name: Optional[str] = None, temperature: float = 0.7):
        """
        初始化大模型调用客户端
        
        Args:
            provider: 厂商代号 ("openai", "deepseek", "zhipu", "minimax", "dashscope", "siliconflow")
            model_name: 可选的大模型具体型号
        """
        self.provider = provider.lower()
        if self.provider not in MODEL_REGISTRY:
            raise ValueError(f"不支持的厂商: {self.provider}。支持列表: {list(MODEL_REGISTRY.keys())}")
            
        config = MODEL_REGISTRY[self.provider]
        
        # 提取并锁定 API KEY
        api_key = _get_env_val(config["api_key_env"])
        if not api_key:
            print(f"⚠️ 警告: 尚未在环境变量中找到 {self.provider} 的 API_KEY")
        
        # 处理特定厂商的环境变量注射（LiteLLM 强依赖这些全局环境变量）
        if self.provider == "dashscope":
            os.environ["DASHSCOPE_API_KEY"] = api_key or ""
        elif self.provider == "zhipu":
            os.environ["ZHIPUAI_API_KEY"] = api_key or ""
        elif self.provider == "deepseek":
            os.environ["DEEPSEEK_API_KEY"] = api_key or ""
        elif self.provider == "minimax":
            os.environ["MINIMAX_API_KEY"] = api_key or ""
            
        # 针对 OpenAPI 兼容模式（如 SiliconFlow, Ollama, vLLM, SGLang 和 OpenAI 原生）
        if self.provider in ["siliconflow", "ollama", "vllm", "sglang", "openai"]:
            os.environ["OPENAI_API_KEY"] = api_key or ""
            base_url = _get_env_val(config.get("base_url_env", ""), config.get("default_base_url"))
            if base_url:
                os.environ["OPENAI_API_BASE"] = base_url
            
            self.model_name = model_name or config["default_model"]
            self.litellm_model = f"openai/{self.model_name}" if self.provider in ["siliconflow", "ollama", "vllm", "sglang"] else self.model_name
        else:
            base_model = model_name or config["default_model"]
            # 自动补全厂商前缀
            if not base_model.startswith(config["prefix"]):
                self.litellm_model = f"{config['prefix']}{base_model}"
            else:
                self.litellm_model = base_model
            self.model_name = base_model
                
        if self.provider == "google" and temperature < 1.0:
            # Gemini 3 模型强制建议 temperature 为 1.0 以避免死循环
            self.temperature = 1.0
        else:
            self.temperature = temperature
            
        super().__init__(api_key=api_key, model_name=self.model_name)

    def get_default_model(self) -> str:
        return self.model_name

    def invoke(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """原生 API 调用，经过 LiteLLM 路由"""
        try:
            from litellm import completion
        except ImportError:
            raise ImportError("环境缺少 litellm 库。请执行: pip install litellm")
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        temp = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", 4000)
        
        try:
            response = completion(
                model=self.litellm_model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content
            return self.validate_response(content)
        except Exception as e:
            print(f"[{self.provider}] LiteLLM API调用错误: {str(e)}")
            return ""

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "routing_target": self.litellm_model
        }


def get_chat_model(provider: str = "siliconflow", model_name: Optional[str] = None, temperature: float = 0.7, timeout: Optional[int] = None, max_tokens: Optional[int] = None):
    """
    获取 LangChain 专属的 ChatModel 对象 (工厂暴露函数)
    适用场景: graphs/music_graph.py 等复杂图节点工作流
    
    Args:
        provider: 厂商代号 ("openai", "deepseek", "zhipu", "minimax", "dashscope", "siliconflow")
        model_name: 可选，如果你不想用默认模型，可以强行指定（比如 "qwen-turbo"）
        temperature: 温度参数
    """
    provider = provider.lower()
    if provider not in MODEL_REGISTRY:
        raise ValueError(f"不支持的厂商: {provider}")
        
    config = MODEL_REGISTRY[provider]
    
    # 获取 API KEY 并强行注入操作系统的环境变量（给底层库用）
    api_key = _get_env_val(config["api_key_env"])
    base_url = _get_env_val(config.get("base_url_env", ""), config.get("default_base_url"))
    
    # 策略 1：如果是 SiliconFlow 或本地 OpenAI 兼容服务 (Ollama, vLLM, SGLang)，为了保证绝对稳定，直接使用 LangChain 原生 ChatOpenAI 包装
    if provider in ["siliconflow", "volcengine", "ollama", "vllm", "sglang"]:
        from langchain_openai import ChatOpenAI
        target_model = model_name or config["default_model"]
        # 读取超时配置（优先用参数，其次用 settings，默认 80s）
        _timeout = timeout
        if _timeout is None:
            try:
                _timeout = settings.llm_timeout
            except Exception:
                _timeout = 80
        # max_tokens 优先使用调用方显式传入的值，否则默认 4000
        _max_tokens = max_tokens if max_tokens is not None else 4000
        # ── SGLang + Qwen3：通过 extra_body 硬关闭 Thinking Mode ──
        # Qwen3 默认启用 thinking mode，会在输出 JSON 前生成数百个推理 token（在 RTX 4070 约 30-60s）。
        # chat_template_kwargs.enable_thinking=False 直接在 Jinja2 chat template 层面跳过 <think> 推理链。
        _model_kwargs = {}
        if provider == "sglang":
            _model_kwargs = {
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            }
        return ChatOpenAI(
            api_key=api_key or "fake-key",
            base_url=base_url,
            model=target_model,
            temperature=temperature,
            max_tokens=_max_tokens,
            request_timeout=_timeout,
            model_kwargs=_model_kwargs,
        )
    
    # 策略 2：其他所有厂商，走 LangChain 的 ChatLiteLLM 通用通道
    try:
        from langchain_litellm import ChatLiteLLM
    except ImportError:
        raise ImportError("请确保安装了依赖: pip install litellm langchain-litellm langchain-openai")
        
    base_model = model_name or config["default_model"]
    if not base_model.startswith(config["prefix"]):
        target_model = f"{config['prefix']}{base_model}"
    else:
        target_model = base_model
        
    # LiteLLM 专属的环境变量装载
    if provider == "dashscope":
        os.environ["DASHSCOPE_API_KEY"] = api_key or ""
    elif provider == "zhipu":
        os.environ["ZHIPUAI_API_KEY"] = api_key or ""
    elif provider == "deepseek":
        os.environ["DEEPSEEK_API_KEY"] = api_key or ""
    elif provider == "minimax":
        os.environ["MINIMAX_API_KEY"] = api_key or ""
    elif provider == "openai":
        os.environ["OPENAI_API_KEY"] = api_key or ""
        if base_url:
            os.environ["OPENAI_API_BASE"] = base_url
    elif provider == "google":
        os.environ["GOOGLE_API_KEY"] = api_key or ""
        # 针对 Gemini 自动调整温度
        if temperature < 1.0:
            temperature = 1.0
            
    return ChatLiteLLM(
        model=target_model,
        temperature=temperature,
        max_tokens=max_tokens if max_tokens is not None else 4000
    )

def deepseek_llm(temperature: float = 0.7):
    """便捷包装函数：快速获取 DeepSeek 模型实例"""
    return get_chat_model(provider="deepseek", temperature=temperature)

def qwen_llm(temperature: float = 0.7):
    """便捷包装函数：快速获取阿里百炼(千问) 模型实例"""
    return get_chat_model(provider="siliconflow", temperature=temperature)

def siliconflow_llm(temperature: float = 0.7):
    """便捷包装函数：快速获取硅基流动 模型实例"""
    return get_chat_model(provider="siliconflow", temperature=temperature)

def get_intent_chat_model():
    """获取意图分析专用 LLM 实例（从 settings 读取配置）
    
    可通过环境变量 INTENT_LLM_PROVIDER / INTENT_LLM_MODEL 配置。
    典型用法：配置一个更小更快的模型专门做意图分类，减少延迟。
    
    max_tokens 锁定 1024：意图分析输出为结构化 JSON（MusicQueryPlan），
    典型输出约 200-400 tokens，1024 提供充足安全余量，同时避免
    与本地模型的 context-length（如 8192）冲突。
    """
    try:
        provider = settings.intent_llm_provider or settings.llm_default_provider
        model_name = settings.intent_llm_model or None  # 空字符串转 None，用 provider 默认模型
        return get_chat_model(provider=provider, model_name=model_name, temperature=0.3, max_tokens=1024)
    except Exception:
        # 回退到默认配置
        return get_chat_model(provider="siliconflow", temperature=0.3, max_tokens=1024)

def gemini_llm(model_name: Optional[str] = None, temperature: float = 1.0):
    """便捷包装函数：快速获取 Google Gemini 模型实例
    
    常用模型：
      - gemini-3-flash-preview  (当前默认，性能强劲)
      - gemini-2.0-flash        (速度极快、成本低)
      - gemini-1.5-pro          (长上下文处理能力强)
    """
    return get_chat_model(provider="google", model_name=model_name, temperature=temperature)

def ollama_llm(model_name: Optional[str] = None, temperature: float = 0.6):
    """便捷包装函数：快速获取本地 Ollama 模型实例"""
    return get_chat_model(provider="ollama", model_name=model_name, temperature=temperature)

def vllm_llm(model_name: Optional[str] = None, temperature: float = 0.6):
    """便捷包装函数：快速获取本地 vLLM 模型实例"""
    return get_chat_model(provider="vllm", model_name=model_name, temperature=temperature)

def sglang_llm(model_name: Optional[str] = None, temperature: float = 0.6):
    """便捷包装函数：快速获取本地 SGLang 模型实例"""
    return get_chat_model(provider="sglang", model_name=model_name, temperature=temperature)
