# ============================================================
# 【V2 升级】统一双模型音频处理接口
# 来源：V2 架构重构方案 → Phase 1
# 
# 本模块封装了两个音频 AI 模型的推理接口：
#   - M2D-CLAP: 跨模态检索（自然语言 ↔ 音频语义匹配）
#   - OMAR-RQ:  纯音频特征提取（multicodebook 版本）
# 
# 设计原则：
#   1. 懒加载（Lazy Loading）── 首次调用时才加载模型权重
#   2. 全局单例缓存 ── 避免多次推理重复加载 600MB+ 权重
#   3. 同时支持 GPU 与 CPU 推理
#   4. 两个模型统一使用 16kHz 采样率
# ============================================================

import os
import sys
import logging
import warnings
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import numpy as np

logger = logging.getLogger(__name__)

# ---- 全局缓存（懒加载单例）----
_M2D2_MODEL = None       # PortableM2D 实例
_OMAR_MODEL = None        # OMAR-RQ Module 实例

# ---- M2D-CLAP 权重路径配置 ----
# 默认路径：用户可通过环境变量 M2D_CLAP_WEIGHT 自定义
_M2D_CLAP_WEIGHT_DIR = os.getenv(
    "M2D_CLAP_WEIGHT_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "m2d_clap",
                 "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025")
)
_M2D_CLAP_CHECKPOINT = os.getenv(
    "M2D_CLAP_CHECKPOINT",
    os.path.join(_M2D_CLAP_WEIGHT_DIR, "checkpoint-30.pth")
)


def _get_device() -> torch.device:
    """探测最佳计算设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# M2D-CLAP: 跨模态检索模型
# 用于将"自然语言描述"与"音频波形"编码到同一向量空间。
# 实现文本搜音频（HyDE 虚拟乐评 → 向量检索）
# ============================================================

def get_m2d2_model():
    """
    【V2 升级】获取 M2D-CLAP 跨模态模型实例（懒加载 + 全局缓存）    
    使用官方 PortableM2D 加载器，从本地 .pth 权重文件加载。    
    M2D-CLAP 支持:
    - encode_clap_audio():  音频 → Embedding
    - encode_clap_text():   文本 → Embedding（与音频同空间）
    - 输入：16kHz 单声道波形 Tensor (B, T)
    
    Returns:
        (model, None) - model 为 PortableM2D 实例，第二个占位保持接口兼容
    """
    global _M2D2_MODEL
    if _M2D2_MODEL is not None:
        return _M2D2_MODEL, None
    
    device = _get_device()
    warnings.filterwarnings('ignore', category=UserWarning)
    
    try:
        # 导入 portable_m2d（放在 rag/ 目录下）
        from retrieval.portable_m2d import PortableM2D
        
        weight_file = _M2D_CLAP_CHECKPOINT
        if not os.path.exists(weight_file):
            raise FileNotFoundError(
                f"M2D-CLAP 权重文件未找到: {weight_file}\n"
                f"请先下载并解压到: {_M2D_CLAP_WEIGHT_DIR}\n"
                f"下载链接: https://github.com/nttcslab/m2d/releases/download/v0.5.0/"
                f"m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip"
            )
        
        logger.info(f"[AudioEmbedder] 加载 M2D-CLAP 跨模态模型: {weight_file}")
        # flat_features=True: 输出 768 维扁平特征（而非默认的 3840=768×5 堆叠特征）
        # CLAP 模式下 audio_proj 投影层期望 768 维输入
        _M2D2_MODEL = PortableM2D(weight_file, flat_features=True)
        _M2D2_MODEL = _M2D2_MODEL.to(device)
        _M2D2_MODEL.eval()
        logger.info(f"[AudioEmbedder] M2D-CLAP 跨模态模型加载成功 → {device}")
    except Exception as e:
        logger.error(f"[AudioEmbedder] M2D-CLAP 模型加载失败: {e}")
        raise
    
    return _M2D2_MODEL, None


def encode_text_to_embedding(text: str) -> List[float]:
    """
    【V2 升级】将自然语言文本编码为向量（用于在 Neo4j 中执行原生向量检索）
    
    这是 HyDE 流程的核心节点：
    用户查询 → LLM 虚拟乐评 → 本函数 → query_vector → Neo4j vector.queryNodes
    
    使用 M2D-CLAP 内置的文本编码器（GTE/BERT-based）。    """
    model, _ = get_m2d2_model()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # encode_clap_text 接受字符串列表，返回 (B, D) 的文本嵌入
        text_features = model.encode_clap_text([text])
        embedding = text_features.cpu().numpy().flatten().tolist()
    
    return embedding


def encode_audio_to_embedding(audio_array: np.ndarray, sample_rate: int = 16000) -> List[float]:
    """
    【V2 升级】将音频波形编码为跨模态 Embedding（与文本同空间）
    
    用于入库时为 Song 节点生成 m2d2_embedding 属性，
    使得可以用文本 query_vector 在 Neo4j 中检索音频。    
    Args:
        audio_array: 单声道音频 NumPy 数组，采样率为 sample_rate
        sample_rate: 输入音频的采样率（M2D-CLAP 要求 16kHz）    """
    model, _ = get_m2d2_model()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # PortableM2D 接受 (B, T) 的 Tensor 裸波形，内部自动做 mel 转换
        batch_audio = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0).to(device)
        audio_features = model.encode_clap_audio(batch_audio)
        embedding = audio_features.cpu().numpy().flatten().tolist()
    
    return embedding


# ============================================================
# OMAR-RQ: 纯音频特征提取模型(multicodebook 版本)
# 用于从音频波形直接提取高质量的音频表征向量
# 擅长：music tagging, pitch estimation, chord recognition, beat tracking
# ============================================================

def get_omar_model():
    """
    【V2 升级】获取 OMAR-RQ (multicodebook) 模型实例（懒加载 + 全局缓存）    
    OMAR-RQ (Open Music Audio Representation, 2025) 是自监督音频表征的最新 SOTA。    multicodebook 版本在自动打标(mAP=0.488) 上表现最佳。    
    输入：16kHz 单声道波形    
    Returns:
        (model, None) - model 为 OMAR-RQ Module 实例，第二个占位保持接口兼容
    """
    global _OMAR_MODEL
    if _OMAR_MODEL is not None:
        return _OMAR_MODEL, None
    
    device = _get_device()
    warnings.filterwarnings('ignore', category=UserWarning)
    
    omar_model_id = os.getenv("OMAR_MODEL_ID", "mtg-upf/omar-rq-multicodebook")
    
    try:
        from omar_rq import get_model
        
        logger.info(f"[AudioEmbedder] 加载 OMAR-RQ 模型: {omar_model_id}")
        
        # PyTorch 2.6+ 将 torch.load 的 weights_only 默认值改为 True。
        # 但 OMAR-RQ 的权重是 legacy .tar 格式，必须用 weights_only=False 加载。
        # 这里临时 monkey-patch torch.load，加载完成后恢复原始行为。
        import functools
        _original_torch_load = torch.load
        torch.load = functools.partial(_original_torch_load, weights_only=False)
        try:
            _OMAR_MODEL = get_model(model_id=omar_model_id, device=str(device))
        finally:
            torch.load = _original_torch_load  # 恢复原始 torch.load
        
        logger.info(f"[AudioEmbedder] OMAR-RQ (multicodebook) 模型加载成功 → {device}")
    except Exception as e:
        logger.error(f"[AudioEmbedder] OMAR-RQ 模型加载失败: {e}")
        raise
    
    return _OMAR_MODEL, None


def extract_audio_representation(audio_array: np.ndarray, sample_rate: int = 16000) -> List[float]:
    """
    【V2 升级】提取 OMAR-RQ 音频表征向量
    
    该向量捕捉音频的声学特征（乐器编排、节奏、调性等），
    可存入 Neo4j Song 节点的 omar_embedding 属性，
    用于纯音频相似度搜索（找类似听感的歌）。    
    Args:
        audio_array: 单声道音频 NumPy 数组，采样率为 sample_rate
        sample_rate: 输入音频的采样率（OMAR-RQ multicodebook 要求 16kHz）    
    Returns:
        定长向量 (List[float])，在时间维度上取平均池化
    """
    model, _ = get_omar_model()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # OMAR-RQ 接受 (B, T') 的 Tensor 波形
        x = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0).to(device)
        
        # extract_embeddings 返回形状 (L, B, T, C)
        # L=层数, B=batch, T=时间步, C=特征维度
        # 默认提取最后一层
        embeddings = model.extract_embeddings(x)
        
        # 取第一层、第一个batch、然后在时间步维度 T 上平均 → (C,)
        embedding = embeddings[0, 0].mean(dim=0).cpu().numpy().flatten().tolist()
    
    return embedding


# ============================================================
# 便捷入口：一次性提取双模型 Embedding
# ============================================================

def extract_dual_embeddings(
    audio_array: np.ndarray,
    m2d2_sample_rate: int = 16000,
    omar_sample_rate: int = 16000
) -> Dict[str, List[float]]:
    """
    【V2 升级】一次调用提取双模型 Embedding
    
    两个模型统一使用 16kHz，如果输入已经是 16kHz 则无需额外重采样。    
    Args:
        audio_array: 单声道音频 NumPy 数组（原始采样率）
        m2d2_sample_rate: audio_array 的实际采样率（会重采样到 16kHz）
        omar_sample_rate: audio_array 的实际采样率（会重采样到 16kHz）    
    Returns:
        {
            "m2d2_embedding": [...],   # 跨模态向量（文本-音频同空间）
            "omar_embedding": [...]    # 纯音频特征向量（声学表征）        }
    """
    import librosa
    
    # 两个模型都需要 16kHz
    m2d2_audio = librosa.resample(audio_array, orig_sr=m2d2_sample_rate, target_sr=16000) if m2d2_sample_rate != 16000 else audio_array
    omar_audio = librosa.resample(audio_array, orig_sr=omar_sample_rate, target_sr=16000) if omar_sample_rate != 16000 else audio_array
    
    m2d2_emb = encode_audio_to_embedding(m2d2_audio, sample_rate=16000)
    omar_emb = extract_audio_representation(omar_audio, sample_rate=16000)
    
    return {
        "m2d2_embedding": m2d2_emb,
        "omar_embedding": omar_emb
    }
