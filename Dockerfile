# ============================================================
# SoulTuner-Agent Python 后端 Dockerfile
#
# 包含 M2D-CLAP 跨模态模型运行时依赖，支持：
#   - 语义向量检索（Semantic Search）
#   - 双锚精排（Dual-Anchor Rerank）
#   - HyDE 声学描述 → 向量匹配
#
# 模型权重通过 volume 挂载宿主机缓存，不打包进镜像
# 预计镜像大小：~11 GB
# ============================================================
FROM python:3.12-slim AS base

# 系统依赖（音频处理需要 libsndfile）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安装业务依赖（利用 Docker 缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 安装 M2D-CLAP 运行时代码库（参见 retrieval/portable_m2d.py）
# 注意：这些是运行模型所需的 Python 库，不是模型权重文件
# 模型权重通过 docker-compose.yml 的 volume 从宿主机挂载
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    timm einops nnAudio transformers sentence-transformers librosa

# 复制项目源码
COPY config/ ./config/
COPY agent/ ./agent/
COPY api/ ./api/
COPY llms/ ./llms/
COPY retrieval/ ./retrieval/
COPY schemas/ ./schemas/
COPY services/ ./services/
COPY tools/ ./tools/

# 数据目录（运行时通过 volume 挂载实际数据）
RUN mkdir -p /app/data

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8501"]
