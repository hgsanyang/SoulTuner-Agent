# ============================================================
# SoulTuner-Agent backend runtime.
# MuQ-MuLan is the text-to-music anchor; M2D-CLAP and OMAR-RQ provide
# fallback and acoustic representations. Model weights are mounted at runtime.
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

# Keep CPU images free of CUDA runtime wheels. The GPU compose overlay replaces
# this index explicitly, so both profiles use the same pinned PyTorch release.
ARG TORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# 可选的国内镜像，默认空（公开构建仍走官方源）。GPU overlay 会设置它：
# cu124 要拉 ~3GB（torch + 13 个 nvidia_* 运行时 + triton），官方源在长链路上
# 会读超时 —— 第一次 CUDA 构建就是在第 1757 秒因此失败的。
ARG TORCH_FIND_LINKS=
# nvidia_* 是普通 PyPI 包，可以走 requirements.txt 已经在用的清华源。
ARG PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# --timeout/--retries 不是保险起见：3GB 长传至少会卡一次，
# 一次卡顿不该让已经下了半小时的东西全部作废。
RUN pip install --no-cache-dir --timeout 120 --retries 10 \
    ${TORCH_FIND_LINKS:+--find-links ${TORCH_FIND_LINKS}} \
    --index-url ${TORCH_INDEX_URL} \
    --extra-index-url ${PYPI_INDEX_URL} \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    --trusted-host mirrors.aliyun.com \
    torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}

# 安装业务依赖（利用 Docker 缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 安装 M2D-CLAP / OMAR-RQ 的补充运行时代码库（MuQ 已在 requirements.txt）
# 注意：这些是运行模型所需的 Python 库，不是模型权重文件
# 模型权重通过 docker-compose.yml 的 volume 从宿主机挂载
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    timm einops nnAudio "transformers>=4.50,<5" sentence-transformers librosa omar-rq==0.2.1
RUN python -m pip check && python -c \
    "import omar_rq, torch; assert torch.__version__.split('+')[0] == '2.6.0', torch.__version__"

# 复制项目源码
COPY config/ ./config/
COPY agent/ ./agent/
COPY api/ ./api/
COPY llms/ ./llms/
COPY retrieval/ ./retrieval/
COPY schemas/ ./schemas/
COPY services/ ./services/
COPY tools/ ./tools/
COPY scripts/ ./scripts/
COPY data/pipeline/ ./data/pipeline/
# The V4.2 prompt and deterministic guard are shared by the self-hosted
# endpoint and the production Agent adapter.
COPY deploy/self_hosted_35b/prompt_v42.py ./deploy/self_hosted_35b/prompt_v42.py
COPY deploy/self_hosted_35b/planner_guard.py ./deploy/self_hosted_35b/planner_guard.py

# 数据目录（运行时通过 volume 挂载实际数据）
RUN mkdir -p /app/data

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8501"]
