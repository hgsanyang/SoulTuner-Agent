# ============================================================
# SoulTuner-Agent Python 后端 Dockerfile
# 多阶段构建：减少最终镜像体积
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

# 先安装依赖（利用 Docker 缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8501"]
