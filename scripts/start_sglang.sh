#!/bin/bash
# ============================================================
# SGLang 部署脚本 - Qwen3-4B Planner (FP8 在线量化)
# 适用环境: WSL2 + RTX 4070 8GB
# ============================================================
#
# 【环境信息】
#   虚拟环境: ~/vllm-env (已安装 SGLang 0.5.9)
#   激活方式: source ~/vllm-env/bin/activate
#
# 【使用方法】
#   1. 打开 Windows Terminal，进入 WSL:
#        wsl
#
#   2. 启动服务:
#        bash /mnt/c/Users/sanyang/sanyangworkspace/music_recommendation/Muisc-Research/scripts/start_sglang.sh
#
#   3. 服务启动后，在 Windows 端用 http://localhost:8000 访问（OpenAI 兼容 API）
#
# ============================================================

# ---------- 配置区 ----------

# 虚拟环境路径（WSL 中已有的 vllm-env）
VENV_DIR="$HOME/vllm-env"

# 微调模型路径（Windows 路径通过 /mnt/ 映射到 WSL）
MODEL_PATH="/mnt/c/Users/sanyang/Models/Qwen3-4B-Instruct-2507"

# 服务端口和地址
PORT=8000
HOST="0.0.0.0"

# --- 显存优化参数 ---
# 针对 RTX 4070 8GB，预留 ~1.5-2GB 给 m2d2 等其他模型
#
# quantization=fp8     → FP8 在线动态量化，模型从 BF16 自动转为 FP8
#                        无需预量化！4070 Ada 架构原生支持 FP8 Tensor Core
#                        模型显存占用: ~4GB (原 BF16 为 ~8GB，减半)
#
# kv-cache-dtype       → KV Cache（AI 的"短期记忆草稿本"）也用 FP8
#                        fp8_e5m2 格式动态范围更大，适合 KV Cache
#                        进一步减少显存占用
#
# mem-fraction-static  → SGLang 预分配的显存比例
#                        0.70 = 只用 70% 显存 ≈ 5.6GB
#                        剩余 ~2.4GB 留给 m2d2 + 系统开销
#
# context-length       → 最大上下文长度
#                        Planner prompt ~800 tokens + JSON output ~500 tokens
#                        4096 tokens 提供足够余量
#
# max-running-requests → 最大同时处理请求数
#                        本地单用户使用，1 个并发就够
#                        每多一个并发 = 多一份 KV Cache 显存开销
#
QUANTIZATION="fp8"
KV_CACHE_DTYPE="fp8_e5m2"
MEM_FRACTION=0.70
CONTEXT_LENGTH=4096
MAX_RUNNING_REQUESTS=1

# ----------------------------

set -e  # 遇到错误立即退出

echo "============================================"
echo "  SGLang 推理服务 - Qwen3-4B Planner"
echo "============================================"

# ---- 步骤1: 激活虚拟环境 ----
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ 错误: 虚拟环境不存在: $VENV_DIR"
    echo "   请先创建环境并安装 SGLang:"
    echo "   python3 -m venv ~/vllm-env"
    echo "   source ~/vllm-env/bin/activate"
    echo "   pip install 'sglang[all]>=0.5.9' --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python"
    exit 1
fi

source "$VENV_DIR/bin/activate"
echo "✅ 已激活环境: $VENV_DIR"

# ---- 步骤2: 检查模型路径 ----
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "❌ 错误: 模型路径不存在或缺少 config.json"
    echo "   模型路径: $MODEL_PATH"
    exit 1
fi
echo "✅ 模型路径确认: $MODEL_PATH"

# ---- 步骤3: 打印启动参数 ----
echo ""
echo "📋 启动参数:"
echo "   模型路径:       $MODEL_PATH"
echo "   量化方式:       $QUANTIZATION (在线动态量化，BF16→FP8)"
echo "   KV Cache 精度:  $KV_CACHE_DTYPE"
echo "   显存占比:       ${MEM_FRACTION} (≈5.6GB / 8GB)"
echo "   上下文长度:     $CONTEXT_LENGTH tokens"
echo "   最大并发:       $MAX_RUNNING_REQUESTS"
echo "   端口:           $PORT"
echo ""
echo "📊 预计显存分配:"
echo "   FP8 模型权重:   ~4.0 GB"
echo "   KV Cache:       ~1.0 GB"
echo "   CUDA 开销:      ~0.5 GB"
echo "   ─────────────────────"
echo "   合计:           ~5.5 GB"
echo "   剩余(给m2d2):   ~2.5 GB"
echo ""
echo "🔗 API 地址: http://localhost:${PORT}/v1/chat/completions"
echo "   按 Ctrl+C 停止服务"
echo "============================================"
echo ""

# ---- 步骤4: 启动 SGLang 服务 ----
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --host "$HOST" \
    --quantization "$QUANTIZATION" \
    --mem-fraction-static "$MEM_FRACTION" \
    --context-length "$CONTEXT_LENGTH" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --disable-cuda-graph \
    --attention-backend triton
