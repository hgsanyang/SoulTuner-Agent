#!/bin/bash
# =============================================
#  Planner 推理服务启动脚本
#  支持三种后端: SGLang(FP8) / SGLang(INT8) / Ollama
#
#  用法:
#    bash scripts/start_vllm.sh           # 默认: SGLang FP8
#    bash scripts/start_vllm.sh sglang    # SGLang FP8
#    bash scripts/start_vllm.sh int8      # SGLang bitsandbytes INT8
#    bash scripts/start_vllm.sh ollama    # Ollama
# =============================================

# 激活虚拟环境
source ~/vllm-env/bin/activate

MODEL_PATH="/mnt/c/Users/sanyang/Models/planner_model/planner_merged_fp16"
OLLAMA_MODEL="planner"   # Ollama 中注册的模型名
HOST="0.0.0.0"
PORT=8000

MODE="${1:-sglang}"

case "$MODE" in

  # ============================================
  #  方案一: SGLang + FP8 量化 (推荐, RTX 4070 原生加速)
  # ============================================
  sglang|fp8)
    echo "🚀 启动 SGLang Planner 服务 (FP8 量化)..."
    echo "   模型路径: $MODEL_PATH"
    echo "   服务地址: http://$HOST:$PORT/v1"
    echo ""
    python -m sglang.launch_server \
      --model-path "$MODEL_PATH" \
      --host "$HOST" \
      --port "$PORT" \
      --trust-remote-code \
      --quantization fp8 \
      --kv-cache-dtype fp8_e5m2 \
      --mem-fraction-static 0.85 \
      --context-length 8192 \
      --chat-template chatml
    ;;

  # ============================================
  #  方案二: SGLang + bitsandbytes INT8 (备选)
  # ============================================
  int8|bnb)
    echo "🚀 启动 SGLang Planner 服务 (INT8 bitsandbytes)..."
    echo "   模型路径: $MODEL_PATH"
    echo "   服务地址: http://$HOST:$PORT/v1"
    echo ""
    python -m sglang.launch_server \
      --model-path "$MODEL_PATH" \
      --host "$HOST" \
      --port "$PORT" \
      --trust-remote-code \
      --quantization bitsandbytes \
      --mem-fraction-static 0.80 \
      --context-length 8192 \
      --chat-template chatml
    ;;

  # ============================================
  #  方案三: Ollama (最简单, 免配置)
  # ============================================
  ollama)
    echo "🚀 启动 Ollama Planner 服务..."
    echo "   模型名: $OLLAMA_MODEL"
    echo "   服务地址: http://localhost:11434"
    echo ""

    # 检查 Ollama 是否已安装
    if ! command -v ollama &> /dev/null; then
      echo "❌ Ollama 未安装, 正在安装..."
      curl -fsSL https://ollama.com/install.sh | sh
    fi

    # 启动 Ollama 服务 (如果未运行)
    if ! pgrep -x "ollama" > /dev/null; then
      echo "📦 启动 Ollama 后台服务..."
      ollama serve &
      sleep 3
    fi

    # 检查模型是否已导入, 若未导入则提示用户
    if ! ollama list | grep -q "$OLLAMA_MODEL"; then
      echo "⚠️  模型 '$OLLAMA_MODEL' 尚未导入 Ollama"
      echo ""
      echo "请先创建 Modelfile 并导入模型:"
      echo "  1. 创建 Modelfile 文件，内容如下:"
      echo "     FROM $MODEL_PATH"
      echo "     TEMPLATE \"{{ .System }}{{ .Prompt }}\""
      echo ""
      echo "  2. 运行: ollama create $OLLAMA_MODEL -f Modelfile"
      echo ""
      exit 1
    fi

    # 运行模型
    echo "✅ 模型已就绪, 启动交互..."
    ollama run "$OLLAMA_MODEL"
    ;;

  *)
    echo "❌ 未知模式: $MODE"
    echo ""
    echo "用法: bash scripts/start_vllm.sh [sglang|int8|ollama]"
    echo "  sglang / fp8  - SGLang FP8 量化 (默认, 推荐)"
    echo "  int8 / bnb    - SGLang bitsandbytes INT8"
    echo "  ollama        - Ollama 本地部署"
    exit 1
    ;;
esac
