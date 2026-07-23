#!/usr/bin/env bash
# =============================================================================
# SoulTuner Planner 蒸馏 — 学生模型 LoRA 微调 (Phase B)
#
# 学生: Qwen3.6-35B-A3B (MoE, 3B active) — ModelScope 最新开源, Apache 2.0
# 老师: qwen3.7-plus (API) 生成的 compact PlannerDecisionV2 决策
# 平台: AMD MI300X 192GB (ModelScope ROCm 7.2.1 / torch 2.9.1)
# 框架: ms-swift (支持 Qwen3.6 + MoE LoRA)
#
# 数据格式: ChatML messages (system/user/assistant), ms-swift 原生可吃。
#   assistant = compact PlannerDecisionV2 JSON (学生要学的 target)
#   system    = build_sft_chatml.py::STUDENT_SYSTEM_PROMPT (~800 token)
#
# 用法: bash train_planner_student.sh
# =============================================================================
set -euo pipefail

# ---- 路径 (按你机器改) ------------------------------------------------------
DATA_DIR="${DATA_DIR:-./data/sft}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train_v2_chatml.jsonl}"   # 1267 条
VAL_FILE="${VAL_FILE:-$DATA_DIR/eval_v2_chatml.jsonl}"        # 224 条
OUTPUT_DIR="${OUTPUT_DIR:-./output/planner-student-35b-lora}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"                        # ModelScope id

# ---- 环境 (ROCm) ------------------------------------------------------------
export USE_MODELSCOPE_HUB=1          # 从 ModelScope 下载模型
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
# export HIP_VISIBLE_DEVICES=0       # 多卡时按需设

# ---- 训练 -------------------------------------------------------------------
# LoRA 超参说明:
#   rank16/alpha32   小数据(1.5k)够表达, 不易过拟合; 想更强表达可升 rank32
#   epochs 3         1.5k 样本 * 3 ~ 快(35B-A3B 只激活 3B, 单卡 192G 约 1-2h)
#   lr 1e-4          LoRA 常用; loss 不降可降到 5e-5
#   max_length 2048  我们样本 system~800 + user + assistant 决策 JSON < 2048
#   eff batch = 4 * 4 = 16
#
# ⚠MoE LoRA 已知坑(ms-swift #8142/#8197): 若首 10 步 loss=0 / token_acc=0,
#   多半是 target_modules 没覆盖或 template 不对 —— 先用 all-linear, 不行再
#   显式加 MoE 专家/router 线性层; 并升级到最新 ms-swift。
swift sft \
  --model "$MODEL" \
  --train_type lora \
  --dataset "$TRAIN_FILE" \
  --val_dataset "$VAL_FILE" \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --max_length 2048 \
  --warmup_ratio 0.05 \
  --gradient_checkpointing true \
  --weight_decay 0.01 \
  --logging_steps 5 \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 3 \
  --dataloader_num_workers 4 \
  --output_dir "$OUTPUT_DIR" \
  --seed 42

echo "== 训练完成. LoRA 权重在 $OUTPUT_DIR =="
echo "== 合并 LoRA (可选, 便于部署/推理):"
echo "   swift export --adapters $OUTPUT_DIR/checkpoint-best --merge_lora true"
echo "== 快速验证 (在 eval 集上跑学生, 看 schema 是否合法/意图是否匹配):"
echo "   python -m data.sft.planner_sft_training.eval_student --adapter $OUTPUT_DIR/checkpoint-best --eval $VAL_FILE"
