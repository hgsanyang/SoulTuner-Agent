#!/usr/bin/env bash
# =============================================================================
# SoulTuner Planner 蒸馏 — 学生 LoRA 微调 (Phase B, GPT-review-hardened)
#
# 学生: Qwen3.6-35B-A3B (MoE 3B active + 视觉编码器) — ModelScope, Apache 2.0
# 老师: qwen3.7-plus (API) 生成的 compact PlannerDecisionV2 决策
# 平台: AMD MI300X 192GB (ModelScope ROCm 镜像) | 框架: ms-swift
#
# 流程 (硬门): ①锁版本 → ②50-step PREFLIGHT (查 loss>0, 防 MoE loss=0 坑) →
#              ③人确认后 RUN_FULL=1 全量 → ④合并 → ⑤swift infer → ⑥score_student
#
# 参考: ms-swift 命令行参数 / AMD 支持 / MoE loss=0 issue #8142
# 用法: bash train_planner_student.sh            # 只跑 preflight
#       RUN_FULL=1 bash train_planner_student.sh  # preflight 通过后跑全量
# =============================================================================
set -euo pipefail

# ---- 路径 (按你机器改) ------------------------------------------------------
DATA_DIR="${DATA_DIR:-./data/sft}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train_v2_chatml.jsonl}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/eval_v2_chatml.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/planner-student-35b-lora}"
PREFLIGHT_DIR="${PREFLIGHT_DIR:-./output/planner-preflight}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"

export USE_MODELSCOPE_HUB=1
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
# export HIP_VISIBLE_DEVICES=0

# ---- ①锁版本 (未锁易踩 MoE 模板/loss=0) ------------------------------------
# 建议在容器里固定并记录; 训练前打印实际版本入日志便于复现。
echo "== 版本快照 =="
python - <<'PY'
import importlib.metadata as m
for p in ("ms-swift","transformers","peft","torch","trl","accelerate"):
    try: print(f"  {p} == {m.version(p)}")
    except Exception: print(f"  {p} == (not installed)")
PY
# 若首次: pip install 'ms-swift>=<锁定版>' 并 pip freeze > requirements-train.lock

# ---- 公共 LoRA 配置 ---------------------------------------------------------
# ⚠MoE + 视觉模型注意:
#   --freeze_vit true      冻结视觉塔 (我们只做文本 planner, 不训视觉/aligner)
#   --target_modules all-linear  仅 LLM 线性层加 LoRA; 若 preflight loss=0,
#     多半是 MoE 专家/router 未被覆盖或 template 不对 → 显式加专家线性层并升级 swift
COMMON=(
  --model "$MODEL"
  --train_type lora
  --dataset "$TRAIN_FILE"
  --val_dataset "$VAL_FILE"
  --torch_dtype bfloat16
  --freeze_vit true
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05
  --target_modules all-linear
  --max_length 2048
  --per_device_train_batch_size 4
  --gradient_accumulation_steps 4
  --learning_rate 1e-4
  --gradient_checkpointing true
  --seed 42
)

# ---- ②PREFLIGHT: 50 步冒烟, 硬门 loss>0 ------------------------------------
echo "== PREFLIGHT: 50-step 冒烟 =="
swift sft "${COMMON[@]}" \
  --max_steps 50 --logging_steps 1 --save_strategy no \
  --output_dir "$PREFLIGHT_DIR" 2>&1 | tee "$PREFLIGHT_DIR.log" || true

# 检查前若干步 loss 是否恒为 0 (MoE 已知坑)
if grep -qE "'loss': 0\.0[, }]" "$PREFLIGHT_DIR.log"; then
  echo "!! PREFLIGHT 检出 loss=0 —— 疑似 MoE LoRA target_modules/template 问题。"
  echo "   先修 (加专家层 / 升级 ms-swift / 核对 template), 不要跑全量。"
  exit 2
fi
echo "== PREFLIGHT 通过 (loss 非零) =="

if [ "${RUN_FULL:-0}" != "1" ]; then
  echo "== 只跑了 preflight. 确认无误后: RUN_FULL=1 bash $0 =="
  exit 0
fi

# ---- ③全量训练 -------------------------------------------------------------
echo "== 全量训练 =="
swift sft "${COMMON[@]}" \
  --num_train_epochs 3 \
  --per_device_eval_batch_size 4 \
  --warmup_ratio 0.05 --weight_decay 0.01 \
  --logging_steps 5 --eval_steps 50 --save_steps 50 --save_total_limit 3 \
  --create_checkpoint_symlink true \
  --dataloader_num_workers 4 \
  --output_dir "$OUTPUT_DIR"

BEST="$OUTPUT_DIR/best"   # --create_checkpoint_symlink 生成 best/last 软链
echo "== 训练完成. 最优 LoRA: $BEST =="

# ---- ④合并 (可选, 便于部署) -------------------------------------------------
# swift export --adapters "$BEST" --merge_lora true

# ---- ⑤在 eval 上出预测 → ⑥确定性打分 ---------------------------------------
echo "== 生成 eval 预测 =="
swift infer \
  --adapters "$BEST" \
  --val_dataset "$VAL_FILE" \
  --max_new_tokens 512 \
  --result_path "$OUTPUT_DIR/eval_predictions.jsonl"

echo "== 打分 (schema/意图/通道F1/澄清精确率/过度澄清/字段匹配, 强制100%覆盖) =="
python -m data.sft.score_student \
  --eval "$VAL_FILE" \
  --pred "$OUTPUT_DIR/eval_predictions.jsonl"

echo "== HyDE 质量另用文搜音尺子: python -m tests.eval.evaluate_alignment_attribute =="
