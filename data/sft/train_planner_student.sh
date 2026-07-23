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

# ---- ①锁版本 (未锁易踩 MoE 模板/loss=0), 硬拒过旧 ms-swift --------------------
: "${MIN_MS_SWIFT:=3.4.0}"
echo "== 版本锁定 (ms-swift >= $MIN_MS_SWIFT) =="
python - "$MIN_MS_SWIFT" <<'PY'
import importlib.metadata as m, sys
try:
    from packaging.version import Version
except Exception:
    print("FAIL: 缺 packaging"); sys.exit(4)
minv = sys.argv[1]; snap = {}
for p in ("ms-swift", "transformers", "peft", "torch", "trl", "accelerate"):
    try: snap[p] = m.version(p)
    except Exception: snap[p] = None
for p, v in snap.items(): print(f"  {p} == {v}")
if not snap.get("ms-swift"):
    print("FAIL: ms-swift 未安装"); sys.exit(4)
if Version(snap["ms-swift"]) < Version(minv):
    print(f"FAIL: ms-swift {snap['ms-swift']} < {minv} (MoE LoRA 已知坑)"); sys.exit(4)
PY
mkdir -p "$PREFLIGHT_DIR" "$OUTPUT_DIR"
# 复现: pip freeze > requirements-train.lock 并纳入版本管理。
# ⚠Qwen3.6 默认可能带 thinking: 训练目标无 <think>, 但推理需关闭 thinking
#   (按你安装的 ms-swift 版本设置 template 的 enable_thinking=false, 并抽查
#    一条 swift infer 输出确认无 <think> 块)。

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
  --freeze_aligner true
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05
  --target_modules all-linear
  --max_length 2048
  --per_device_train_batch_size 4
  --gradient_accumulation_steps 4
  --learning_rate 1e-4
  --gradient_checkpointing true
  --seed 42
)

# ---- ②PREFLIGHT: 50 步冒烟, 硬门 (真实退出码 + 结构化日志) ------------------
echo "== PREFLIGHT: 50-step 冒烟 =="
set +e
swift sft "${COMMON[@]}" \
  --max_steps 50 --logging_steps 1 \
  --save_steps 50 --save_total_limit 1 \
  --output_dir "$PREFLIGHT_DIR" 2>&1 | tee "$PREFLIGHT_DIR.log"
SWIFT_RC=${PIPESTATUS[0]}
set -e
if [ "$SWIFT_RC" -ne 0 ]; then
  echo "!! PREFLIGHT: swift sft 退出码 $SWIFT_RC (非零) — 训练本身失败(依赖/OOM/模型缺失), 不要继续。"
  exit 2
fi
# 结构化硬门: 完成50步 + 末步 loss>0 且有限 + token_acc(若有)>0 + adapter 落盘。
python - "$PREFLIGHT_DIR" <<'PY'
import glob, json, math, os, sys
d = sys.argv[1]
logs = (glob.glob(os.path.join(d, "**", "logging.jsonl"), recursive=True)
        or glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True))
steps = []
for f in logs:
    for line in open(f, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if isinstance(r, dict) and "loss" in r:
            steps.append(r)
if len(steps) < 10:
    print(f"PREFLIGHT FAIL: 只解析出 {len(steps)} 条含 loss 的训练记录 (<10)"); sys.exit(3)
last = steps[-1]
loss = float(last.get("loss", 0) or 0)
if not math.isfinite(loss) or loss <= 0:
    print(f"PREFLIGHT FAIL: 末步 loss={loss} (应 >0 且有限) — 疑似 MoE loss=0 坑 (#8142)"); sys.exit(3)
acc = last.get("token_acc", last.get("acc"))
acc_checked = acc is not None
if acc_checked and float(acc) <= 0:
    print(f"PREFLIGHT FAIL: token_acc={acc} <= 0"); sys.exit(3)
# ms-swift 不同版本用 global_step 或 current_steps
gstep = max(int(s.get("global_step") or s.get("current_steps") or 0) for s in steps)
if gstep < 50:
    print(f"PREFLIGHT FAIL: 只完成 {gstep} 步 (<50)"); sys.exit(3)
# adapter 必须同时有 config + 权重 (不接受任意 .safetensors)
cfgs = glob.glob(os.path.join(d, "**", "adapter_config.json"), recursive=True)
wts = glob.glob(os.path.join(d, "**", "adapter_model.safetensors"), recursive=True)
if not (cfgs and wts):
    print(f"PREFLIGHT FAIL: 缺 adapter_config.json({len(cfgs)}) 或 adapter_model.safetensors({len(wts)})"); sys.exit(3)
print(f"PREFLIGHT OK: steps={gstep} records={len(steps)} last_loss={loss:.4f} "
      f"token_acc={'(absent)' if not acc_checked else acc} adapter=config+weights")
PY
echo "== PREFLIGHT 通过 =="

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
