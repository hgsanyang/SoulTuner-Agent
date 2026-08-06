#!/usr/bin/env bash
# =============================================================================
# SoulTuner Planner 蒸馏 — 学生 LoRA 微调 (Phase B, GPT-review-hardened)
#
# 学生: Qwen3.6-35B-A3B (MoE 3B active + 视觉编码器) — ModelScope, Apache 2.0
# 老师: qwen3.7-plus (API) 生成并审计的 compact PlannerDecisionV3 决策
# 平台: AMD MI300X 192GB (ModelScope ROCm 镜像) | 框架: ms-swift
#
# 流程 (硬门): ①锁版本 + 目录归属 + 冻结 manifest + swift 参数名核对 →
#              ②50-step PREFLIGHT (查 loss>0, 防 MoE loss=0 坑) →
#              ③人确认后 RUN_FULL=1 全量 (先复核指纹) → ④合并 →
#              ⑤swift infer + 验证无 think 块 → ⑥score_student
#
# 参考: ms-swift 命令行参数 / AMD 支持 / MoE loss=0 issue #8142
# 用法: bash train_planner_student.sh            # 只跑 preflight
#       RUN_FULL=1 bash train_planner_student.sh  # preflight 通过后跑全量
#
# 9B 基线 vs 35B-A3B 对照: 只改 MODEL, 其余不动 ——
#   MODEL=Qwen/Qwen3.5-9B bash train_planner_student.sh
#   MODEL=Qwen/Qwen3.6-35B-A3B bash train_planner_student.sh
# MODEL_TAG 会自动派生, model/run 各自独立占用 preflight 与 output 目录, 且
# $OUTPUT_DIR/.run_owner.json 会拒绝第二个 model 复用同一个目录。
#
# 冻结数据集: MANIFEST_FILE=data/sft/v4/MANIFEST.json bash train_planner_student.sh
# =============================================================================
set -euo pipefail

# ---- 路径 (按你机器改) ------------------------------------------------------
DATA_DIR="${DATA_DIR:-./data/sft}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train_v3_chatml.jsonl}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/eval_v3_chatml.jsonl}"
SEALED_FILE="${SEALED_FILE:-}"
# Optional frozen V4 manifest. When set, it is validated against
# data/sft/v4/MANIFEST.schema.json and its recorded split digests must match the
# files on disk before any GPU time is spent. See data/sft/verify_frozen_manifest.py.
MANIFEST_FILE="${MANIFEST_FILE:-}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
MODEL_TAG="${MODEL_TAG:-$(printf '%s' "$MODEL" | tr -c 'A-Za-z0-9._-' '_')}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
PREFLIGHT_RUN_ID="${PREFLIGHT_RUN_ID:-$RUN_ID}"
SEED="${SEED:-42}"
# 9B baseline and 35B-A3B contrast must never land in the same tree: the merge
# step resolves `best` by globbing, so a shared directory silently scores one
# model's adapter as the other's. Model tag AND run id, both.
OUTPUT_DIR="${OUTPUT_DIR:-./output/planner-student-${MODEL_TAG}-lora/${RUN_ID}}"
# Every invocation gets a fresh directory. Reusing one directory lets a failed
# second model accidentally pass by reading the first model's old logging.jsonl.
PREFLIGHT_DIR="${PREFLIGHT_DIR:-./output/preflight/${MODEL_TAG}/${PREFLIGHT_RUN_ID}}"
RUN_RECORD="$PREFLIGHT_DIR/environment.json"

export USE_MODELSCOPE_HUB=1
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"
export NVTE_USE_GROUPED_GEMM_TRITON="${NVTE_USE_GROUPED_GEMM_TRITON:-1}"
export USE_MCORE_GDN="${USE_MCORE_GDN:-1}"
# export HIP_VISIBLE_DEVICES=0

# ---- ①锁版本 (未锁易踩 MoE 模板/loss=0), 硬拒过旧 ms-swift --------------------
: "${MIN_MS_SWIFT:=4.2.0}"
: "${MIN_TRANSFORMERS:=5.2.0}"
echo "== 版本锁定 (ms-swift >= $MIN_MS_SWIFT, transformers >= $MIN_TRANSFORMERS) =="
for required in "$TRAIN_FILE" "$VAL_FILE"; do
  if [ ! -s "$required" ]; then
    echo "FAIL: 训练数据不存在或为空: $required"
    exit 4
  fi
done
python - "$MIN_MS_SWIFT" "$MIN_TRANSFORMERS" <<'PY'
import importlib.metadata as m, sys
try:
    from packaging.version import Version
except Exception:
    print("FAIL: 缺 packaging"); sys.exit(4)
min_swift, min_transformers = sys.argv[1:3]
snap = {}
for p in (
    "ms-swift",
    "transformers",
    "peft",
    "torch",
    "trl",
    "accelerate",
    "qwen-vl-utils",
    "decord",
):
    try: snap[p] = m.version(p)
    except Exception: snap[p] = None
for p, v in snap.items(): print(f"  {p} == {v}")
if not snap.get("ms-swift"):
    print("FAIL: ms-swift 未安装"); sys.exit(4)
if Version(snap["ms-swift"]) < Version(min_swift):
    print(
        f"FAIL: ms-swift {snap['ms-swift']} < {min_swift} "
        "(Qwen3.6 需要 ms-swift 4.2+)"
    )
    sys.exit(4)
if not snap.get("transformers") or Version(snap["transformers"]) < Version(min_transformers):
    print(
        f"FAIL: transformers {snap.get('transformers')} < {min_transformers} "
        "(Qwen3.6-35B-A3B 的模型加载要求)"
    )
    sys.exit(4)
for required in ("qwen-vl-utils", "decord"):
    if not snap.get(required):
        print(f"FAIL: 缺 Qwen3.6 模型依赖: {required}")
        sys.exit(4)
PY
mkdir -p "$PREFLIGHT_DIR" "$OUTPUT_DIR"

# ---- ①b 输出目录归属: 一个目录只属于一个 model+run ---------------------------
# OUTPUT_DIR is model- and run-tagged by default, but it is also overridable, so
# the ownership claim is written down and checked rather than assumed.
OWNER_FILE="$OUTPUT_DIR/.run_owner.json"
if [ -f "$OWNER_FILE" ]; then
  python - "$OWNER_FILE" "$MODEL" "$RUN_ID" <<'PY'
import json, sys
owner_file, model, run_id = sys.argv[1:4]
with open(owner_file, encoding="utf-8") as handle:
    owner = json.load(handle)
if owner.get("model") != model or owner.get("run_id") != run_id:
    print(
        "FAIL: 输出目录已被别的运行占用 —— 9B 基线与 35B-A3B 对照不能共享输出目录。\n"
        f"  已占用: model={owner.get('model')} run_id={owner.get('run_id')}\n"
        f"  本次:   model={model} run_id={run_id}\n"
        "  设置 OUTPUT_DIR 或 RUN_ID 后重跑。"
    )
    sys.exit(9)
print("输出目录归属一致")
PY
else
  printf '{"model": "%s", "model_tag": "%s", "run_id": "%s"}\n' \
    "$MODEL" "$MODEL_TAG" "$RUN_ID" > "$OWNER_FILE"
fi

# ---- ①c 冻结 manifest 校验 (有就必须过) -------------------------------------
if [ "${RUN_FULL:-0}" = "1" ] && [ -z "$MANIFEST_FILE" ]; then
  echo "FAIL: RUN_FULL=1 requires MANIFEST_FILE; formal training cannot use mutable JSONL"
  exit 6
fi
if [ "${RUN_FULL:-0}" = "1" ] && [ -z "$SEALED_FILE" ]; then
  echo "FAIL: RUN_FULL=1 requires SEALED_FILE; regression data cannot substitute for the sealed gate"
  exit 6
fi
if [ -n "$MANIFEST_FILE" ]; then
  echo "== 冻结 manifest 校验: $MANIFEST_FILE =="
  MANIFEST_ARGS=(
    --manifest "$MANIFEST_FILE" \
    --expect-train "$TRAIN_FILE" \
    --expect-val "$VAL_FILE" \
    --json "$PREFLIGHT_DIR/manifest_check.json"
  )
  if [ -n "$SEALED_FILE" ]; then
    MANIFEST_ARGS+=(--expect-sealed "$SEALED_FILE")
  fi
  python -m data.sft.verify_frozen_manifest "${MANIFEST_ARGS[@]}"
else
  echo "== 未提供 MANIFEST_FILE: 本次运行不受冻结数据集契约约束 (仅允许环境 preflight) =="
fi

# ---- ①f ChatML 投影 ---------------------------------------------------------
# 冻结的 train 分片把溯源信息和对话放在一起，而 lineage 在不同行上形状不同
# (5700 行 {builder,builder_version}，411 行多一个 clarification_trope)。
# Arrow 的 struct 是强类型的，datasets 从文件头推断出两键结构后，读到三键的行
# 就会 cast 失败，训练在加载数据集时崩溃、模型都没加载。
#
# 训练只读 messages。改写冻结文件会让 SHA-256 与 manifest 对不上，等于把已审计的
# 数据身份改掉；所以冻结字节保持原样，这里派生一份只含 messages 的副本喂给 swift。
# 投影会逐行校验：行数一致、每行 messages 与原文完全相同，不一致就退出。
PROJECTED_TRAIN="$PREFLIGHT_DIR/train_chatml_projected.jsonl"
PROJECTED_VAL="$PREFLIGHT_DIR/val_chatml_projected.jsonl"
echo "== ChatML 投影 (冻结文件只读, 派生副本供 datasets 加载) =="
python -m data.sft.project_chatml --source "$TRAIN_FILE" --target "$PROJECTED_TRAIN"   --json "$PREFLIGHT_DIR/projection_train.json"
python -m data.sft.project_chatml --source "$VAL_FILE" --target "$PROJECTED_VAL"   --json "$PREFLIGHT_DIR/projection_val.json"
SWIFT_TRAIN_FILE="$PROJECTED_TRAIN"
SWIFT_VAL_FILE="$PROJECTED_VAL"
if [ "${RUN_FULL:-0}" = "1" ] && [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "FAIL: formal training requires a clean tracked worktree; commit or revert code changes first"
  exit 6
fi

# ---- ①d ms-swift 参数名核对 --------------------------------------------------
# 这些名字在 ms-swift 版本之间改过 (--train_type / --tuner_type 是已知的一对)。
# 本地 conda env 没有 ms-swift，无法离线确认，所以问装好的 CLI 而不是猜。
echo "== ms-swift 参数名核对 (待云端执行; 本地无 ms-swift 时此步会失败并说明原因) =="
python -m data.sft.check_swift_flags --subcommand sft \
  --json "$PREFLIGHT_DIR/swift_flags_sft.json" \
  --flags model tuner_type dataset val_dataset torch_dtype \
          enable_thinking freeze_vit freeze_aligner \
          lora_rank lora_alpha lora_dropout target_modules \
          max_length per_device_train_batch_size \
          gradient_accumulation_steps learning_rate \
          gradient_checkpointing seed report_to \
          max_steps logging_steps save_steps save_total_limit \
          output_dir num_train_epochs per_device_eval_batch_size \
          warmup_ratio weight_decay eval_steps \
          create_checkpoint_symlink dataloader_num_workers
python -m data.sft.check_swift_flags --subcommand infer \
  --json "$PREFLIGHT_DIR/swift_flags_infer.json" \
  --flags adapters val_dataset enable_thinking max_new_tokens result_path

# 只允许在 AMD ROCm 训练实例运行，并落盘设备/依赖/数据指纹。
python - "$TRAIN_FILE" "$VAL_FILE" "$SEALED_FILE" "$MODEL" "$MODEL_TAG" "$PREFLIGHT_DIR" "$RUN_RECORD" \
        "$RUN_ID" "$SEED" "$MANIFEST_FILE" "$OUTPUT_DIR" <<'PY'
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import torch

(
    train_file,
    val_file,
    sealed_file,
    model,
    model_tag,
    preflight_dir,
    output_file,
    run_id,
    seed,
    manifest_file,
    output_dir,
) = sys.argv[1:12]
hip = getattr(torch.version, "hip", None)
if not torch.cuda.is_available() or not hip:
    print(
        "FAIL: 50-step preflight 只允许在 AMD ROCm 实例运行；"
        f"cuda_available={torch.cuda.is_available()} hip={hip!r} "
        f"cuda={getattr(torch.version, 'cuda', None)!r}"
    )
    sys.exit(5)

def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def git(*args: str):
    """Repo identity. A run whose code version is unknown cannot be reproduced."""
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=30, check=False
        )
    except Exception:
        return None
    return out.stdout.strip() or None

def git_succeeds(*args: str) -> bool:
    """Run a provenance predicate without confusing empty stdout with failure."""
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=30, check=False
        )
    except Exception:
        return False
    return out.returncode == 0

def package_version(name: str):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

git_sha = git("rev-parse", "HEAD")
generator_commit = None
if manifest_file:
    with open(manifest_file, encoding="utf-8") as handle:
        frozen_manifest = json.load(handle)
    generator_commit = str(frozen_manifest.get("generator_commit") or "")
    if not git_sha or not generator_commit:
        print(
            "FAIL: training code or frozen-data provenance is unknown: "
            f"manifest={generator_commit or '-'} checkout={git_sha or '-'}"
        )
        sys.exit(6)
    if not git_succeeds("cat-file", "-e", f"{generator_commit}^{{commit}}"):
        print(
            "FAIL: the manifest generator commit is unavailable in this checkout: "
            f"manifest={generator_commit}; fetch/deepen history before training"
        )
        sys.exit(6)
    if not git_succeeds("merge-base", "--is-ancestor", generator_commit, git_sha):
        print(
            "FAIL: training code does not descend from the frozen-data generator: "
            f"manifest={generator_commit} checkout={git_sha}"
        )
        sys.exit(6)
manifest = {
    "run_id": run_id,
    "seed": int(seed),
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_hip": hip,
    "torch_cuda": getattr(torch.version, "cuda", None),
    "packages": {
        name: package_version(name)
        for name in (
            "ms-swift",
            "transformers",
            "peft",
            "trl",
            "accelerate",
            "qwen-vl-utils",
            "decord",
        )
    },
    "started_at": datetime.now(timezone.utc).isoformat(),
    "device_count": torch.cuda.device_count(),
    "devices": [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
        }
        for index in range(torch.cuda.device_count())
    ],
    "model": model,
    "model_tag": model_tag,
    "git_sha": git_sha,
    "dataset_generator_commit": generator_commit,
    "git_dirty": bool(git("status", "--porcelain")),
    "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "preflight_dir": os.path.abspath(preflight_dir),
    "output_dir": os.path.abspath(output_dir),
    "manifest_file": os.path.abspath(manifest_file) if manifest_file else None,
    "manifest_sha256": sha256(manifest_file) if manifest_file else None,
    "train_file": os.path.abspath(train_file),
    "train_sha256": sha256(train_file),
    "val_file": os.path.abspath(val_file),
    "val_sha256": sha256(val_file),
    "sealed_file": os.path.abspath(sealed_file) if sealed_file else None,
    "sealed_sha256": sha256(sealed_file) if sealed_file else None,
    "runtime_env": {
        key: os.environ.get(key)
        for key in (
            "HIP_VISIBLE_DEVICES",
            "NPROC_PER_NODE",
            "HSA_NO_SCRATCH_RECLAIM",
            "NVTE_USE_GROUPED_GEMM_TRITON",
            "USE_MCORE_GDN",
        )
    },
}
with open(output_file, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2)
print(
    "AMD ROCm 环境通过: "
    f"hip={hip} devices={[item['name'] for item in manifest['devices']]}"
)
print(f"运行记录 (model/git/data/manifest/seed/环境): {output_file}")
if not git_sha:
    print("WARN: 取不到 git SHA —— 这次运行无法被精确复现")
PY
python -m pip freeze > "$PREFLIGHT_DIR/pip-freeze.txt"

# ---- 公共 LoRA 配置 ---------------------------------------------------------
# ⚠MoE + 视觉模型注意:
#   --freeze_vit true      冻结视觉塔 (我们只做文本 planner, 不训视觉/aligner)
#   --target_modules all-linear  仅 LLM 线性层加 LoRA; 若 preflight loss=0,
#     多半是 MoE 专家/router 未被覆盖或 template 不对 → 显式加专家线性层并升级 swift
#
# thinking: Qwen3.6 默认可能输出 <think> 块。训练目标里没有 think 块，所以训练与
#   推理都显式关掉，不再只写在注释里。关掉这件事是对运行时的断言，只能由运行时
#   证实 —— 见 ⑤ 的 verify_infer_output（**待云端执行**，本机没有 GPU 也没有
#   ms-swift，不跑任何付费 API）。
#
# ⚠**待云端核对** 的参数名 (①d 会用 `swift <sub> --help` 自动核对并给出别名):
#   --tuner_type      LoRA 选择在 ms-swift 版本间用过 --train_type / --sft_type
#   --enable_thinking 某些版本只在 infer 侧接受, sft 侧要走 template_kwargs
#   两者都不在这里猜: ①d 失败会在烧掉 GPU 时间之前指名道姓地告诉你换哪一个。
COMMON=(
  --model "$MODEL"
  --tuner_type lora
  --dataset "$SWIFT_TRAIN_FILE"
  --val_dataset "$SWIFT_VAL_FILE"
  --torch_dtype bfloat16
  --enable_thinking false
  --freeze_vit true
  --freeze_aligner true
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05
  --target_modules all-linear
  --max_length 2048
  --per_device_train_batch_size 4
  --gradient_accumulation_steps 4
  --learning_rate 1e-4
  --gradient_checkpointing true
  --seed "$SEED"
  --report_to none
)

# ---- ②PREFLIGHT: 50 步冒烟, 硬门 (真实退出码 + 结构化日志) ------------------
echo "== PREFLIGHT: 50-step 冒烟 =="
# Timestamped so the gate below can only read logs this invocation produced. A
# unique PREFLIGHT_DIR already makes a collision unlikely; the mtime floor makes
# a leftover file unusable rather than merely improbable.
PREFLIGHT_STARTED_AT="$(date +%s)"
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
python - "$PREFLIGHT_DIR" "$PREFLIGHT_STARTED_AT" <<'PY'
import glob, json, math, os, sys
d = sys.argv[1]
started_at = float(sys.argv[2])
found = glob.glob(os.path.join(d, "**", "logging.jsonl"), recursive=True)
if not found:
    print(f"PREFLIGHT FAIL: 未找到本次运行的 logging.jsonl: {d}")
    sys.exit(3)
# One run, one log. PREFLIGHT_DIR is already unique per invocation, but swift can
# leave a nested vN- directory beside the top-level one, and a re-used directory
# can still hold a previous attempt's file. Only files written after this
# invocation started count; merging an older one would top up the step count with
# steps this run never took. 5s of clock slack for filesystem granularity.
logs = [f for f in found if os.path.getmtime(f) >= started_at - 5]
if not logs:
    print(
        f"PREFLIGHT FAIL: 目录里的 {len(found)} 个 logging.jsonl 都早于本次运行 "
        f"(started_at={started_at:.0f}) —— 这是上一次运行的日志，不能用来判定本次"
    )
    sys.exit(3)
logs.sort(key=os.path.getmtime)
log_file = logs[-1]
if len(logs) > 1:
    print(f"NOTE: 本次运行写了 {len(logs)} 个 logging.jsonl, 只读最新的一个: {log_file}")
steps = []
for line in open(log_file, encoding="utf-8"):
    try:
        r = json.loads(line)
    except Exception:
        continue
    if isinstance(r, dict) and "loss" in r:
        steps.append(r)
if len(steps) < 10:
    print(f"PREFLIGHT FAIL: 只解析出 {len(steps)} 条含 loss 的训练记录 (<10)"); sys.exit(3)

def step_number(record):
    """Step index across the shapes ms-swift has used.

    Seen in the wild: an int under `global_step`; an int under `current_steps`;
    and the combined string "12/50" under either `global_step` or the literal
    key `global_step/max_steps`. int("12/50") raises, so the string form has to
    be handled wherever the value comes from, not only for the combined key.
    """
    for key in ("global_step", "current_steps", "global_step/max_steps", "step"):
        raw = record.get(key)
        if raw is None:
            continue
        if isinstance(raw, (int, float)):
            return int(raw)
        text = str(raw).strip()
        if not text:
            continue
        try:
            return int(float(text.split("/", 1)[0]))
        except (TypeError, ValueError):
            continue
    return 0

def max_steps_of(record):
    for key in ("max_steps", "global_step/max_steps", "total_steps"):
        raw = record.get(key)
        if raw is None:
            continue
        if isinstance(raw, (int, float)):
            return int(raw)
        parts = str(raw).split("/")
        if len(parts) == 2:
            try:
                return int(float(parts[1]))
            except (TypeError, ValueError):
                continue
    return None

steps.sort(key=step_number)
last = steps[-1]
declared_max = next(
    (m for m in (max_steps_of(s) for s in reversed(steps)) if m), None
)
if declared_max is not None and declared_max != 50:
    print(f"PREFLIGHT FAIL: 日志声明 max_steps={declared_max}, 期望 50 —— 跑的不是这次的 preflight")
    sys.exit(3)
loss = float(last.get("loss", 0) or 0)
if not math.isfinite(loss) or loss <= 0:
    print(f"PREFLIGHT FAIL: 末步 loss={loss} (应 >0 且有限) — 疑似 MoE loss=0 坑 (#8142)"); sys.exit(3)
acc = last.get("token_acc", last.get("acc"))
acc_checked = acc is not None
if acc_checked and float(acc) <= 0:
    print(f"PREFLIGHT FAIL: token_acc={acc} <= 0"); sys.exit(3)
# ms-swift versions use global_step, current_steps, or global_step/max_steps.
gstep = max(step_number(s) for s in steps)
if gstep < 50:
    print(f"PREFLIGHT FAIL: 只完成 {gstep} 步 (<50)"); sys.exit(3)
# adapter 必须同时有 config + 权重 (不接受任意 .safetensors)
cfgs = glob.glob(os.path.join(d, "**", "adapter_config.json"), recursive=True)
wts = glob.glob(os.path.join(d, "**", "adapter_model.safetensors"), recursive=True)
if not (cfgs and wts):
    print(f"PREFLIGHT FAIL: 缺 adapter_config.json({len(cfgs)}) 或 adapter_model.safetensors({len(wts)})"); sys.exit(3)
print(f"PREFLIGHT OK: steps={gstep} records={len(steps)} last_loss={loss:.4f} "
      f"token_acc={'(absent)' if not acc_checked else acc} adapter=config+weights "
      f"log={os.path.relpath(log_file, d)}")
PY
echo "== PREFLIGHT 通过 =="

if [ "${RUN_FULL:-0}" != "1" ]; then
  echo "== 只跑了 preflight. 确认无误后: RUN_FULL=1 bash $0 =="
  exit 0
fi

# A full run is allowed only on the exact data that passed preflight.
python - "$RUN_RECORD" "$TRAIN_FILE" "$VAL_FILE" "$SEALED_FILE" "$MANIFEST_FILE" "$MODEL" <<'PY'
import hashlib
import json
import sys

record_path, train_file, val_file, sealed_file, manifest_file, model = sys.argv[1:7]
with open(record_path, encoding="utf-8") as handle:
    record = json.load(handle)

def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for key, path in (("train_sha256", train_file), ("val_sha256", val_file)):
    actual = sha256(path)
    if record.get(key) != actual:
        print(f"FAIL: {path} 已在 preflight 后变化，请重新跑 preflight")
        sys.exit(6)
if record.get("sealed_sha256") != sha256(sealed_file):
    print(f"FAIL: {sealed_file} 已在 preflight 后变化，请重新跑 preflight")
    sys.exit(6)
# The frozen manifest is part of the fingerprint, not a separate courtesy check:
# editing it after preflight changes which dataset the run claims to be.
if manifest_file:
    if record.get("manifest_sha256") != sha256(manifest_file):
        print(f"FAIL: {manifest_file} 已在 preflight 后变化，请重新跑 preflight")
        sys.exit(6)
elif record.get("manifest_sha256"):
    print("FAIL: preflight 记录了一个 manifest，本次全量却没提供 MANIFEST_FILE")
    sys.exit(6)
if record.get("model") != model:
    print(f"FAIL: preflight 的 model={record.get('model')!r} 与本次 {model!r} 不同")
    sys.exit(6)
print(
    "指纹一致: "
    f"train={record['train_sha256'][:12]} val={record['val_sha256'][:12]} "
    f"manifest={(record.get('manifest_sha256') or '-')[:12]} "
    f"git={(record.get('git_sha') or '-')[:12]} seed={record.get('seed')}"
)
PY

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

BEST="$OUTPUT_DIR/best"   # --create_checkpoint_symlink 通常生成 best/last 软链
if [ ! -e "$BEST" ]; then
  BEST="$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' |
    sort -V | tail -1)"
fi
if [ -z "$BEST" ] || [ ! -e "$BEST" ]; then
  echo "FAIL: 训练完成但找不到 best 或 checkpoint-*"
  exit 7
fi
echo "== 训练完成. 最优 LoRA: $BEST =="

# ---- ④合并 (可选, 便于部署) -------------------------------------------------
# swift export --adapters "$BEST" --merge_lora true

# ---- ⑤在 eval 上出预测 → ⑥确定性打分 ---------------------------------------
echo "== 生成 eval 预测 =="
# max_new_tokens 1024: 512 truncates the longer PlannerDecisionV3 payloads, and a
# truncated JSON scores as a schema failure, which reads as a model regression.
swift infer \
  --adapters "$BEST" \
  --val_dataset "$VAL_FILE" \
  --enable_thinking false \
  --max_new_tokens 1024 \
  --result_path "$OUTPUT_DIR/eval_predictions.jsonl"

# ⑤b thinking 关闭的**运行时证据**。grep '<think>' 会漏掉只剩闭合标签、
#    reasoning_content 旁路字段、以及 ◁think▷ 全角分隔符这三种泄漏方式。
echo "== 验证真实 infer 输出不含 think 块 =="
python -m data.sft.verify_infer_output \
  --pred "$OUTPUT_DIR/eval_predictions.jsonl" \
  --schema planner_v3 \
  --json "$OUTPUT_DIR/infer_thinking_check.json"

echo "== 打分 (V3 schema/request kind/通道F1/澄清精确率/字段匹配, 强制100%覆盖) =="
python -m data.sft.score_student \
  --eval "$VAL_FILE" \
  --pred "$OUTPUT_DIR/eval_predictions.jsonl" \
  --json "$OUTPUT_DIR/eval_score.json"

echo "== 密封集最终评测（训练与选 checkpoint 均未读取该 split） =="
swift infer \
  --adapters "$BEST" \
  --val_dataset "$SEALED_FILE" \
  --enable_thinking false \
  --max_new_tokens 1024 \
  --result_path "$OUTPUT_DIR/sealed_predictions.jsonl"
python -m data.sft.verify_infer_output \
  --pred "$OUTPUT_DIR/sealed_predictions.jsonl" \
  --schema planner_v3 \
  --json "$OUTPUT_DIR/sealed_thinking_check.json"
python -m data.sft.score_student \
  --eval "$SEALED_FILE" \
  --pred "$OUTPUT_DIR/sealed_predictions.jsonl" \
  --json "$OUTPUT_DIR/sealed_score.json"

echo "== HyDE 质量另用文搜音尺子: python -m tests.eval.evaluate_alignment_attribute =="
