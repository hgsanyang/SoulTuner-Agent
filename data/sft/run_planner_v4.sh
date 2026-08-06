#!/usr/bin/env bash
# Planner V4 云端入口。取代 /root/soultuner-ops/ 里那个手改 SHA 的副本。
#
# 那个副本把期望的训练提交硬编码在脚本自己里，于是每推一次修复就要在服务器上
# 手改一次——改的正是本该拦住"跑了不该跑的代码"的那道守卫。这里改成必填环境
# 变量：调用方声明要跑哪个提交，脚本只负责核对，不知道也不猜自己的 SHA。
#
# 用法：
#   EXPECTED_TRAINING_COMMIT=<sha> bash data/sft/run_planner_v4.sh                     # preflight，两模型串行
#   EXPECTED_TRAINING_COMMIT=<sha> MODELS=9b bash data/sft/run_planner_v4.sh            # preflight，只跑 9B
#   EXPECTED_TRAINING_COMMIT=<sha> RUN_FULL=1 MODELS=9b bash data/sft/run_planner_v4.sh # 正式训练，必须点名
#
# RUN_FULL 默认 0。正式训练必须显式 RUN_FULL=1 **并且**显式 MODELS=9b|35b——
# 一个实例窗口只训一个模型。脚本会为每个模型重新跑一遍 50-step preflight，
# 那不是浪费，是把本次运行的环境和数据指纹绑进记录里。
set -uo pipefail

fail() { echo "FAIL: $*" >&2; exit 3; }

: "${EXPECTED_TRAINING_COMMIT:?set EXPECTED_TRAINING_COMMIT to the commit you intend to train}"
GENERATOR_COMMIT="${GENERATOR_COMMIT:-48d87edc3fe52d52031cbb3ad78633fc5a4e54d4}"

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FROZEN="${FROZEN:-data/teacher/private/v4/frozen-v4.0.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to a writable, ideally persistent directory}"
VENV="${VENV:-/mnt/workspace/_venvs/soultuner-swift}"
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:?set MODELSCOPE_CACHE to the existing model cache}"
SEED="${SEED:-42}"
RUN_FULL="${RUN_FULL:-0}"
# 记下调用方是否显式点了名：RUN_FULL=1 下"没选"和"选了 both"要给不同的提示，
# 否则空变量会被默认值吞掉，报错说的是一件没发生的事。
MODELS_EXPLICIT=1
[ -n "${MODELS:-}" ] || MODELS_EXPLICIT=0
MODELS="${MODELS:-both}"
BASE_RUN_ID="${BASE_RUN_ID:-planner-v4-$(date -u +%Y%m%dT%H%M%SZ)}"

cd "$REPO" || fail "cannot enter repo: $REPO"

# ---- 正式训练安全门 ---------------------------------------------------------
# RUN_FULL=1 必须显式点名一个模型。`MODELS=both` 在正式训练下被拒绝：
# 一次完整对照是两个各约 3 epoch 的 run 加上推理评测，塞进一个实例窗口意味着
# 第二个模型大概率在窗口耗尽时被腰斩——而被腰斩的 35B 会留下一个看起来完整、
# 实际只训了一部分的目录。preflight 不受影响，50 步串行两个模型很便宜。
if [ "$RUN_FULL" = "1" ]; then
  case "$MODELS" in
    9b|35b) : ;;
    both)
      if [ "$MODELS_EXPLICIT" = "0" ]; then
        fail "RUN_FULL=1 requires an explicit MODELS=9b or MODELS=35b; it will not
      default to running both in one window"
      fi
      fail "RUN_FULL=1 refuses MODELS=both; run one model per instance window
      (MODELS=9b then MODELS=35b, each with its own BASE_RUN_ID)" ;;
    *)
      fail "RUN_FULL=1 requires MODELS=9b or MODELS=35b explicitly (got '$MODELS')" ;;
  esac
fi

# ---- 身份 -------------------------------------------------------------------
HEAD_SHA="$(git rev-parse HEAD)"
[ "$HEAD_SHA" = "$EXPECTED_TRAINING_COMMIT" ] ||
  fail "HEAD is $HEAD_SHA, expected $EXPECTED_TRAINING_COMMIT"
[ -z "$(git status --porcelain --untracked-files=no)" ] ||
  fail "tracked worktree is dirty; a run that cannot be reproduced is not a result"
git cat-file -e "${GENERATOR_COMMIT}^{commit}" 2>/dev/null ||
  fail "dataset generator commit $GENERATOR_COMMIT is not in this repo"
git merge-base --is-ancestor "$GENERATOR_COMMIT" HEAD ||
  fail "dataset generator is not an ancestor of the training code"

# ---- 环境 -------------------------------------------------------------------
[ -f "$VENV/bin/activate" ] || fail "missing training venv: $VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
[ -d "$MODELSCOPE_CACHE" ] || fail "missing ModelScope cache: $MODELSCOPE_CACHE"
export MODELSCOPE_CACHE

MANIFEST="$FROZEN/MANIFEST.json"
TRAIN_FILE="$FROZEN/train_v4_chatml.jsonl"
VAL_FILE="$FROZEN/regression_v4_chatml.jsonl"
SEALED_FILE="$FROZEN/sealed_v4_chatml.jsonl"
for f in "$MANIFEST" "$TRAIN_FILE" "$VAL_FILE" "$SEALED_FILE"; do
  [ -s "$f" ] || fail "missing frozen split: $f (extract the private bundle first)"
done

GATES="$OUTPUT_ROOT/$BASE_RUN_ID/gates"
mkdir -p "$GATES" || fail "cannot write to OUTPUT_ROOT: $OUTPUT_ROOT"

echo "== 依赖门 =="
python -m data.sft.check_training_deps --json "$GATES/training_deps.json" ||
  fail "training dependencies are not what this run requires"

echo "== 冻结 manifest =="
python -m data.sft.verify_frozen_manifest --manifest "$MANIFEST" \
  --expect-train "$TRAIN_FILE" --expect-val "$VAL_FILE" --expect-sealed "$SEALED_FILE" \
  --json "$GATES/manifest_check.json" || fail "frozen manifest does not match the files on disk"

echo "== 单卡 ROCm =="
python - <<'PY' || fail "ROCm/device check failed"
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False")
if not torch.version.hip:
    sys.exit("torch.version.hip is empty; this is not the ROCm build")
if torch.cuda.device_count() != 1:
    sys.exit(f"expected a single GPU, found {torch.cuda.device_count()}")
print("device OK:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GiB")
PY

# ---- 串行执行 ---------------------------------------------------------------
# 并行会让两个模型抢同一张卡的显存，峰值互相污染，对照就不再公平。
run_one() {
  local model="$1" tag="$2"
  local run_id="${BASE_RUN_ID}-${tag}"
  echo
  echo "======== $model  (run_id=$run_id, RUN_FULL=$RUN_FULL) ========"
  # 输出目录带上模型、run_id 和代码 SHA：三者任一不同就是不同的实验。
  local out="$OUTPUT_ROOT/$BASE_RUN_ID/${tag}-${HEAD_SHA:0:12}/$run_id"
  mkdir -p "$out" || fail "cannot create output dir: $out"

  MODEL="$model" \
  RUN_ID="$run_id" \
  OUTPUT_DIR="$out" \
  PREFLIGHT_DIR="$out/preflight" \
  TRAIN_FILE="$TRAIN_FILE" VAL_FILE="$VAL_FILE" SEALED_FILE="$SEALED_FILE" \
  MANIFEST_FILE="$MANIFEST" SEED="$SEED" RUN_FULL="$RUN_FULL" \
    bash data/sft/train_planner_student.sh
  local rc=$?
  echo "exit($tag)=$rc"
  [ "$rc" -eq 0 ] || fail "$model exited $rc"
}

case "$MODELS" in
  9b)   run_one "Qwen/Qwen3.5-9B" "qwen35-9b" ;;
  35b)  run_one "Qwen/Qwen3.6-35B-A3B" "qwen36-35b-a3b" ;;
  both) run_one "Qwen/Qwen3.5-9B" "qwen35-9b"
        run_one "Qwen/Qwen3.6-35B-A3B" "qwen36-35b-a3b" ;;
  *)    fail "MODELS must be 9b, 35b or both (got '$MODELS')" ;;
esac

echo
echo "ALL OK  base_run_id=$BASE_RUN_ID  code=$HEAD_SHA  RUN_FULL=$RUN_FULL"
[ "$RUN_FULL" = "1" ] || echo "只跑了 50-step preflight。正式训练需显式 RUN_FULL=1。"
