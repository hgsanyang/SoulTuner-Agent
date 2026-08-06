#!/usr/bin/env python
"""Fail closed around ``swift infer`` instead of repairing its output afterwards.

The 9B evaluation handed ms-swift a 226-row file and got back 452 rows: the same
226 questions in the same order, twice, with different generations (only 4 of the
226 pairs were identical). swift reported ``num_samples: 226`` and its progress
bar ran 226 iterations, so nothing in its own output said anything was wrong. The
scorer kept the first prediction for each gold row and counted the rest as
duplicates, which drove ``coverage.complete`` to false — the only reason it was
noticed at all.

Splitting that file back into two passes and scoring each was a workaround, and a
workaround is the wrong shape here: it silently decides which of two different
answers to a question is *the* answer. This refuses instead, and says why.

Three checks, all fail-closed:

* ``--reserve`` before inference: a result path that already exists is never
  written to. Reusing one is how a killed run's partial output gets concatenated
  with a new run's, producing a file no downstream check can interpret.
* ``--input``/``--pred`` after inference: the prediction file must have exactly
  as many rows as the input, every row must map to a distinct input row, and
  none may be missing or extra.
* ``--record`` writes the parameters inference actually ran with into the run
  record, so a score six weeks from now can be attributed to a decoding config.

Exit codes: 0 clean, 4 unusable input, 8 the contract was broken.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.sft.score_student import _row_keys  # noqa: E402  (path set above)

EXIT_OK = 0
EXIT_BAD_INPUT = 4
EXIT_CONTRACT = 8


def _rows(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def reserve(target: Path) -> list[str]:
    """A result path must be new. An existing one is never appended to or reused."""
    if target.exists():
        return [
            f"result path already exists: {target} "
            f"({target.stat().st_size} bytes). Inference must write a fresh path; "
            "reusing one lets a previous run's rows be mistaken for this one's."
        ]
    return []


def check_contract(input_path: Path, pred_path: Path) -> dict:
    """Predictions must correspond one-to-one with the inputs, in count and identity."""
    inputs = _rows(input_path)
    preds = _rows(pred_path)

    index: dict[str, int] = {}
    for position, row in enumerate(inputs):
        for key in _row_keys(row):
            index.setdefault(key, position)

    seen: dict[int, int] = {}
    unmatched = 0
    for row in preds:
        position = next((index[k] for k in _row_keys(row) if k in index), None)
        if position is None:
            unmatched += 1
        else:
            seen[position] = seen.get(position, 0) + 1

    repeated = sum(count - 1 for count in seen.values() if count > 1)
    missing = len(inputs) - len(seen)

    problems: list[str] = []
    if len(preds) != len(inputs):
        problems.append(
            f"prediction row count {len(preds)} != input row count {len(inputs)}. "
            "Do not split, deduplicate or trim the file to make it fit — a run that "
            "answered every question more than once did not answer it once."
        )
    if repeated:
        problems.append(f"{repeated} prediction rows map to an input row already answered")
    if unmatched:
        problems.append(f"{unmatched} prediction rows match no input row")
    if missing:
        problems.append(f"{missing} input rows received no prediction")

    return {
        "input": str(input_path),
        "input_rows": len(inputs),
        "pred": str(pred_path),
        "pred_rows": len(preds),
        "matched_inputs": len(seen),
        "repeated": repeated,
        "unmatched": unmatched,
        "missing": missing,
        "one_to_one": not problems,
        "problems": problems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reserve", type=Path,
                        help="fail if this result path already exists (run before inference)")
    parser.add_argument("--input", type=Path, help="the file inference was given")
    parser.add_argument("--pred", type=Path, help="the file inference produced")
    parser.add_argument("--record", type=str,
                        help="JSON object of the decoding parameters actually used")
    parser.add_argument("--json", dest="json_out", type=Path)
    args = parser.parse_args(argv)

    report: dict = {}
    problems: list[str] = []

    if args.reserve is not None:
        problems += reserve(args.reserve)
        report["reserved_path"] = str(args.reserve)

    if (args.input is None) != (args.pred is None):
        print("CONTRACT FAIL: --input and --pred must be given together")
        return EXIT_BAD_INPUT

    if args.input is not None:
        for path in (args.input, args.pred):
            if not path.is_file():
                print(f"CONTRACT FAIL: no such file: {path}")
                return EXIT_BAD_INPUT
        try:
            result = check_contract(args.input, args.pred)
        except json.JSONDecodeError as exc:
            print(f"CONTRACT FAIL: unreadable JSONL: {exc}")
            return EXIT_BAD_INPUT
        report.update(result)
        problems += result["problems"]

    if args.record:
        try:
            report["inference_params"] = json.loads(args.record)
        except json.JSONDecodeError as exc:
            print(f"CONTRACT FAIL: --record is not JSON: {exc}")
            return EXIT_BAD_INPUT

    report["problems"] = problems
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    if problems:
        print("CONTRACT FAIL:")
        for problem in problems:
            print(f"  - {problem}")
        return EXIT_CONTRACT

    if args.input is not None:
        print(
            f"CONTRACT OK: {report['pred_rows']} predictions for "
            f"{report['input_rows']} inputs, one to one"
        )
    else:
        print(f"CONTRACT OK: {report.get('reserved_path')} is free")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
