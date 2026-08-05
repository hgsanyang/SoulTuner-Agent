#!/usr/bin/env python
"""Check the ms-swift flags this repo passes actually exist on the installed CLI.

Every flag name in ``train_planner_student.sh`` was written against a version of
ms-swift nobody here can run: there is no ms-swift in the local conda env, so a
renamed argument is invisible until the cloud instance is already billing. Two
of them are known to move between releases — ``--train_type`` vs
``--tuner_type`` for LoRA selection, and whether ``--enable_thinking`` is
accepted by ``sft`` or only by ``infer``.

Rather than guess, this asks the installed CLI. ``swift <subcommand> --help`` is
free, needs no GPU, and answers the question exactly. When a flag is missing it
prints the alias this repo knows about so the operator changes one line instead
of bisecting a crash.

Exit codes: 0 all flags accepted, 4 the CLI could not be queried or the flag
list was empty, 10 a flag is not accepted by this ms-swift build.

Flag names are accepted with or without leading dashes and normalised
internally. That is not cosmetic: argparse cannot take ``--model`` as a *value*
for ``--flags`` — it reads it as the next option — so the caller has to pass
bare names, while the help text lists dashed ones. Matching one form against the
other is how this gate silently checked nothing.

    python -m data.sft.check_swift_flags --subcommand sft --flags model tuner_type
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable

EXIT_OK = 0
EXIT_UNUSABLE = 4
EXIT_UNKNOWN_FLAG = 10

# Names that are known to differ between ms-swift releases. If the configured
# flag is rejected, these are printed as the candidates to try -- they are NOT
# substituted automatically, because silently training with a different argument
# than the script says is exactly the failure this file exists to prevent.
KNOWN_ALIASES: dict[str, tuple[str, ...]] = {
    "--tuner_type": ("--train_type", "--sft_type"),
    "--train_type": ("--tuner_type", "--sft_type"),
    "--enable_thinking": ("--template_kwargs", "--model_kwargs"),
    "--target_modules": ("--lora_target_modules",),
    "--tuner_backend": (),
    "--freeze_vit": ("--freeze_vision_tower",),
    "--freeze_aligner": ("--freeze_projector",),
    "--create_checkpoint_symlink": (),
    "--result_path": ("--result_dir",),
    "--max_new_tokens": ("--max_tokens",),
}

_FLAG = re.compile(r"(--[A-Za-z0-9][A-Za-z0-9_-]*)")


def _run_help(subcommand: str) -> str:
    executable = shutil.which("swift")
    if not executable:
        raise FileNotFoundError("`swift` is not on PATH")
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [executable, subcommand, "--help"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    # argparse writes --help to stdout; some wrappers use stderr. Take both.
    help_text = f"{completed.stdout}\n{completed.stderr}"

    # ms-swift 4.4's ``swift sft`` wrapper first runs a tiny parser used only
    # to discover ``--tuner_backend``. Passing ``--help`` makes that parser
    # exit before ``sft_main`` builds the real training parser, so the wrapper
    # advertises only two flags and makes every useful flag look unsupported.
    # Query the installed pipeline entry point when that shallow-help shape is
    # detected. This is the same parser the command invokes after bootstrap;
    # it does not load a model or use the GPU.
    if subcommand == "sft" and len(parse_supported_flags(help_text)) <= 2:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                sys.executable,
                "-c",
                "from swift.pipelines import sft_main; sft_main()",
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        help_text = f"{completed.stdout}\n{completed.stderr}"
    return help_text


def parse_supported_flags(help_text: str) -> set[str]:
    return set(_FLAG.findall(help_text or ""))


def normalize_flags(flags: Iterable[str]) -> set[str]:
    """Accept ``model`` or ``--model``; compare in the dashed form the help uses.

    The caller cannot pass dashed names — argparse would read them as options —
    but the help text only ever lists dashed ones. Normalising here is what makes
    the two sides comparable at all.
    """
    return {f"--{item.strip().lstrip('-')}" for item in flags if item.strip().strip("-")}


def check_flags(
    subcommand: str,
    flags: Iterable[str],
    *,
    help_reader: Callable[[str], str] = _run_help,
) -> tuple[int, dict]:
    report: dict = {
        "subcommand": subcommand,
        "checked": sorted(normalize_flags(flags)),
        "supported": [],
        "missing": {},
        "problems": [],
    }
    # An empty list is a configuration failure, never a pass. The previous
    # version filtered on `startswith("--")` while the caller could only pass
    # bare names, so every invocation checked zero flags and reported OK — a
    # flag that does not exist in any ms-swift build sailed through.
    if not report["checked"]:
        report["problems"].append(
            "no flags were configured to check; refusing to report a passing gate"
        )
        return EXIT_UNUSABLE, report
    try:
        help_text = help_reader(subcommand)
    except Exception as exc:  # noqa: BLE001 - any failure here means "cannot ask"
        report["problems"].append(f"could not run `swift {subcommand} --help`: {exc}")
        return EXIT_UNUSABLE, report

    supported = parse_supported_flags(help_text)
    if not supported:
        report["problems"].append(
            f"`swift {subcommand} --help` produced no recognisable flags; "
            "the CLI may have failed to import"
        )
        return EXIT_UNUSABLE, report

    report["supported_count"] = len(supported)
    for flag in report["checked"]:
        if flag in supported:
            report["supported"].append(flag)
        else:
            report["missing"][flag] = [
                alias for alias in KNOWN_ALIASES.get(flag, ()) if alias in supported
            ]
    if report["missing"]:
        report["problems"].append(
            f"{len(report['missing'])} flag(s) are not accepted by this ms-swift build"
        )
        return EXIT_UNKNOWN_FLAG, report
    return EXIT_OK, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subcommand", default="sft")
    parser.add_argument("--flags", nargs="+", required=True)
    parser.add_argument("--json", dest="json_out", type=Path)
    args = parser.parse_args(argv)

    code, report = check_flags(args.subcommand, args.flags)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if code == EXIT_OK:
        print(
            f"SWIFT FLAGS OK: `swift {args.subcommand}` accepts all "
            f"{len(report['checked'])} configured flag(s)"
        )
        return EXIT_OK
    print(f"SWIFT FLAGS FAIL ({args.subcommand}):")
    for problem in report["problems"]:
        print(f"  - {problem}")
    for flag, aliases in (report.get("missing") or {}).items():
        hint = f" -> this build accepts {aliases}" if aliases else " (no known alias present)"
        print(f"  - {flag} is not accepted{hint}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
