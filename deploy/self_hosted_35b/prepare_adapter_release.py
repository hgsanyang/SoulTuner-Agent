"""Create a minimal, sanitized PEFT adapter release without changing the checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


REQUIRED_FILES = ("adapter_model.safetensors", "adapter_config.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_release(checkpoint: Path, output: Path, base_model: str) -> list[Path]:
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    if checkpoint == output:
        raise ValueError("output must be different from the private checkpoint")
    missing = [name for name in REQUIRED_FILES if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint is missing: {', '.join(missing)}")

    output.mkdir(parents=True, exist_ok=True)
    adapter_target = output / "adapter_model.safetensors"
    shutil.copy2(checkpoint / "adapter_model.safetensors", adapter_target)

    config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = base_model
    config["inference_mode"] = True
    config_target = output / "adapter_config.json"
    config_target.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    released = [adapter_target, config_target]
    checksum_target = output / "SHA256SUMS"
    checksum_target.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in released),
        encoding="utf-8",
    )
    return [*released, checksum_target]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    args = parser.parse_args()
    for path in prepare_release(args.checkpoint, args.output, args.base_model):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
