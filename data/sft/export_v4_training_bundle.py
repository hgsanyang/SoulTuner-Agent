"""Create a verified private archive for transfer to the AMD training host."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile

from data.sft.verify_frozen_manifest import EXIT_OK, check_manifest, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def export_bundle(manifest: Path, output: Path) -> dict:
    manifest = manifest if manifest.is_absolute() else PROJECT_ROOT / manifest
    output = output if output.is_absolute() else PROJECT_ROOT / output
    code, report = check_manifest(manifest, root=PROJECT_ROOT)
    if code != EXIT_OK:
        raise ValueError(f"manifest is not training-ready: {report['problems'][:5]}")
    if "private" not in {part.casefold() for part in output.parts}:
        raise ValueError("training bundle must stay in a private path")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    files = [manifest]
    for split in payload["splits"].values():
        path = Path(split["path"])
        files.append(path if path.is_absolute() else PROJECT_ROOT / path)
    validator_report = (payload.get("validator") or {}).get("report_path")
    if validator_report:
        path = Path(validator_report)
        files.append(path if path.is_absolute() else PROJECT_ROOT / path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    sidecar = output.with_suffix(output.suffix + ".sha256")
    digest = sha256_file(output)
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    return {"archive": str(output), "sha256": digest, "bytes": output.stat().st_size, "files": len(files)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_bundle(args.manifest, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
