"""
Unified entrypoint for the project.
Usage:
  python start.py              # Start FastAPI backend
  python start.py --mode api   # Same as above (explicit)
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _ensure_project_root_on_path() -> None:
    root = _project_root()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _start_api() -> None:
    from api.start_server import main  # local import after path setup

    main()


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="SoulTuner backend")
    parser.add_argument("--mode", default="api", choices=["api"])
    parser.add_argument("--mock", action="store_true", help="Run without LLM, Neo4j or embedding models")
    args = parser.parse_args(argv)
    if args.mock:
        os.environ["MUSIC_MOCK_MODE"] = "1"
    _ensure_project_root_on_path()
    _start_api()


if __name__ == "__main__":
    main()

