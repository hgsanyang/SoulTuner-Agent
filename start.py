"""
Unified entrypoint for the project.
Usage:
  python start.py              # Start FastAPI backend
  python start.py --mode api   # Same as above (explicit)
"""

from __future__ import annotations

import os
import sys
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
    _ensure_project_root_on_path()
    _start_api()


if __name__ == "__main__":
    main()

