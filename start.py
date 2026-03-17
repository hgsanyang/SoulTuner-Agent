"""
Unified entrypoint for the project.
Usage:
  python start.py --mode api
  python start.py --mode app
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _ensure_project_root_on_path() -> None:
    root = _project_root()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _check_env(required_keys: list[str]) -> bool:
    missing = [key for key in required_keys if not os.getenv(key)]
    if not missing:
        return True
    print("Missing required environment variables:")
    for key in missing:
        print(f"  - {key}")
    print("Set them in your environment or config/setting.json.")
    return False


def _start_api() -> None:
    from api.start_server import main  # local import after path setup

    main()


def _start_app() -> None:
    if not _check_env(["SILICONFLOW_API_KEY"]):
        sys.exit(1)

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "music_app.py",
                "--server.headless=true",
            ],
            check=False,
        )
    except KeyboardInterrupt:
        print("Application stopped.")
    except Exception as exc:  # pragma: no cover - fallback safety
        print(f"Failed to start app: {exc}")
        print("You can also run: streamlit run music_app.py")
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Project entrypoint.")
    parser.add_argument(
        "--mode",
        choices=["api", "app"],
        default="api",
        help="Start mode: api (FastAPI) or app (Streamlit).",
    )
    args = parser.parse_args(argv)

    _ensure_project_root_on_path()

    if args.mode == "api":
        _start_api()
        return
    _start_app()


if __name__ == "__main__":
    main()
