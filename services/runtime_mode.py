"""Runtime guards shared by persistence and audit-log adapters."""

from __future__ import annotations

import os


def side_effects_disabled() -> bool:
    """Return whether the current process must avoid persistent writes.

    Import settings lazily so low-level logging helpers do not create a module
    cycle during application startup.  The environment fallback also protects
    small standalone scripts that run before settings is initialized.
    """

    try:
        from config.settings import settings

        if bool(getattr(settings, "eval_disable_side_effects", False)):
            return True
    except Exception:
        pass
    return os.getenv("EVAL_DISABLE_SIDE_EFFECTS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

