from __future__ import annotations
import os
from pathlib import Path

def repo_root() -> Path:
    # Works in notebooks (cwd should be repo root or notebooks/)
    here = Path.cwd().resolve()
    # if we're inside /notebooks, go one level up
    if here.name.lower() == "notebooks":
        return here.parent
    return here

def get_path(env_var: str, default_rel: str) -> Path:
    root = os.getenv(env_var)
    if root:
        return Path(root).expanduser().resolve()
    return (repo_root() / default_rel).resolve()

CRISISMMD_ROOT = get_path("CRISISMMD_ROOT", "CrisisMMD_v2.0")
BRIGHT_ROOT    = get_path("BRIGHT_ROOT", "BRIGHT")
