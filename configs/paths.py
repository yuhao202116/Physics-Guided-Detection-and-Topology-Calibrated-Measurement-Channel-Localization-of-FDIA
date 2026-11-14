"""Repository-relative storage paths, optionally overridden by environment."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PGAE_DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()
CHECKPOINT_ROOT = Path(
    os.environ.get("PGAE_CHECKPOINT_DIR", PROJECT_ROOT / "saved_model")
).expanduser().resolve()
OUTPUT_ROOT = Path(
    os.environ.get("PGAE_OUTPUT_DIR", PROJECT_ROOT / "outputs")
).expanduser().resolve()


def data_path(filename, *, required=True):
    from configs.config import sys_config

    path = DATA_ROOT / sys_config["case_name"] / filename
    if required and not path.is_file():
        raise FileNotFoundError(
            f"Missing experiment data: {path}. Prepare the arrays described in "
            "docs/DATA.md, or run `python smoke_test.py` for a synthetic check."
        )
    return path


def checkpoint_path(filename):
    from configs.config import sys_config

    return CHECKPOINT_ROOT / sys_config["case_name"] / filename
