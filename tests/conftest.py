from __future__ import annotations

from pathlib import Path
import sys

import pytest

# Ensure local package is importable at module import time.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from xyz2_mdr.mdr_table import MDRTable
from xyz2_mdr.workflows import build_code_inputs


@pytest.fixture
def d3_table() -> MDRTable:
    """
    Create a distance-3 MDR table for reuse across tests.

    Returns:
        MDRTable: Fully compiled table instance for distance 3.
    """
    return MDRTable(distance=3)


@pytest.fixture
def d3_code_inputs(tmp_path: Path) -> dict[str, object]:
    """
    Build reusable distance-3 code inputs backed by a temporary table file.

    Args:
        tmp_path: Per-test temporary directory provided by pytest.

    Returns:
        dict[str, object]: Code-input mapping from `build_code_inputs`.
    """
    table_csv = tmp_path / "mdr_table_d3.csv"
    return build_code_inputs(distance=3, table_csv=table_csv)
