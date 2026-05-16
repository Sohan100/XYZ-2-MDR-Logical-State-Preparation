"""

test_mdr_table.py
----------------------------------------------------------------------------
Pytest coverage for mdr table behavior and regression checks.
"""

from __future__ import annotations

from pathlib import Path

from mdr.mdr_table import MDRTable


def test_mdr_table_save_and_reload(tmp_path: Path) -> None:
    """
    Verify generated MDR tables persist and reload correctly.

    Args:
    tmp_path: Per-test temporary directory provided by pytest.

    Returns:
    None
    """
    out_csv = tmp_path / "mdr_table_xyz2_d3.csv"
    table = MDRTable(distance=3, save_filename=out_csv)
    assert out_csv.exists()

    reloaded = MDRTable.from_csv(out_csv)
    assert reloaded.get_stabilizers() == table.get_stabilizers()
    assert (
        reloaded.get_logicals_dict()["Logical X"]
        == table.get_logicals_dict()["Logical X"]
    )
    stab_toggles, logical_toggle = reloaded.get_toggles()
    assert len(stab_toggles) == len(reloaded.get_stabilizers())
    assert isinstance(logical_toggle, str) and logical_toggle


def test_surface_mdr_table_save_and_reload(tmp_path: Path) -> None:
    """
    Surface-code MDR tables should persist and reload with family metadata.
    """
    out_csv = tmp_path / "mdr_table_surface_d3.csv"
    table = MDRTable(distance=3, save_filename=out_csv, code_family="surface")
    assert out_csv.exists()

    reloaded = MDRTable.from_csv(out_csv)
    assert reloaded.code_family == "surface"
    assert reloaded.get_logicals_dict()["Logical X"] == "X0 X5 X10"
    assert {len(spec.split()) for spec in reloaded.get_stabilizers()} == {3, 4}
    stab_toggles, logical_toggle = reloaded.get_toggles()
    assert len(stab_toggles) == len(reloaded.get_stabilizers())
    assert isinstance(logical_toggle, str) and logical_toggle
