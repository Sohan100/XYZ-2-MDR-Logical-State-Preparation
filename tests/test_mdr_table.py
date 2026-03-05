from __future__ import annotations

from pathlib import Path

from xyz2_mdr.mdr_table import MDRTable


def test_mdr_table_save_and_reload(tmp_path: Path) -> None:
    """
    Verify generated MDR tables persist and reload correctly.

    Args:
        tmp_path: Per-test temporary directory provided by pytest.

    Returns:
        None
    """
    out_csv = tmp_path / "mdr_table_d3.csv"
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
