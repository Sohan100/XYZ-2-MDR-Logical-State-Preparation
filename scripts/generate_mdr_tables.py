"""

generate_mdr_tables.py
----------------------------------------------------------------------------
Command-line helpers for generating mdr tables artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _ensure_src_on_path() -> None:
    """
    Add the repository `src/` directory to `sys.path` if needed.

    This allows running the script directly via `python scripts/...` without
    requiring prior package installation.
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

from mdr.constants import (  # noqa: E402
    CODE_FAMILY_DISPLAY_NAMES,
    DEFAULT_DISTANCES,
    DEFAULT_TABLES_DIR,
)
from mdr.mdr_table import MDRTable  # noqa: E402
from mdr.workflows import (
    code_family_subdir,
    default_table_filename,
)  # noqa: E402


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for standalone MDR table generation.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate MDR operator tables for selected distances "
            "and code families."
        )
    )
    parser.add_argument(
        "--code-family",
        choices=sorted(CODE_FAMILY_DISPLAY_NAMES),
        default="xyz2",
    )
    parser.add_argument(
        "--distances",
        type=int,
        nargs="+",
        default=DEFAULT_DISTANCES,
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES_DIR,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing table CSVs instead of skipping them.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Generate and save MDR tables for the requested distances.
    """
    args = parse_args()
    tables_dir = code_family_subdir(args.tables_dir, args.code_family)
    tables_dir.mkdir(parents=True, exist_ok=True)

    for distance in args.distances:
        out_csv = tables_dir / default_table_filename(
            distance=distance,
            code_family=args.code_family,
        )
        print(f"-> Building {args.code_family} table for d={distance}")
        if args.force:
            table = MDRTable(distance=distance, code_family=args.code_family)
            table.save_csv(out_csv)
        else:
            MDRTable(
                distance=distance,
                save_filename=out_csv,
                code_family=args.code_family,
            )

    print("\nDone. All requested tables are available locally.")


if __name__ == "__main__":
    main()
