"""

mdr_table.py
----------------------------------------------------------------------------
mdr_table_generator.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import stim

from .constants import SUPPORTED_CODE_FAMILIES
from .robust_toggle_generator import RobustToggleGenerator
from surface_code.logical_generator import (
    SurfaceCodeLogicalGenerator,
)
from surface_code.stabilizer_generator import (
    SurfaceCodeStabilizerGenerator,
)
from xyz2.logical_generator import XYZ2LogicalGenerator
from xyz2.stabilizer_generator import XYZ2StabilizerGenerator


class MDRTable:
    """
    A unified container that generates, aligns, and serializes the full
    algebraic definition of an MDR code instance.

    Attributes
    ----------
    d : int Code distance associated with the generated or loaded table.
    n_qubits : int Number of physical qubits in the code instance.
    save_filename : Path | None Optional deferred save target used during table
    generation. df : pd.DataFrame Dataframe containing one row per stabilizer
    or logical operator.

    Methods
    -------
    __init__(...) Generate the full stabilizer, logical, and toggle table for a
    given code distance. from_csv(csv_path) Load a serialized table from CSV
    without regenerating operators. _compile_data() Build the internal
    dataframe from the stabilizer, logical, and toggle generators.
    _handle_save_request() Write the generated table to disk when a deferred
    save target is provided and does not already exist. get_stabilizers()
    Return the stabilizer operator strings in table order. get_logicals_dict()
    Return the logical operators as a label-to-operator mapping. get_toggles()
    Return the stabilizer toggles and the Logical-X toggle. get_table() Return
    the underlying dataframe representation of the table. save_csv(filename)
    Save the table to CSV. save_latex(filename) Save the table to LaTeX tabular
    format.
    """

    # ─────────────────────────────────────────────────────────────────────
    # construction
    # ─────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        distance: int,
        save_filename: Optional[str | Path] = None,
        code_family: str = "xyz2",
    ) -> None:
        """
        Build an MDR operator table for a given code distance.

        This constructor generates stabilizers, logicals, and corresponding
        toggles, then stores them in a single tabular structure.

        Args:
        distance: Code distance `d` (odd integer, `d >= 3`). save_filename:
        Optional output path. If provided and the file does not already exist,
        the generated table is written to CSV. code_family: Code family to
        generate. Supported values are `"xyz2"` and `"surface"`.

        Raises:
        ValueError: If `distance` is even or less than 3.
        """
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Distance d must be odd and >= 3")
        if code_family not in SUPPORTED_CODE_FAMILIES:
            raise ValueError(
                "code_family must be one of: "
                + ", ".join(SUPPORTED_CODE_FAMILIES)
            )

        self.d = distance
        self.code_family = code_family
        if code_family == "xyz2":
            self.n_qubits = 2 * distance * distance
        else:
            self.n_qubits = distance * distance + (distance - 1) * (
                distance - 1
            )
        self.save_filename = (
            Path(save_filename) if save_filename is not None else None
        )
        self.df = pd.DataFrame()
        self._compile_data()
        if self.save_filename is not None:
            self._handle_save_request()

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "MDRTable":
        """
        Load a precomputed table from CSV without regenerating operators.

        Args:
        csv_path: Path to an existing MDR table CSV file.

        Returns:
        MDRTable: Instance populated with the loaded dataframe.

        Notes: The loaded instance does not recover the original `distance` or
        `n_qubits` metadata from the CSV. Those fields are set to sentinel
        values because the serialized table itself is treated as the
        authoritative representation.
        """
        obj = cls.__new__(cls)
        obj.d = -1
        obj.n_qubits = -1
        obj.code_family = "xyz2"
        obj.save_filename = None
        obj.df = pd.read_csv(csv_path)
        if "CodeFamily" in obj.df.columns and not obj.df.empty:
            obj.code_family = str(obj.df["CodeFamily"].iloc[0])
        obj.n_qubits = cls._infer_num_qubits_from_dataframe(obj.df)
        return obj

    @staticmethod
    def _infer_num_qubits_from_dataframe(df: pd.DataFrame) -> int:
        """
        Infer the data-register size from serialized operators and toggles.
        """
        max_idx = -1
        for column in ("Operator", "Toggle"):
            if column not in df.columns:
                continue
            for value in df[column].dropna():
                for token in str(value).split():
                    if token in {"-", "I"}:
                        continue
                    if (
                        len(token) < 2
                        or token[0].upper() not in {"X", "Y", "Z"}
                    ):
                        continue
                    max_idx = max(max_idx, int(token[1:]))
        return max_idx + 1 if max_idx >= 0 else -1

    # ─────────────────────────────────────────────────────────────────────
    # internal compile helpers
    # ─────────────────────────────────────────────────────────────────────
    def _compile_data(self) -> None:
        """
        Generate and align stabilizers, logicals, and toggles into `self.df`.

        The resulting dataframe has one row per operator with columns:
        `Category`, `Label`, `Operator`, and `Toggle`.

        Raises:
        ValueError: If Logical X is unavailable from the logical generator
        output.
        """
        if self.code_family == "xyz2":
            stabs = XYZ2StabilizerGenerator(self.d).generate_stabilizers()
            logicals = XYZ2LogicalGenerator(self.d).generate_logicals()
        else:
            stabs = SurfaceCodeStabilizerGenerator(
                self.d
            ).generate_stabilizers()
            logicals = SurfaceCodeLogicalGenerator(self.d).generate_logicals()
        logical_x = logicals.get("Logical X")
        if not logical_x:
            raise ValueError(
                "Logical X not found in logical generator output."
            )

        if self.code_family == "xyz2":
            stab_toggles, log_x_toggle = RobustToggleGenerator(
                stabs,
                logical_x,
                self.n_qubits,
            ).generate_toggles()
        else:
            stab_toggles, log_x_toggle = self._build_tableau_toggles(
                stabs=stabs,
                logical_x=logical_x,
            )

        rows: List[Dict[str, Any]] = []
        for idx, (stab, toggle) in enumerate(zip(stabs, stab_toggles)):
            rows.append(
                {
                    "CodeFamily": self.code_family,
                    "Category": "Stabilizer",
                    "Label": f"S_{idx}",
                    "Operator": stab,
                    "Toggle": toggle,
                }
            )

        rows.append(
            {
                "CodeFamily": self.code_family,
                "Category": "Logical",
                "Label": "Logical X",
                "Operator": logical_x,
                "Toggle": log_x_toggle,
            }
        )

        for label in ("Logical Y", "Logical Z"):
            if label in logicals:
                rows.append(
                    {
                        "CodeFamily": self.code_family,
                        "Category": "Logical",
                        "Label": label,
                        "Operator": logicals[label],
                        "Toggle": "-",
                    }
                )

        self.df = pd.DataFrame(rows)

    def _build_tableau_toggles(
        self,
        stabs: List[str],
        logical_x: str,
    ) -> Tuple[List[str], str]:
        """
        Build the dual toggle basis directly from Stim's tableau.
        """
        constraints = stabs + [logical_x]
        tableau = stim.Tableau.from_stabilizers(
            [
                self._sparse_pauli_to_stim(spec, self.n_qubits)
                for spec in constraints
            ]
        )
        toggles = [
            self._stim_pauli_to_sparse(tableau.x_output(idx))
            for idx in range(len(constraints))
        ]
        return toggles[:-1], toggles[-1]

    @staticmethod
    def _sparse_pauli_to_stim(
        spec: str,
        num_qubits: int,
    ) -> stim.PauliString:
        """
        Convert a sparse project-format Pauli string into a dense Stim object.
        """
        chars = ["_"] * num_qubits
        if spec.strip() and spec.strip() != "I":
            for token in spec.split():
                chars[int(token[1:])] = token[0]
        return stim.PauliString("+" + "".join(chars))

    @staticmethod
    def _stim_pauli_to_sparse(pauli: stim.PauliString) -> str:
        """
        Convert a dense Stim Pauli string into the project's sparse format.
        """
        dense = str(pauli)
        if dense and dense[0] in "+-":
            dense = dense[1:]
        tokens = [
            f"{gate}{idx}" for idx, gate in enumerate(dense) if gate != "_"
        ]
        return " ".join(tokens) if tokens else "I"

    def _handle_save_request(self) -> None:
        """
        Handle deferred save behavior for `save_filename`.

        If the target file already exists, the write is skipped to preserve
        existing data. Otherwise, the current table is written via
        :meth:`save_csv`.

        Returns:
        None
        """
        assert self.save_filename is not None
        if self.save_filename.exists():
            print(
                "   -> Table already exists at: "
                f"{self.save_filename.resolve()}"
            )
            print("   -> Skipping write operation.")
            return
        self.save_csv(self.save_filename)

    # ─────────────────────────────────────────────────────────────────────
    # extraction api
    # ─────────────────────────────────────────────────────────────────────
    def get_stabilizers(self) -> List[str]:
        """
        Return stabilizer operator strings from the compiled table.

        This is a convenience accessor over rows where `Category ==
        "Stabilizer"`.

        Returns:
        List[str]: Stabilizer operators in table order.
        """
        return self.df[self.df["Category"] == "Stabilizer"][
            "Operator"
        ].tolist()

    def get_logicals_dict(self) -> Dict[str, str]:
        """
        Return logical operators as a label-to-operator mapping.

        The mapping is built from rows where `Category == "Logical"` and
        preserves table labels such as `"Logical X"`, `"Logical Y"`, and
        `"Logical Z"`.

        Returns:
        Dict[str, str]: Mapping such as `{"Logical X": "...", ...}`.
        """
        logical_df = self.df[self.df["Category"] == "Logical"]
        return pd.Series(
            logical_df.Operator.values,
            index=logical_df.Label,
        ).to_dict()

    def get_toggles(self) -> Tuple[List[str], str]:
        """
        Return stabilizer toggles and the Logical-X toggle.

        Returns:
        Tuple[List[str], str]: Pair `(stabilizer_toggles, logical_x_toggle)`.

        Raises:
        ValueError: If the Logical-X row is missing from the table.
        """
        stab_toggles = self.df[self.df["Category"] == "Stabilizer"][
            "Toggle"
        ].tolist()
        logical_x_row = self.df[self.df["Label"] == "Logical X"]
        if logical_x_row.empty:
            raise ValueError("Logical X missing from table.")
        return stab_toggles, str(logical_x_row["Toggle"].values[0])

    # ─────────────────────────────────────────────────────────────────────
    # io methods
    # ─────────────────────────────────────────────────────────────────────
    def get_table(self) -> pd.DataFrame:
        """
        Return the underlying dataframe.

        The returned object is the live internal dataframe, so downstream
        mutations will affect the table instance.

        Returns:
        pd.DataFrame: Full MDR operator table.
        """
        return self.df

    def save_csv(self, filename: str | Path) -> None:
        """
        Save the MDR table to CSV.

        Parent directories are created automatically when needed.

        Args:
        filename: Destination CSV path.

        Returns:
        None
        """
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path, index=False)
        print(f"   -> Table saved locally to: {path.resolve()}")

    def save_latex(self, filename: str | Path) -> None:
        """
        Save the MDR table as a LaTeX tabular file.

        The dataframe is exported with `escape=False` so Pauli strings remain
        unescaped for direct manuscript usage.

        Args:
        filename: Destination `.tex` path.
        """
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        latex_code = self.df.to_latex(index=False, escape=False)
        path.write_text(latex_code, encoding="utf-8")
        print(f"   -> LaTeX table saved locally to: {path.resolve()}")
