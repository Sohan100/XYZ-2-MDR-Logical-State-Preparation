"""
mdr_table_generator.py
────────────────────────────────────────────────────────────────────────────
Utility for compiling, archiving, and retrieving the complete algebraic 
structure (Stabilizers, Logicals, Toggles) of the MDR code instance.
"""

import pandas as pd
import os
from typing import Optional, List, Dict, Tuple, Any

class MDRTable:
    """
    A unified container that generates, aligns, and serializes the full
    algebraic definition of an MDR code instance.
    """

    def __init__(
        self, 
        distance: int,
        save_filename: Optional[str] = None
    ) -> None:
        """
        Initialize the MDRTable, compile operator data, and handle file I/O.

        Args:
            distance: Code distance (odd integer >= 3).
            save_filename: Optional CSV output path.

        Returns:
            None
        """
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Distance d must be odd and >= 3")

        self.d = distance
        self.n_qubits = 2 * distance * distance
        self.save_filename = save_filename
        self.df = pd.DataFrame()
        
        # Trigger compilation
        self._compile_data()

        # Handle saving logic
        if self.save_filename:
            self._handle_save_request()

    def _compile_data(self) -> None:
        """
        Internal orchestration method to generate and align all operators.

        This method constructs stabilizers, logical operators, and toggles,
        then assembles them into the table dataframe.

        Returns:
            None
        """
        # 1. Fetch Code Stabilizers
        s_gen = XYZ2StabilizerGenerator(self.d)
        stabs: List[str] = s_gen.generate_stabilizers()

        # 2. Fetch Logical Operators
        l_gen = XYZ2LogicalGenerator(self.d)
        logs_dict: Dict[str, str] = l_gen.generate_logicals()
        
        logical_x_str = logs_dict.get("Logical X")
        if not logical_x_str:
            raise ValueError("Logical X not found in logical generator output.")

        # 3. Generate Toggles
        # FIX: Using positional arguments to match the RobustToggleGenerator definition
        tg = RobustToggleGenerator(stabs, logical_x_str, self.n_qubits)
        
        stab_toggles, log_x_toggle = tg.generate_toggles()

        # 4. Build Table Rows
        data_rows: List[Dict[str, Any]] = []

        # A. Stabilizer Rows
        for i, (stab_op, toggle_op) in enumerate(zip(stabs, stab_toggles)):
            data_rows.append({
                "Category": "Stabilizer",
                "Label": f"S_{i}",
                "Operator": stab_op,
                "Toggle": toggle_op
            })

        # B. Logical X Row
        data_rows.append({
            "Category": "Logical",
            "Label": "Logical X",
            "Operator": logical_x_str,
            "Toggle": log_x_toggle
        })

        # C. Other Logicals
        for label in ["Logical Y", "Logical Z"]:
            if label in logs_dict:
                data_rows.append({
                    "Category": "Logical",
                    "Label": label,
                    "Operator": logs_dict[label],
                    "Toggle": "—"
                })

        self.df = pd.DataFrame(data_rows)

    def _handle_save_request(self) -> None:
        """
        Persist the compiled table unless the target file already exists.

        This mirrors the notebook workflow that avoids overwriting
        precomputed table files.

        Returns:
            None
        """
        if self.save_filename and os.path.exists(self.save_filename):
            print(f"   -> Table already exists at: {os.path.abspath(self.save_filename)}")
            print("   -> Skipping write operation.")
        elif self.save_filename:
            self.save_csv(self.save_filename)

    # ──────────────── Extraction API ────────────────
    def get_stabilizers(self) -> List[str]:
        """
        Return stabilizer operator strings from the internal table.

        These values are taken from rows where `Category == "Stabilizer"`.

        Returns:
            List[str]: Stabilizer operators in table order.
        """
        return self.df[self.df["Category"] == "Stabilizer"]["Operator"].tolist()

    def get_logicals_dict(self) -> Dict[str, str]:
        """
        Return logical operators as a label-to-operator dictionary.

        This helper converts logical rows into an easy lookup map.

        Returns:
            Dict[str, str]: Mapping from logical label to operator string.
        """
        log_df = self.df[self.df["Category"] == "Logical"]
        return pd.Series(log_df.Operator.values, index=log_df.Label).to_dict()

    def get_toggles(self) -> Tuple[List[str], str]:
        """
        Return stabilizer toggles and the Logical-X toggle string.

        The first element corresponds to stabilizers only; the second is
        the dedicated Logical-X toggle.

        Returns:
            Tuple[List[str], str]: Stabilizer toggles and Logical-X toggle.
        """
        stab_toggles = self.df[self.df["Category"] == "Stabilizer"]["Toggle"].tolist()
        log_x_row = self.df[self.df["Label"] == "Logical X"]
        if log_x_row.empty:
            raise ValueError("Logical X missing from table.")
        log_x_toggle = log_x_row["Toggle"].values[0]
        return stab_toggles, log_x_toggle

    # ──────────────── IO Methods ────────────────
    def get_table(self) -> pd.DataFrame:
        """
        Return the full dataframe backing this table object.

        The returned dataframe includes stabilizer and logical rows.

        Returns:
            pd.DataFrame: Complete MDR table.
        """
        return self.df

    def save_csv(self, filename: str) -> None:
        """
        Write the table to a CSV file on disk.

        Args:
            filename: Destination CSV path.

        Returns:
            None
        """
        self.df.to_csv(filename, index=False)
        print(f"   -> Table saved locally to: {os.path.abspath(filename)}")

    def save_latex(self, filename: str) -> None:
        """
        Write the table to a LaTeX tabular file on disk.

        Args:
            filename: Destination LaTeX file path.

        Returns:
            None
        """
        latex_code = self.df.to_latex(index=False, escape=False)
        with open(filename, "w") as f:
            f.write(latex_code)
        print(f"   -> LaTeX table saved locally to: {os.path.abspath(filename)}")
