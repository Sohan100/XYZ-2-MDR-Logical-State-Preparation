"""
robust_toggle_generator.py
────────────────────────────────────────────────────────────────────────────
Utility for generating low-weight toggles for stabilizers and Logical X.
"""

from __future__ import annotations

from collections import deque
import random
from typing import List, Tuple

import networkx as nx
import numpy as np


class RobustToggleGenerator:
    """
    Finds minimal-weight toggles using 'Fat' Radial Beam Search.
    
    Improvement:
    - If the strict shortest-path beam fails, it 'fattens' the beam 
      by including neighbors (Width=1). 
    - This fixes validity errors where the operator needed to zig-zag 
      slightly off the perfect geodesic.
    """

    # ─────────────────────────────────────────────────────────────────────
    # construction
    # ─────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        stabilizer_specs: List[str],
        logical_x_spec: str,
        num_qubits: int,
        random_seed: int | None = 0,
    ) -> None:
        """
        Build the binary symplectic system used for toggle synthesis.

        The constructor concatenates all stabilizer constraints with Logical X
        and converts them into a GF(2) matrix in interleaved column order
        `[x_0, z_0, x_1, z_1, ...]`. It also precomputes graph distances for
        beam construction and caches stabilizer vectors for local weight
        optimization.

        Args:
            stabilizer_specs: Stabilizer operator strings.
            logical_x_spec: Logical-X operator string appended as the final
                constraint.
            num_qubits: Number of data qubits in the code.
            random_seed: Seed used for deterministic tie-breaking and search
                randomization.
        """
        self.stab_specs = stabilizer_specs
        self.log_x_spec = logical_x_spec
        self.n = num_qubits
        self._rng = random.Random(random_seed)

        self.constraints = stabilizer_specs + [logical_x_spec]
        self.num_constraints = len(self.constraints)

        self.full_matrix = np.zeros(
            (self.num_constraints, 2 * self.n),
            dtype=np.uint8,
        )
        for row_idx, spec in enumerate(self.constraints):
            z_arr, x_arr = self._str_to_zx_arrays(spec)
            for qubit in range(self.n):
                self.full_matrix[row_idx, 2 * qubit] = x_arr[qubit]
                self.full_matrix[row_idx, 2 * qubit + 1] = z_arr[qubit]

        self.qubit_graph = self._build_qubit_graph(stabilizer_specs)
        self.all_pairs_dist = dict(
            nx.all_pairs_shortest_path_length(self.qubit_graph)
        )
        self.stab_vecs = [
            self._str_to_vec_standard(spec) for spec in stabilizer_specs
        ]

    # ─────────────────────────────────────────────────────────────────────
    # public api
    # ─────────────────────────────────────────────────────────────────────
    def generate_toggles(self) -> Tuple[List[str], str]:
        """
        Generate one anti-commuting toggle for each constraint row.

        For each target syndrome vector `e_i`, the search first tries narrow
        shortest-path beams, then a width-1 expanded beam, and finally a full
        unrestricted solve as fallback. Any valid solution is post-processed by
        stabilizer multiplication to reduce Pauli weight.

        Returns:
            Tuple[List[str], str]:
            `(stabilizer_toggles, logical_x_toggle)` where the final element
            corresponds to the Logical-X constraint.
        """
        toggles: List[str] = []
        for idx in range(self.num_constraints):
            target_vec = np.zeros(self.num_constraints, dtype=np.uint8)
            target_vec[idx] = 1

            op_str = self.constraints[idx]
            qubits_in_op = self._get_qubits_in_op(op_str)
            center_node = qubits_in_op[0] if qubits_in_op else 0
            layers = self._get_distance_layers(center_node)
            found_solution: str | None = None

            for layer in layers:
                if found_solution is not None:
                    break
                targets = layer[:]
                self._rng.shuffle(targets)
                for target_node in targets:
                    beam_nodes = self._get_beam(center_node, target_node)
                    sol = self._solve_restricted(beam_nodes, target_vec)
                    if sol is None:
                        fat_beam = self._expand_beam(beam_nodes)
                        sol = self._solve_restricted(fat_beam, target_vec)
                    if sol is not None:
                        best = self._optimize_weight_deep(sol)
                        found_solution = self._vec_standard_to_str(best)
                        break

            if found_solution is None:
                all_nodes = list(range(self.n))
                sol = self._solve_restricted(all_nodes, target_vec)
                if sol is None:
                    found_solution = "I"
                else:
                    best = self._optimize_weight_deep(sol)
                    found_solution = self._vec_standard_to_str(best)

            toggles.append(found_solution)

        return toggles[:-1], toggles[-1]

    # ─────────────────────────────────────────────────────────────────────
    # beam search helpers
    # ─────────────────────────────────────────────────────────────────────
    def _get_beam(self, source: int, target: int) -> List[int]:
        """
        Return nodes that lie on at least one shortest `source -> target` path.

        Args:
            source: Start qubit index.
            target: End qubit index.

        Returns:
            List[int]: Node set defining the strict radial beam. If either node
            is disconnected in the cached distance map, the fallback beam is
            `[source, target]`.
        """
        try:
            d_st = self.all_pairs_dist[source][target]
        except KeyError:
            return [source, target]

        beam: List[int] = []
        for node in range(self.n):
            try:
                ds = self.all_pairs_dist[source][node]
                dt = self.all_pairs_dist[node][target]
            except KeyError:
                continue
            if ds + dt == d_st:
                beam.append(node)
        return beam

    def _expand_beam(self, nodes: List[int]) -> List[int]:
        """
        Expand a beam by one graph hop around every node.

        Args:
            nodes: Input beam nodes.

        Returns:
            List[int]: Unique node list containing the original nodes and all
            immediate graph neighbors.
        """
        expanded = set(nodes)
        for node in nodes:
            for neighbor in self.qubit_graph.neighbors(node):
                expanded.add(neighbor)
        return list(expanded)

    def _get_distance_layers(self, start_node: int) -> List[List[int]]:
        """
        Partition graph nodes into BFS distance layers from `start_node`.

        Args:
            start_node: Root node used to define the ring ordering.

        Returns:
            List[List[int]]: `layers[k]` contains all nodes at graph distance
            `k` from `start_node`.
        """
        layers: List[List[int]] = []
        seen = {start_node}
        queue: deque[Tuple[int, int]] = deque([(start_node, 0)])
        current_dist = 0
        current_layer: List[int] = []

        while queue:
            node, dist = queue.popleft()
            if dist > current_dist:
                layers.append(current_layer)
                current_layer = []
                current_dist = dist
            current_layer.append(node)
            for neighbor in self.qubit_graph.neighbors(node):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, dist + 1))

        if current_layer:
            layers.append(current_layer)
        return layers

    # ─────────────────────────────────────────────────────────────────────
    # linear algebra and optimization
    # ─────────────────────────────────────────────────────────────────────
    def _solve_restricted(
        self,
        qubit_indices: List[int],
        target_vec: np.ndarray,
    ) -> np.ndarray | None:
        """
        Solve the syndrome equation using only columns tied to `qubit_indices`.

        The internal matrix uses interleaved column order per qubit. The
        returned vector is converted to standard stacked order `[z | x]` to
        match the rest of this module.

        Args:
            qubit_indices: Qubits whose `(x,z)` columns are enabled.
            target_vec: Desired syndrome bit vector.

        Returns:
            np.ndarray | None: A full-length binary symplectic vector in
            standard order `[z | x]`, or `None` if the restricted system is
            inconsistent.
        """
        col_indices: List[int] = []
        for q in qubit_indices:
            col_indices.extend([2 * q, 2 * q + 1])
        cols = np.array(sorted(col_indices), dtype=int)

        sub_matrix = self.full_matrix[:, cols]
        x_sub = self._solve_gf2(sub_matrix, target_vec)
        if x_sub is None:
            return None

        full_sol = np.zeros(2 * self.n, dtype=np.uint8)
        full_sol[cols] = x_sub
        z_part = full_sol[0::2]
        x_part = full_sol[1::2]
        return np.concatenate([z_part, x_part])

    def _optimize_weight_deep(
        self,
        vec: np.ndarray,
        attempts: int = 10,
    ) -> np.ndarray:
        """
        Reduce operator weight by greedy stabilizer-coset descent.

        Each restart shuffles stabilizer order and repeatedly applies any
        single stabilizer that strictly lowers weight until a local minimum is
        reached.

        Args:
            vec: Initial binary symplectic vector in standard order.
            attempts: Number of randomized restarts.

        Returns:
            np.ndarray: Best vector found across all restarts.
        """
        current_best = vec.copy()
        current_best_wt = self._get_weight(vec)
        indices = list(range(len(self.stab_vecs)))

        for _ in range(attempts):
            temp_vec = current_best.copy()
            self._rng.shuffle(indices)
            improved = True
            while improved:
                improved = False
                curr_wt = self._get_weight(temp_vec)
                for idx in indices:
                    cand = temp_vec ^ self.stab_vecs[idx]
                    cand_wt = self._get_weight(cand)
                    if cand_wt < curr_wt:
                        temp_vec = cand
                        curr_wt = cand_wt
                        improved = True
            temp_wt = self._get_weight(temp_vec)
            if temp_wt < current_best_wt:
                current_best = temp_vec
                current_best_wt = temp_wt

        return current_best

    def _get_weight(self, vec: np.ndarray) -> int:
        """
        Compute Pauli support size of a standard-order binary symplectic
        vector.

        Args:
            vec: Vector encoded as `[z | x]`.

        Returns:
            int: Number of qubits with non-identity action.
        """
        return int(np.sum(vec[: self.n] | vec[self.n:]))

    def _symp_product(self, v1: np.ndarray, v2: np.ndarray) -> int:
        """
        Compute the GF(2) symplectic inner product `<v1, v2>_s`.

        Args:
            v1: First vector in `[z | x]` order.
            v2: Second vector in `[z | x]` order.

        Returns:
            int: `1` if operators anti-commute, `0` if they commute.
        """
        left = np.sum(v1[: self.n] & v2[self.n:])
        right = np.sum(v1[self.n:] & v2[: self.n])
        return int((left + right) % 2)

    def _solve_gf2(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray,
    ) -> np.ndarray | None:
        """
        Solve `matrix * x = rhs` using Gaussian elimination over GF(2).

        Args:
            matrix: Binary coefficient matrix.
            rhs: Binary right-hand side vector.

        Returns:
            np.ndarray | None: One solution vector if the system is consistent;
            otherwise `None`.
        """
        rows, cols = matrix.shape
        augmented = np.hstack([matrix, rhs.reshape(-1, 1)])
        pivots: List[Tuple[int, int]] = []
        pivot_row = 0

        for col in range(cols):
            if pivot_row >= rows:
                break
            candidates = np.where(augmented[pivot_row:, col] == 1)[0]
            if len(candidates) == 0:
                continue
            current = int(candidates[0] + pivot_row)
            if current != pivot_row:
                augmented[[pivot_row, current]] = augmented[
                    [current, pivot_row]
                ]

            pivot_vec = augmented[pivot_row]
            rows_to_xor = (
                np.where(augmented[pivot_row + 1 :, col] == 1)[0]
                + pivot_row
                + 1
            )
            if len(rows_to_xor) > 0:
                augmented[rows_to_xor] ^= pivot_vec
            pivots.append((col, pivot_row))
            pivot_row += 1

        if np.any(augmented[pivot_row:, -1]):
            return None

        x = np.zeros(cols, dtype=np.uint8)
        for col, row in reversed(pivots):
            val = augmented[row, -1]
            dot = np.dot(augmented[row, col + 1 : cols], x[col + 1 :]) % 2
            x[col] = val ^ dot
        return x

    # ─────────────────────────────────────────────────────────────────────
    # representation helpers
    # ─────────────────────────────────────────────────────────────────────
    def _build_qubit_graph(self, specs: List[str]) -> nx.Graph:
        """
        Build the interaction graph induced by shared support in `specs`.

        Two qubits are connected if they co-appear in at least one operator
        string.

        Args:
            specs: Pauli operator strings.

        Returns:
            nx.Graph: Undirected qubit interaction graph.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(self.n))
        for op in specs:
            qubits = self._get_qubits_in_op(op)
            for i, q_i in enumerate(qubits):
                for q_j in qubits[i + 1 :]:
                    graph.add_edge(q_i, q_j)
        return graph

    def _str_to_zx_arrays(self, op_str: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a sparse Pauli string into separate `z` and `x` indicator
        arrays.

        Args:
            op_str: Sparse Pauli string like `"X0 Y3 Z9"`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: `(z, x)` arrays of length `n`.
        """
        z = np.zeros(self.n, dtype=np.uint8)
        x = np.zeros_like(z)
        for term in op_str.split():
            if not term or term == "I":
                continue
            pauli, qubit = term[0], int(term[1:])
            if pauli in "ZY":
                z[qubit] = 1
            if pauli in "XY":
                x[qubit] = 1
        return z, x

    def _str_to_vec_standard(self, op_str: str) -> np.ndarray:
        """
        Convert a sparse Pauli string to stacked binary symplectic form.

        Args:
            op_str: Sparse Pauli string.

        Returns:
            np.ndarray: Vector in standard order `[z | x]`.
        """
        z, x = self._str_to_zx_arrays(op_str)
        return np.concatenate([z, x])

    def _vec_standard_to_str(self, vec: np.ndarray) -> str:
        """
        Convert a `[z | x]` binary symplectic vector to sparse Pauli text.

        Args:
            vec: Vector in standard order.

        Returns:
            str: Sparse Pauli string, or `"I"` if weight is zero.
        """
        z = vec[: self.n]
        x = vec[self.n:]
        terms: List[str] = []
        for i in range(self.n):
            if x[i] and z[i]:
                terms.append(f"Y{i}")
            elif x[i]:
                terms.append(f"X{i}")
            elif z[i]:
                terms.append(f"Z{i}")
        return " ".join(terms) if terms else "I"

    def _get_qubits_in_op(self, op_str: str) -> List[int]:
        """
        Extract sorted qubit indices referenced by a sparse Pauli string.

        Args:
            op_str: Sparse Pauli operator.

        Returns:
            List[int]: Qubit indices in the order they appear.
        """
        qubits: List[int] = []
        for term in op_str.split():
            if term and term != "I":
                qubits.append(int(term[1:]))
        return qubits
