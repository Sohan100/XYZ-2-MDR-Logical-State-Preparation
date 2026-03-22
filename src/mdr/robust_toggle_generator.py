"""
robust_toggle_generator.py
--------------------------
Protocol for synthesizing MDR recovery toggles.

This module builds one Pauli *toggle* for each measured code constraint:
every stabilizer row plus the final `Logical X` row. A toggle `T_i` is
constructed so that it

1. anti-commutes with exactly one target constraint `C_i`,
2. commutes with every other constraint row,
3. commutes with every previously accepted toggle, and
4. is as low-weight as the configured optimizer can make it.

The implementation uses the following pipeline.

Representation
--------------
Operators are stored in binary symplectic form. A Pauli on `n` qubits is
encoded as a length-`2n` vector in standard stacked order `[z | x]`, where

- `X_q` has `x[q] = 1`,
- `Z_q` has `z[q] = 1`,
- `Y_q` has both bits set.

Constraint rows are additionally assembled into an interleaved GF(2) matrix
`[x_0, z_0, x_1, z_1, ...]` because that layout is convenient for restricted
linear solves over selected qubits.

Step 1: Build the target syndrome system
----------------------------------------
Let the ordered constraint list be

`constraints = stabilizers + [logical_x]`.

For toggle `i`, the desired commutation pattern is the unit syndrome vector
`e_i`: the toggle must anti-commute with row `i` and commute with all other
rows. Solving the binary symplectic system with right-hand side `e_i`
produces a valid toggle candidate.

Step 2: Graph-guided restricted solves
--------------------------------------
The stabilizer support induces a qubit interaction graph: two qubits are
connected if they appear together in at least one stabilizer. For each target
row:

- pick the first qubit in the target operator as a center node,
- enumerate graph-distance layers from that center,
- for each candidate target node in those layers, form a *beam* consisting of
  all qubits that lie on at least one shortest path from the center to that
  node,
- try a GF(2) solve using only columns associated with those beam qubits,
- if that fails, expand the beam by one graph hop and try again,
- if all local solves fail, fall back to a full unrestricted solve over all
  qubits.

This does not change the algebraic constraint. It is only a structured way to
find compact seed solutions before any weight optimization happens.

Step 3: Weight reduction inside the commuting coset
---------------------------------------------------
Once a valid seed toggle is found, the code reduces its support by multiplying
it by operators that preserve the target syndrome. Those operators are exactly
the commuting constraints:

`coset_vecs = stabilizers + [logical_x]`.

Because each of those rows commutes with every constraint except its own dual
toggle role, XORing a valid toggle with any commuting constraint keeps the
toggle paired to the same target row. The heuristic pass does:

- greedy single-row descent: accept any single commuting constraint that
  strictly lowers support weight,
- pair-move descent: if no single move helps, try products of two commuting
  constraints and take the best improving one,
- randomized restarts: reshuffle move order several times and keep the best
  result found.

Step 4: Enforce pairwise toggle commutation
-------------------------------------------
The toggle family is built sequentially. After heuristic reduction, the new
toggle is adjusted so it commutes with every previously accepted toggle.

If the current toggle anti-commutes with an earlier toggle `T_j`, multiplying
the current toggle by the corresponding earlier constraint row `C_j`

- preserves the current target syndrome, and
- flips commutation only with `T_j`,

so this fixes pairwise toggle commutation without breaking already enforced
relations to constraints or to earlier toggles.

Step 5: Exact minimum-support refinement when available
-------------------------------------------------------
If OR-Tools CP-SAT is available and `optimization_mode="exact"`, the module
solves a binary optimization problem seeded by the heuristic toggle.

Decision variables:

- `x_q`, `z_q`: Pauli action on each qubit,
- `w_q`: whether qubit `q` is in the support.

Constraints:

- `x_q <= w_q`, `z_q <= w_q`, `w_q <= x_q + z_q`,
- symplectic parity equations enforcing the desired syndrome `e_i`,
- additional parity equations enforcing commutation with all previously
  accepted toggles,
- `sum(w_q) <= current_heuristic_weight`, so the exact solve never returns a
  heavier result than the heuristic seed.

Objective:

- minimize total support `sum(w_q)`,
- then use a deterministic lexicographic-style tie break on the `(x, z)` bits
  so equal-weight optima are reproducible.

If the exact solver finds an optimal or feasible solution within the timeout,
that refined toggle is returned. Otherwise the heuristic commuting toggle is
kept. When the first exact pass returns only a feasible incumbent, the module
can retry that row with a longer timeout schedule until the result is proven
optimal or the retry budget is exhausted.

Environment-dependent behavior
------------------------------
- If OR-Tools is installed and `optimization_mode="exact"`, the full pipeline
  runs.
- If OR-Tools is unavailable, or `optimization_mode="heuristic"`, the module
  still guarantees the required commutation relations, but the final toggle
  weights are the heuristic ones.

Outputs
-------
`generate_toggles()` returns

- a list of stabilizer toggles aligned with the stabilizer order, and
- one final toggle paired to `Logical X`.

The auxiliary field `last_optimization_statuses` records, for each generated
row, whether the result came from heuristic optimization, an exact optimal
solve, an exact feasible solve, or an exact fallback.

LaTeX-style summary
-------------------
Let the ordered constraints be

`C = (C_0, C_1, ..., C_{m-1}) = (S_0, S_1, ..., S_{m-2}, L_X)`.

For each row index `i`, the goal is to construct a Pauli toggle `T_i` such
that

`<T_i, C_j>_s = delta_{ij}`

where `<., .>_s` is the binary symplectic inner product and `delta_{ij}` is
the Kronecker delta. Sequentially, we also enforce

`<T_i, T_j>_s = 0  for all j < i`.

If exact refinement is available, the final optimization problem is

`min   sum_q w_q`

subject to

`<T, C_j>_s = delta_{ij}                  for all constraint rows j`

`<T, T_j>_s = 0                           for all previously accepted toggles`

`x_q <= w_q,  z_q <= w_q,  w_q <= x_q + z_q`

with binary decision variables `x_q, z_q, w_q in {0,1}` and Pauli vector
`T = [z | x]`.

The heuristic phase solves the same algebraic constraints but uses graph-local
GF(2) seed solves followed by greedy coset descent instead of a global binary
optimization.
"""

from __future__ import annotations

from collections import deque
import random
from typing import List, Tuple

import networkx as nx
import numpy as np

try:
    from ortools.sat.python import cp_model
except ImportError:  # pragma: no cover - optional dependency fallback
    cp_model = None


class RobustToggleGenerator:
    """
    Generate a commuting family of low-weight MDR toggles.

    For each ordered constraint row `C_i` in
    `stabilizers + [logical_x]`, this class constructs a Pauli operator
    `T_i` such that

    - `<T_i, C_i>_s = 1`,
    - `<T_i, C_j>_s = 0` for all `j != i`,
    - `<T_i, T_j>_s = 0` for all previously generated toggles `T_j`.

    Here `<., .>_s` denotes the GF(2) symplectic inner product, so `1`
    means anti-commutation and `0` means commutation.

    The actual synthesis protocol is:

    1. solve the target syndrome equation on graph-local qubit subsets,
    2. reduce weight by commuting-coset descent,
    3. enforce commutation with earlier toggles,
    4. optionally run an exact CP-SAT minimum-support refinement, retrying
       only the unproven rows with longer time limits.

    The long module docstring above describes the protocol in full detail.

    Attributes
    ----------
    stab_specs : List[str]
        Input stabilizer specifications used as toggle constraints.
    log_x_spec : str
        Logical-X specification appended as the final toggle constraint.
    n : int
        Number of physical qubits in the code.
    _rng : random.Random
        Deterministic random source used for tie-breaking and restart order.
    constraints : List[str]
        Combined list of stabilizers plus Logical X.
    num_constraints : int
        Number of syndrome constraints to satisfy.
    full_matrix : np.ndarray
        Binary symplectic matrix in interleaved column order.
    constraint_vecs : List[np.ndarray]
        Constraint vectors in standard `[z | x]` order, aligned with
        `constraints`.
    qubit_graph : nx.Graph
        Interaction graph induced by shared stabilizer support.
    all_pairs_dist : Dict[int, Dict[int, int]]
        Cached all-pairs shortest-path lengths on `qubit_graph`.
    stab_vecs : List[np.ndarray]
        Stabilizer vectors in standard `[z | x]` order.
    logical_x_vec : np.ndarray
        Logical-X constraint vector in standard `[z | x]` order.
    coset_vecs : List[np.ndarray]
        Commuting constraint vectors that preserve a toggle's syndrome label
        when multiplied into it.
    pair_move_vecs : np.ndarray
        Precomputed pairwise products of commuting constraints used to escape
        single-step local minima during weight optimization.
    optimization_mode : str
        Weight minimization strategy. `"exact"` uses CP-SAT after a heuristic
        seed when OR-Tools is available; `"heuristic"` skips the exact pass.
    exact_timeout_s : float
        Base time limit for each exact toggle optimization attempt.
    exact_retry_timeouts : Tuple[float, ...]
        Additional per-row retry limits used only when the first exact pass
        finds an incumbent but does not yet prove optimality.
    last_optimization_statuses : List[str]
        Status labels from the most recent toggle-generation run, aligned with
        the returned toggle order including the Logical-X row.

    Methods
    -------
    __init__(...)
        Build the binary symplectic system and graph structures used for
        toggle synthesis.
    generate_toggles()
        Generate one toggle for each stabilizer plus one for Logical X.
    _get_beam(source, target)
        Return nodes on a shortest-path beam between two qubits.
    _expand_beam(nodes)
        Expand a beam by one graph hop around each node.
    _get_distance_layers(start_node)
        Group graph nodes by their distance from a starting node.
    _solve_restricted(qubit_indices, target_vec)
        Solve the binary symplectic system using only selected qubits.
    _optimize_weight(vec, target_idx)
        Run the configured heuristic/exact support minimization pipeline.
    _optimize_weight_deep(vec, attempts)
        Reduce operator weight by commuting-constraint local search.
    _get_weight(vec)
        Compute the support weight of a symplectic vector.
    _symp_product(v1, v2)
        Compute the GF(2) symplectic inner product of two vectors.
    _solve_gf2(matrix, rhs)
        Solve a linear system over GF(2).
    _build_qubit_graph(specs)
        Build the interaction graph induced by shared operator support.
    _str_to_zx_arrays(op_str)
        Convert a sparse Pauli string into separate z and x indicator arrays.
    _str_to_vec_standard(op_str)
        Convert a sparse Pauli string into standard `[z | x]` vector form.
    _vec_standard_to_str(vec)
        Convert a standard symplectic vector back to sparse Pauli text.
    _get_qubits_in_op(op_str)
        Return the qubit indices referenced by a sparse Pauli string.
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
        optimization_mode: str = "exact",
        exact_timeout_s: float = 5.0,
        exact_retry_timeouts: Tuple[float, ...] = (30.0, 60.0),
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
            optimization_mode: `"exact"` to run a CP-SAT minimum-support
                solve after heuristic seeding, or `"heuristic"` to keep the
                cheaper local-search-only path.
            exact_timeout_s: Per-toggle time limit, in seconds, for the exact
                CP-SAT solve when enabled.
            exact_retry_timeouts: Additional time limits, in seconds, retried
                only when an exact pass returns an incumbent without proving
                optimality. Values at or below `exact_timeout_s` are ignored.
        """
        if optimization_mode not in {"exact", "heuristic"}:
            raise ValueError(
                "optimization_mode must be 'exact' or 'heuristic'."
            )
        if exact_timeout_s <= 0:
            raise ValueError("exact_timeout_s must be positive.")
        self.stab_specs = stabilizer_specs
        self.log_x_spec = logical_x_spec
        self.n = num_qubits
        self._rng = random.Random(random_seed)
        self.optimization_mode = optimization_mode
        self.exact_timeout_s = exact_timeout_s
        self.exact_retry_timeouts = tuple(
            timeout
            for timeout in sorted(set(exact_retry_timeouts))
            if timeout > exact_timeout_s
        )
        self.last_optimization_statuses: List[str] = []
        self._last_optimization_status = "uninitialized"

        self.constraints = stabilizer_specs + [logical_x_spec]
        self.num_constraints = len(self.constraints)
        self.constraint_vecs = [
            self._str_to_vec_standard(spec) for spec in self.constraints
        ]

        self.full_matrix = np.zeros(
            (self.num_constraints, 2 * self.n),
            dtype=np.uint8,
        )
        for row_idx, constraint_vec in enumerate(self.constraint_vecs):
            z_arr = constraint_vec[: self.n]
            x_arr = constraint_vec[self.n:]
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
        self.logical_x_vec = self._str_to_vec_standard(logical_x_spec)
        self.coset_vecs = self.stab_vecs + [self.logical_x_vec]
        self.pair_move_vecs = self._build_pair_move_vecs()

    # ─────────────────────────────────────────────────────────────────────
    # public api
    # ─────────────────────────────────────────────────────────────────────
    def generate_toggles(self) -> Tuple[List[str], str]:
        """
        Generate one anti-commuting toggle for each constraint row.

        For each target syndrome vector `e_i`, the search first tries narrow
        shortest-path beams, then a width-1 expanded beam, and finally a full
        unrestricted solve as fallback. Any valid solution is then refined by
        the configured weight optimizer, which can include an exact CP-SAT
        minimum-support pass.

        Returns:
            Tuple[List[str], str]:
            `(stabilizer_toggles, logical_x_toggle)` where the final element
            corresponds to the Logical-X constraint.
        """
        toggles: List[str] = []
        toggle_vecs: List[np.ndarray] = []
        self.last_optimization_statuses = []
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
                        best = self._optimize_weight(
                            sol,
                            target_idx=idx,
                            previous_toggle_vecs=toggle_vecs,
                        )
                        found_solution = self._vec_standard_to_str(best)
                        break

            if found_solution is None:
                all_nodes = list(range(self.n))
                sol = self._solve_restricted(all_nodes, target_vec)
                if sol is None:
                    found_solution = "I"
                else:
                    best = self._optimize_weight(
                        sol,
                        target_idx=idx,
                        previous_toggle_vecs=toggle_vecs,
                    )
                    found_solution = self._vec_standard_to_str(best)

            toggles.append(found_solution)
            toggle_vecs.append(self._str_to_vec_standard(found_solution))
            self.last_optimization_statuses.append(
                self._last_optimization_status
            )

        return toggles[:-1], toggles[-1]

    def _optimize_weight(
        self,
        vec: np.ndarray,
        target_idx: int,
        previous_toggle_vecs: List[np.ndarray],
    ) -> np.ndarray:
        """
        Minimize toggle support weight using the configured optimization mode.

        Args:
            vec: Initial valid toggle vector in standard order `[z | x]`.
            target_idx: Index of the targeted syndrome constraint.
            previous_toggle_vecs: Already-accepted toggles that the new toggle
                must commute with.

        Returns:
            np.ndarray: Lowest-weight valid toggle found.
        """
        heuristic_best = self._optimize_weight_deep(vec)
        commuting_seed = self._enforce_commutation_with_previous_toggles(
            heuristic_best,
            previous_toggle_vecs,
        )
        if self.optimization_mode != "exact" or cp_model is None:
            self._last_optimization_status = "heuristic"
            return commuting_seed
        exact_best = self._optimize_weight_exact(
            commuting_seed,
            target_idx=target_idx,
            previous_toggle_vecs=previous_toggle_vecs,
        )
        for retry_timeout in self.exact_retry_timeouts:
            if self._last_optimization_status == "exact_optimal":
                break
            if self._last_optimization_status not in {
                "exact_feasible",
                "exact_fallback",
            }:
                break
            exact_best = self._optimize_weight_exact(
                exact_best,
                target_idx=target_idx,
                previous_toggle_vecs=previous_toggle_vecs,
                timeout_s=retry_timeout,
            )
        return exact_best

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
        Reduce operator weight by greedy coset descent plus pair moves.

        Each restart shuffles the commuting-constraint order and repeatedly
        applies any single constraint that strictly lowers weight until a
        local minimum is reached. If single-step descent stalls, the optimizer
        also checks whether multiplying by a product of two commuting
        constraints lowers the weight, which helps remove support redundancy
        missed by purely greedy single-move descent.

        Args:
            vec: Initial binary symplectic vector in standard order.
            attempts: Number of randomized restarts.

        Returns:
            np.ndarray: Best vector found across all restarts.
        """
        current_best = vec.copy()
        current_best_wt = self._get_weight(vec)
        indices = list(range(len(self.coset_vecs)))

        for _ in range(attempts):
            temp_vec = current_best.copy()
            improved = True
            while improved:
                improved = False
                self._rng.shuffle(indices)
                curr_wt = self._get_weight(temp_vec)
                for idx in indices:
                    cand = temp_vec ^ self.coset_vecs[idx]
                    cand_wt = self._get_weight(cand)
                    if cand_wt < curr_wt:
                        temp_vec = cand
                        curr_wt = cand_wt
                        improved = True
                if improved:
                    continue

                pair_vec = self._best_pair_move(temp_vec, curr_wt)
                if pair_vec is not None:
                    temp_vec = pair_vec
                    improved = True
            temp_wt = self._get_weight(temp_vec)
            if temp_wt < current_best_wt:
                current_best = temp_vec
                current_best_wt = temp_wt

        return current_best

    def _best_pair_move(
        self,
        vec: np.ndarray,
        current_weight: int,
    ) -> np.ndarray | None:
        """
        Return the best improving product of two commuting constraints.

        Args:
            vec: Current toggle candidate in standard order `[z | x]`.
            current_weight: Weight of `vec`.

        Returns:
            np.ndarray | None: Improved vector, or `None` if no pair move
            lowers the weight.
        """
        if len(self.pair_move_vecs) == 0:
            return None

        candidates = self.pair_move_vecs ^ vec
        candidate_weights = np.sum(
            candidates[:, : self.n] | candidates[:, self.n :],
            axis=1,
            dtype=np.int32,
        )
        best_idx = int(np.argmin(candidate_weights))
        best_weight = int(candidate_weights[best_idx])
        if best_weight >= current_weight:
            return None
        return candidates[best_idx].copy()

    def _enforce_commutation_with_previous_toggles(
        self,
        vec: np.ndarray,
        previous_toggle_vecs: List[np.ndarray],
    ) -> np.ndarray:
        """
        Adjust a valid toggle so it commutes with previously accepted toggles.

        Multiplying by constraint `C_j` preserves the target syndrome pattern
        and toggles commutation with the already accepted dual toggle `T_j`
        while leaving commutation with earlier toggles unchanged.

        Args:
            vec: Valid toggle vector with the desired syndrome signature.
            previous_toggle_vecs: Already accepted toggles in constraint order.

        Returns:
            np.ndarray: Adjusted toggle that commutes with all previous
            toggles.
        """
        adjusted = vec.copy()
        for idx, previous_toggle in enumerate(previous_toggle_vecs):
            if self._symp_product(adjusted, previous_toggle) == 1:
                adjusted ^= self.constraint_vecs[idx]
        return adjusted

    def _optimize_weight_exact(
        self,
        vec: np.ndarray,
        target_idx: int,
        previous_toggle_vecs: List[np.ndarray],
        timeout_s: float | None = None,
    ) -> np.ndarray:
        """
        Solve for a minimum-support toggle with CP-SAT using `vec` as a hint.

        Args:
            vec: Heuristic valid toggle used as an incumbent and solver hint.
            target_idx: Index of the targeted syndrome constraint.
            previous_toggle_vecs: Already accepted toggles that the new toggle
                must commute with.
            timeout_s: Optional override for the CP-SAT time limit used for
                this solve attempt.

        Returns:
            np.ndarray: Exact minimum-support toggle if solved, otherwise the
            best feasible toggle found within the time limit, falling back to
            the heuristic input if the exact pass fails.
        """
        assert cp_model is not None

        current_weight = self._get_weight(vec)
        model = cp_model.CpModel()
        x_vars = [model.NewBoolVar(f"x_{q}") for q in range(self.n)]
        z_vars = [model.NewBoolVar(f"z_{q}") for q in range(self.n)]
        w_vars = [model.NewBoolVar(f"w_{q}") for q in range(self.n)]

        for q in range(self.n):
            model.Add(x_vars[q] <= w_vars[q])
            model.Add(z_vars[q] <= w_vars[q])
            model.Add(w_vars[q] <= x_vars[q] + z_vars[q])

        model.Add(sum(w_vars) <= current_weight)

        for row_idx, constraint in enumerate(self.constraints):
            target_bit = 1 if row_idx == target_idx else 0
            constraint_vec = self.constraint_vecs[row_idx]
            z_constraint = constraint_vec[: self.n]
            x_constraint = constraint_vec[self.n:]
            expr_terms = []
            for q in range(self.n):
                if z_constraint[q]:
                    expr_terms.append(x_vars[q])
                if x_constraint[q]:
                    expr_terms.append(z_vars[q])
            parity_aux = model.NewIntVar(
                0,
                len(expr_terms) // 2 + 1,
                f"k_{row_idx}",
            )
            model.Add(sum(expr_terms) - 2 * parity_aux == target_bit)

        for toggle_idx, toggle_vec in enumerate(previous_toggle_vecs):
            z_toggle = toggle_vec[: self.n]
            x_toggle = toggle_vec[self.n:]
            expr_terms = []
            for q in range(self.n):
                if z_toggle[q]:
                    expr_terms.append(x_vars[q])
                if x_toggle[q]:
                    expr_terms.append(z_vars[q])
            parity_aux = model.NewIntVar(
                0,
                len(expr_terms) // 2 + 1,
                f"t_{toggle_idx}",
            )
            model.Add(sum(expr_terms) - 2 * parity_aux == 0)

        z_hint = vec[: self.n]
        x_hint = vec[self.n:]
        for q in range(self.n):
            model.AddHint(x_vars[q], int(x_hint[q]))
            model.AddHint(z_vars[q], int(z_hint[q]))
            model.AddHint(w_vars[q], int(z_hint[q] | x_hint[q]))

        support_scale = 2 * self.n * self.n + self.n + 1
        tie_break = sum(
            (q + 1) * x_vars[q] + (self.n + q + 1) * z_vars[q]
            for q in range(self.n)
        )
        model.Minimize(support_scale * sum(w_vars) + tie_break)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = (
            self.exact_timeout_s if timeout_s is None else timeout_s
        )
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            self._last_optimization_status = "exact_fallback"
            return vec

        z_out = np.array(
            [int(solver.Value(z_vars[q])) for q in range(self.n)],
            dtype=np.uint8,
        )
        x_out = np.array(
            [int(solver.Value(x_vars[q])) for q in range(self.n)],
            dtype=np.uint8,
        )
        exact_vec = np.concatenate([z_out, x_out])
        if self._get_weight(exact_vec) > current_weight:
            self._last_optimization_status = "exact_fallback"
            return vec
        self._last_optimization_status = (
            "exact_optimal"
            if status == cp_model.OPTIMAL
            else "exact_feasible"
        )
        return exact_vec

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

    def _build_pair_move_vecs(self) -> np.ndarray:
        """
        Precompute all pairwise products of commuting constraint vectors.

        Returns:
            np.ndarray: Array of shape `(num_pairs, 2n)` containing pairwise
            XOR combinations of the commuting constraint basis.
        """
        num_vecs = len(self.coset_vecs)
        if num_vecs < 2:
            return np.zeros((0, 2 * self.n), dtype=np.uint8)

        pair_count = num_vecs * (num_vecs - 1) // 2
        pair_moves = np.zeros((pair_count, 2 * self.n), dtype=np.uint8)
        cursor = 0
        for i in range(num_vecs - 1):
            left = self.coset_vecs[i]
            for j in range(i + 1, num_vecs):
                pair_moves[cursor] = left ^ self.coset_vecs[j]
                cursor += 1
        return pair_moves

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
