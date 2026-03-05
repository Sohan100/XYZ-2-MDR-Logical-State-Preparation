import numpy as np
import networkx as nx
from typing import List, Tuple
import random
from collections import deque

class RobustToggleGenerator:
    """
    Finds minimal-weight toggles using 'Fat' Radial Beam Search.
    
    Improvement:
    - If the strict shortest-path beam fails, it 'fattens' the beam 
      by including neighbors (Width=1). 
    - This fixes validity errors where the operator needed to zig-zag 
      slightly off the perfect geodesic.
    """

    def __init__(self, stabilizer_specs: List[str], logical_x_spec: str, num_qubits: int):
        """
        Initialize constraints, matrices, connectivity, and caches.

        Args:
            stabilizer_specs: Stabilizer operator strings.
            logical_x_spec: Logical-X operator string.
            num_qubits: Number of physical qubits.
        """
        self.stab_specs = stabilizer_specs
        self.log_x_spec = logical_x_spec
        self.n = num_qubits
        
        # 1. Constraints
        self.constraints = stabilizer_specs + [logical_x_spec]
        self.num_constraints = len(self.constraints)
        
        # 2. Pre-compute Full Matrix
        self.full_matrix = np.zeros((self.num_constraints, 2 * self.n), dtype=np.uint8)
        for r, spec in enumerate(self.constraints):
            z_C, x_C = self._str_to_zx_arrays(spec)
            for i in range(self.n):
                self.full_matrix[r, 2*i]     = x_C[i] 
                self.full_matrix[r, 2*i + 1] = z_C[i] 
                
        # 3. Connectivity
        self.qubit_graph = self._build_qubit_graph(stabilizer_specs)
        # Pre-compute shortest paths (Fast for N < 1000)
        self.all_pairs_dist = dict(nx.all_pairs_shortest_path_length(self.qubit_graph))
        
        # 4. Precompute vectors for optimizer
        self.stab_vecs = [self._str_to_vec_standard(s) for s in stabilizer_specs]

    def generate_toggles(self) -> Tuple[List[str], str]:
        """
        Generate stabilizer toggles and the Logical-X toggle.

        The search uses radial beam expansion with a global fallback.

        Returns:
            Tuple[List[str], str]: Stabilizer toggles and Logical-X toggle.
        """
        toggles = []
        
        for i in range(self.num_constraints):
            target_vec = np.zeros(self.num_constraints, dtype=np.uint8)
            target_vec[i] = 1
            
            # Identify search center
            op_str = self.constraints[i]
            qs_in_op = self._get_qubits_in_op(op_str)
            center_node = qs_in_op[0] if qs_in_op else 0
            
            # Organize lattice by distance from this center
            layers = self._get_distance_layers(center_node)
            
            found_solution = None
            
            # --- RADIAL BEAM SEARCH ---
            max_search_dist = len(layers)
            
            for r in range(max_search_dist):
                if found_solution: break
                
                targets = layers[r]
                # Shuffle targets to avoid directional bias artifacts
                random.shuffle(targets)
                
                for target_node in targets:
                    # 1. Thin Beam (Strict Shortest Path)
                    beam_nodes = self._get_beam(center_node, target_node)
                    sol = self._solve_restricted(beam_nodes, target_vec)
                    
                    # 2. Fat Beam (If Thin fails)
                    if sol is None:
                        fat_beam_nodes = self._expand_beam(beam_nodes)
                        sol = self._solve_restricted(fat_beam_nodes, target_vec)
                    
                    if sol is not None:
                        # Found valid path -> Optimize and Save
                        optimized = self._optimize_weight_deep(sol)
                        found_solution = self._vec_standard_to_str(optimized)
                        break
            
            # Fallback: If Beam Search failed entirely (rare, usually only for Logical),
            # run a global solve to guarantee validity.
            if found_solution is None:
                # Global solve on all qubits
                all_nodes = list(range(self.n))
                sol = self._solve_restricted(all_nodes, target_vec)
                if sol is not None:
                     optimized = self._optimize_weight_deep(sol)
                     found_solution = self._vec_standard_to_str(optimized)
                else:
                    found_solution = "I"

            toggles.append(found_solution)

        return toggles[:-1], toggles[-1]

    # ------------------ Beam Logic ------------------

    def _get_beam(self, source, target):
        """
        Return the union of shortest-path nodes between source and target.

        Args:
            source: Start node index.
            target: End node index.

        Returns:
            List[int]: Beam nodes on geodesic paths.
        """
        try:
            d_st = self.all_pairs_dist[source][target]
        except KeyError:
            return [source, target] # Should not happen if connected

        beam = []
        # Filter nodes that are strictly on a geodesic
        for n in range(self.n):
            try:
                ds = self.all_pairs_dist[source][n]
                dt = self.all_pairs_dist[n][target]
                if ds + dt == d_st:
                    beam.append(n)
            except KeyError:
                pass
        return beam
    
    def _expand_beam(self, nodes):
        """
        Add immediate neighbors to the current beam node set.

        Args:
            nodes: Base beam nodes.

        Returns:
            List[int]: Expanded beam nodes.
        """
        expanded = set(nodes)
        for n in nodes:
            for neighbor in self.qubit_graph.neighbors(n):
                expanded.add(neighbor)
        return list(expanded)

    def _get_distance_layers(self, start_node):
        """
        Organize nodes into BFS distance layers from the start node.

        Args:
            start_node: BFS root node.

        Returns:
            List[List[int]]: Nodes grouped by graph distance.
        """
        layers = []
        seen = {start_node}
        queue = deque([(start_node, 0)])
        
        current_dist = 0
        current_layer = []
        
        while queue:
            node, dist = queue.popleft()
            if dist > current_dist:
                layers.append(current_layer)
                current_layer = []
                current_dist = dist
            current_layer.append(node)
            for n in self.qubit_graph.neighbors(node):
                if n not in seen:
                    seen.add(n)
                    queue.append((n, dist + 1))
        
        if current_layer: layers.append(current_layer)
        return layers

    def _solve_restricted(self, qubit_indices, target_vec):
        """
        Solve a restricted linear system using only selected qubit columns.

        Args:
            qubit_indices: Enabled qubit indices.
            target_vec: Desired syndrome vector.

        Returns:
            np.ndarray | None: Candidate solution vector or None.
        """
        col_indices = []
        for q in qubit_indices:
            col_indices.extend([2*q, 2*q+1]) 
        col_indices = np.array(sorted(col_indices), dtype=int)
        
        sub_matrix = self.full_matrix[:, col_indices]
        x_sub = self._solve_gf2(sub_matrix, target_vec)
        
        if x_sub is not None:
            full_sol = np.zeros(2 * self.n, dtype=np.uint8)
            full_sol[col_indices] = x_sub
            z_part = full_sol[0::2]
            x_part = full_sol[1::2]
            return np.concatenate([z_part, x_part])
        return None

    # ------------------ Optimization ------------------

    def _optimize_weight_deep(self, vec, attempts=10):
        """
        Greedily reduce operator weight with randomized restarts.

        Args:
            vec: Initial candidate vector.
            attempts: Number of random-restart passes.

        Returns:
            np.ndarray: Best low-weight vector found.
        """
        current_best = vec.copy()
        current_best_wt = self._get_weight(vec)
        indices = list(range(len(self.stab_vecs)))

        for _ in range(attempts):
            temp_vec = current_best.copy()
            random.shuffle(indices)
            
            improved = True
            while improved:
                improved = False
                curr_wt = self._get_weight(temp_vec)
                for idx in indices:
                    stab = self.stab_vecs[idx]
                    cand = temp_vec ^ stab
                    cand_wt = self._get_weight(cand)
                    if cand_wt < curr_wt:
                        temp_vec = cand
                        curr_wt = cand_wt
                        improved = True
            
            if self._get_weight(temp_vec) < current_best_wt:
                current_best = temp_vec
                current_best_wt = self._get_weight(temp_vec)
                
        return current_best

    def _get_weight(self, vec):
        """
        Return the number of non-identity qubits in a symplectic vector.

        Args:
            vec: Candidate [z|x] vector.

        Returns:
            int: Operator weight.
        """
        return np.sum(vec[:self.n] | vec[self.n:])
    
    def _symp_product(self, v1, v2):
        """
        Compute the GF(2) symplectic product of two vectors.

        Args:
            v1: First [z|x] vector.
            v2: Second [z|x] vector.

        Returns:
            int: 0 for commute, 1 for anti-commute.
        """
        n = self.n
        return (np.sum(v1[:n] & v2[n:]) + np.sum(v1[n:] & v2[:n])) % 2

    # ------------------ Helpers ------------------

    def _build_qubit_graph(self, specs):
        """
        Build a graph where qubits co-occurring in checks share an edge.

        Args:
            specs: Stabilizer operator strings.

        Returns:
            nx.Graph: Qubit connectivity graph.
        """
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        for op in specs:
            qs = self._get_qubits_in_op(op)
            for i in range(len(qs)):
                for j in range(i+1, len(qs)):
                    G.add_edge(qs[i], qs[j])
        return G

    def _solve_gf2(self, A, b):
        """
        Solve Ax=b over GF(2) via Gaussian elimination.

        Args:
            A: Binary coefficient matrix.
            b: Binary right-hand-side vector.

        Returns:
            np.ndarray | None: Solution vector or None if inconsistent.
        """
        h, w = A.shape
        M = np.hstack([A, b.reshape(-1, 1)])
        pivots = []
        pivot_row = 0
        for col in range(w):
            if pivot_row >= h: break
            cand = np.where(M[pivot_row:, col] == 1)[0]
            if len(cand) == 0: continue
            curr = cand[0] + pivot_row
            if curr != pivot_row: M[[pivot_row, curr]] = M[[curr, pivot_row]]
            pivot_vec = M[pivot_row]
            rows_to_xor = np.where(M[pivot_row+1:, col] == 1)[0] + pivot_row + 1
            if len(rows_to_xor) > 0: M[rows_to_xor] ^= pivot_vec
            pivots.append((col, pivot_row))
            pivot_row += 1
        if np.any(M[pivot_row:, -1]): return None
        x = np.zeros(w, dtype=np.uint8)
        for col, row in reversed(pivots):
            val = M[row, -1]
            dot = np.dot(M[row, col+1:w], x[col+1:]) % 2
            val ^= dot
            x[col] = val
        return x

    def _str_to_zx_arrays(self, op_str):
        """
        Convert a sparse Pauli string into separate z/x bit arrays.

        Args:
            op_str: Sparse Pauli-string operator.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Z bits and X bits.
        """
        z = np.zeros(self.n, dtype=np.uint8); x = np.zeros_like(z)
        for term in op_str.split():
            if not term or term == 'I': continue
            p, q = term[0], int(term[1:])
            if p in 'ZY': z[q] = 1
            if p in 'XY': x[q] = 1
        return z, x
    
    def _str_to_vec_standard(self, op_str):
        """
        Convert a sparse Pauli string into standard [z|x] vector form.

        Args:
            op_str: Sparse Pauli-string operator.

        Returns:
            np.ndarray: Concatenated [z|x] vector.
        """
        z, x = self._str_to_zx_arrays(op_str)
        return np.concatenate([z, x])

    def _vec_standard_to_str(self, vec):
        """
        Convert standard [z|x] vector form into sparse Pauli text.

        Args:
            vec: Binary symplectic vector.

        Returns:
            str: Sparse Pauli string representation.
        """
        z, x = vec[:self.n], vec[self.n:]
        terms = []
        for i in range(self.n):
            if x[i] and z[i]: terms.append(f"Y{i}")
            elif x[i]:        terms.append(f"X{i}")
            elif z[i]:        terms.append(f"Z{i}")
        return " ".join(terms) if terms else "I"

    def _get_qubits_in_op(self, op_str):
        """
        Extract qubit indices appearing in a sparse Pauli string.

        Args:
            op_str: Sparse Pauli-string operator.

        Returns:
            List[int]: Qubit indices referenced by the operator.
        """
        q = []
        for t in op_str.split():
            if t and t != 'I': q.append(int(t[1:]))
        return q
