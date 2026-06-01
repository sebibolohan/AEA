#!/usr/bin/env python3
"""
Advanced PartialCol Reactive Tabu Search for fixed-k Graph Coloring.

Main improvements over a naive PartialCol implementation:
  - partial legal colorings: k independent color classes + U uncolored vertices
  - O(1) move evaluation using adj_color_count[v][c]
  - incremental updates after every recoloring/uncoloring
  - best-move sampling from U with tabu + aspiration
  - reactive tabu tenure and diversification shakes
  - multi-restart inside a time limit
  - CSV logging compatible with previous scripts

Usage examples:
  python partialcol_advanced.py data/raw/flat300_28_0.col --k 28 --time-limit 300 --runs 5 --csv results/partialcol_advanced.csv
  python partialcol_advanced.py data/raw/DSJC500.9.col --k 126 --time-limit 600 --runs 5 --csv results/partialcol_advanced.csv
"""

import argparse
import csv
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Graph:
    n: int
    edges: List[Tuple[int, int]]
    adj: List[List[int]]
    degree: List[int]

    @classmethod
    def from_dimacs_col(cls, path: str) -> "Graph":
        n = 0
        edges: List[Tuple[int, int]] = []

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("c"):
                    continue
                parts = line.split()
                tag = parts[0].lower()

                if tag == "p":
                    if len(parts) < 4 or parts[1].lower() != "edge":
                        raise ValueError("Invalid DIMACS problem line: " + line)
                    n = int(parts[2])
                elif tag == "e":
                    if len(parts) != 3:
                        raise ValueError("Invalid DIMACS edge line: " + line)
                    u = int(parts[1]) - 1
                    v = int(parts[2]) - 1
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    edges.append((u, v))

        if n <= 0:
            raise ValueError("Missing or invalid DIMACS line: p edge n m")

        edges = sorted(set(edges))
        adj = [[] for _ in range(n)]
        for u, v in edges:
            if u < 0 or v < 0 or u >= n or v >= n:
                raise ValueError("Edge endpoint out of range")
            adj[u].append(v)
            adj[v].append(u)

        degree = [len(a) for a in adj]
        return cls(n=n, edges=edges, adj=adj, degree=degree)


@dataclass
class RunResult:
    k: int
    feasible: bool
    uncolored: int
    colors: List[int]
    iterations: int
    restarts: int
    time_seconds: float
    seed: int


def verify_coloring(graph: Graph, colors: List[int], k: int) -> int:
    if len(colors) != graph.n:
        raise ValueError("Coloring length mismatch")
    conflicts = 0
    for c in colors:
        if c < 0 or c >= k:
            raise ValueError("Color out of range")
    for u, v in graph.edges:
        if colors[u] == colors[v]:
            conflicts += 1
    return conflicts


def verify_partial_coloring(graph: Graph, colors: List[int], k: int) -> Tuple[int, int]:
    uncolored = sum(1 for c in colors if c == -1)
    conflicts = 0
    for c in colors:
        if c != -1 and (c < 0 or c >= k):
            raise ValueError("Color out of range")
    for u, v in graph.edges:
        if colors[u] != -1 and colors[u] == colors[v]:
            conflicts += 1
    return uncolored, conflicts


def dsatur_coloring(graph: Graph) -> Tuple[int, List[int]]:
    n = graph.n
    colors = [-1] * n
    neighbor_colors = [set() for _ in range(n)]
    uncolored = set(range(n))

    first = max(range(n), key=lambda v: graph.degree[v])
    colors[first] = 0
    uncolored.remove(first)
    for nb in graph.adj[first]:
        neighbor_colors[nb].add(0)

    used = 1
    while uncolored:
        v = max(uncolored, key=lambda x: (len(neighbor_colors[x]), graph.degree[x], x))
        c = 0
        while c in neighbor_colors[v]:
            c += 1
        colors[v] = c
        used = max(used, c + 1)
        uncolored.remove(v)
        for nb in graph.adj[v]:
            if colors[nb] == -1:
                neighbor_colors[nb].add(c)
    return used, colors


def randomized_dsatur_order(graph: Graph, rng: random.Random) -> List[int]:
    # Produces an ordering similar to DSATUR, with random tie-breaking.
    n = graph.n
    colored = [False] * n
    sat_sets = [set() for _ in range(n)]
    order: List[int] = []

    for _ in range(n):
        candidates = [v for v in range(n) if not colored[v]]
        max_sat = max(len(sat_sets[v]) for v in candidates)
        candidates = [v for v in candidates if len(sat_sets[v]) == max_sat]
        max_deg = max(graph.degree[v] for v in candidates)
        candidates = [v for v in candidates if graph.degree[v] == max_deg]
        v = rng.choice(candidates)
        colored[v] = True
        order.append(v)
        pseudo_color = len(order)  # only for saturation diversity in ordering construction
        for nb in graph.adj[v]:
            if not colored[nb]:
                sat_sets[nb].add(pseudo_color)
    return order


class AdvancedPartialCol:
    def __init__(
        self,
        graph: Graph,
        k: int,
        rng: random.Random,
        candidate_sample: int = 96,
        base_tenure: int = 12,
        stagnation_limit: int = 6000,
        restart_limit: int = 18000,
    ) -> None:
        self.g = graph
        self.k = k
        self.rng = rng
        self.candidate_sample = max(8, candidate_sample)
        self.base_tenure = max(1, base_tenure)
        self.stagnation_limit = max(200, stagnation_limit)
        self.restart_limit = max(self.stagnation_limit + 1, restart_limit)

    def solve(self, time_limit: float, seed_colors: Optional[List[int]] = None) -> RunResult:
        start = time.perf_counter()
        deadline = start + time_limit

        global_best_colors: List[int] = []
        global_best_uncolored = math.inf
        total_iterations = 0
        restarts = 0

        # First attempt uses the deterministic seed if available; later restarts use diversified construction.
        attempt = 0
        while time.perf_counter() < deadline:
            current_seed = seed_colors if attempt == 0 else None
            colors, U, adj_color_count = self._construct_initial(current_seed)
            best_local_colors = colors[:]
            best_local_uncolored = len(U)

            if best_local_uncolored < global_best_uncolored:
                global_best_uncolored = best_local_uncolored
                global_best_colors = best_local_colors[:]
                if global_best_uncolored == 0:
                    break

            tabu_until = [[0] * self.k for _ in range(self.g.n)]
            tenure = self.base_tenure + self.rng.randrange(0, self.base_tenure + 1)
            last_improvement_iter = 0
            local_iterations = 0

            while time.perf_counter() < deadline and U:
                local_iterations += 1
                total_iterations += 1

                if local_iterations - last_improvement_iter > self.restart_limit:
                    break

                if local_iterations - last_improvement_iter > self.stagnation_limit:
                    tenure = min(150, int(tenure * 1.25) + 1)
                    self._shake(colors, U, adj_color_count, strength=self._shake_strength(len(U)))
                    last_improvement_iter = local_iterations
                elif local_iterations % 2500 == 0 and tenure > self.base_tenure:
                    tenure = max(self.base_tenure, int(tenure * 0.95))

                move = self._select_move(colors, U, adj_color_count, tabu_until, local_iterations, global_best_uncolored)
                if move is None:
                    self._shake(colors, U, adj_color_count, strength=0.02)
                    continue

                v, c, removed_vertices = move
                self._apply_move(colors, U, adj_color_count, v, c, removed_vertices, tabu_until, local_iterations, tenure)

                u_size = len(U)
                if u_size < best_local_uncolored:
                    best_local_uncolored = u_size
                    best_local_colors = colors[:]
                    last_improvement_iter = local_iterations
                    tenure = max(self.base_tenure, int(tenure * 0.88))

                    if u_size < global_best_uncolored:
                        global_best_uncolored = u_size
                        global_best_colors = best_local_colors[:]
                        if global_best_uncolored == 0:
                            break

            if global_best_uncolored == 0:
                break

            restarts += 1
            attempt += 1

        elapsed = time.perf_counter() - start
        if not global_best_colors:
            global_best_colors = [-1] * self.g.n
            global_best_uncolored = self.g.n

        feasible = global_best_uncolored == 0
        if feasible:
            conflicts = verify_coloring(self.g, global_best_colors, self.k)
            if conflicts != 0:
                raise RuntimeError(f"Internal error: feasible solution has {conflicts} conflicts")
        else:
            _, conflicts = verify_partial_coloring(self.g, global_best_colors, self.k)
            if conflicts != 0:
                raise RuntimeError(f"Internal error: partial coloring has {conflicts} conflicts")

        return RunResult(
            k=self.k,
            feasible=feasible,
            uncolored=int(global_best_uncolored),
            colors=global_best_colors,
            iterations=total_iterations,
            restarts=restarts,
            time_seconds=elapsed,
            seed=-1,
        )

    def _construct_initial(self, seed_colors: Optional[List[int]]) -> Tuple[List[int], Set[int], List[List[int]]]:
        n = self.g.n
        colors = [-1] * n
        U: Set[int] = set()
        adj_color_count = [[0] * self.k for _ in range(n)]

        if seed_colors is not None:
            # Seed-aware construction: try to preserve DSATUR color classes after reducing labels modulo k.
            order = sorted(range(n), key=lambda v: (seed_colors[v] % self.k, -self.g.degree[v], v))
        else:
            mode = self.rng.random()
            if mode < 0.55:
                order = randomized_dsatur_order(self.g, self.rng)
            else:
                order = list(range(n))
                order.sort(key=lambda v: (self.g.degree[v] + 0.25 * self.rng.random()), reverse=True)

        for v in order:
            if seed_colors is not None:
                preferred = seed_colors[v] % self.k
                color_order = [preferred] + [c for c in range(self.k) if c != preferred]
            else:
                color_order = list(range(self.k))
                self.rng.shuffle(color_order)
                # Prefer colors that are currently feasible and small classes implicitly through conflict count.
                color_order.sort(key=lambda c: (adj_color_count[v][c], self.rng.random()))

            placed = False
            for c in color_order:
                if adj_color_count[v][c] == 0:
                    self._color_vertex(colors, adj_color_count, v, c)
                    placed = True
                    break
            if not placed:
                U.add(v)

        return colors, U, adj_color_count

    def _color_vertex(self, colors: List[int], adj_color_count: List[List[int]], v: int, c: int) -> None:
        colors[v] = c
        for nb in self.g.adj[v]:
            adj_color_count[nb][c] += 1

    def _uncolor_vertex(self, colors: List[int], adj_color_count: List[List[int]], v: int) -> int:
        old = colors[v]
        if old == -1:
            return -1
        colors[v] = -1
        for nb in self.g.adj[v]:
            adj_color_count[nb][old] -= 1
        return old

    def _select_move(
        self,
        colors: List[int],
        U: Set[int],
        adj_color_count: List[List[int]],
        tabu_until: List[List[int]],
        iteration: int,
        global_best_uncolored: int,
    ) -> Optional[Tuple[int, int, List[int]]]:
        if not U:
            return None

        # Prefer hard uncolored vertices with high degree / many constraints.
        if len(U) <= self.candidate_sample:
            candidates = list(U)
        else:
            sampled = self.rng.sample(tuple(U), self.candidate_sample)
            sampled.sort(key=lambda v: self.g.degree[v], reverse=True)
            candidates = sampled[: self.candidate_sample]

        best_v = -1
        best_c = -1
        best_score = math.inf
        best_removed_count = math.inf
        best_removed: List[int] = []

        current_u = len(U)
        for v in candidates:
            # Try promising colors first: low removal count.
            color_order = list(range(self.k))
            color_order.sort(key=lambda c: (adj_color_count[v][c], self.rng.random()))

            # Checking all colors is okay because adj_color_count is O(1).
            for c in color_order:
                removed_count = adj_color_count[v][c]
                new_u = current_u - 1 + removed_count

                is_tabu = tabu_until[v][c] > iteration
                aspiration = new_u < global_best_uncolored
                if is_tabu and not aspiration:
                    continue

                # Main objective: minimize |U|. Tie-breakers: remove fewer high-degree vertices.
                # A slight random term prevents deterministic cycling.
                score = new_u
                if score < best_score or (score == best_score and removed_count < best_removed_count):
                    removed = [nb for nb in self.g.adj[v] if colors[nb] == c]
                    best_v = v
                    best_c = c
                    best_score = score
                    best_removed_count = removed_count
                    best_removed = removed
                elif score == best_score and removed_count == best_removed_count and self.rng.random() < 0.03:
                    removed = [nb for nb in self.g.adj[v] if colors[nb] == c]
                    best_v = v
                    best_c = c
                    best_removed = removed

                if removed_count == 0:
                    # This is an improving move; no need to inspect weaker colors for this vertex.
                    break

        if best_v == -1:
            # All moves tabu; pick a random admissible-like perturbative move.
            v = self.rng.choice(tuple(U))
            c = min(range(self.k), key=lambda col: adj_color_count[v][col])
            removed = [nb for nb in self.g.adj[v] if colors[nb] == c]
            return v, c, removed

        return best_v, best_c, best_removed

    def _apply_move(
        self,
        colors: List[int],
        U: Set[int],
        adj_color_count: List[List[int]],
        v: int,
        c: int,
        removed_vertices: List[int],
        tabu_until: List[List[int]],
        iteration: int,
        tenure: int,
    ) -> None:
        # v is uncolored, put it into color c.
        if v in U:
            U.remove(v)
        self._color_vertex(colors, adj_color_count, v, c)

        # Remove conflicting vertices from the same color class.
        for u in removed_vertices:
            if colors[u] == c:
                self._uncolor_vertex(colors, adj_color_count, u)
                U.add(u)
                # Prevent immediate reinsertion of u into c.
                tabu_until[u][c] = iteration + tenure + self.rng.randrange(0, max(2, tenure // 2 + 1))

        # Prevent undo pressure around v as well.
        tabu_until[v][c] = iteration + max(1, tenure // 3)

    def _shake_strength(self, u_size: int) -> float:
        # Stronger shakes when far from feasibility; lighter when near target.
        ratio = u_size / max(1, self.g.n)
        if ratio > 0.15:
            return 0.06
        if ratio > 0.05:
            return 0.04
        return 0.02

    def _shake(self, colors: List[int], U: Set[int], adj_color_count: List[List[int]], strength: float) -> None:
        colored = [v for v, c in enumerate(colors) if c != -1]
        if not colored:
            return

        count = max(1, int(self.g.n * strength))
        # Shake vertices that constrain many uncolored vertices: high degree first, with randomness.
        colored.sort(key=lambda v: (self.g.degree[v], self.rng.random()), reverse=True)
        pool = colored[: min(len(colored), max(20, count * 8))]
        self.rng.shuffle(pool)

        for v in pool[:count]:
            if colors[v] != -1:
                self._uncolor_vertex(colors, adj_color_count, v)
                U.add(v)


def solve_single_k(graph: Graph, k: int, time_limit: float, seed: int) -> RunResult:
    rng = random.Random(seed)
    ub, dsatur_colors = dsatur_coloring(graph)
    seed_colors = dsatur_colors if ub >= k else None

    # Larger candidate samples help dense graphs; O(1) scoring keeps this affordable.
    solver = AdvancedPartialCol(
        graph=graph,
        k=k,
        rng=rng,
        candidate_sample=max(64, min(320, graph.n)),
        base_tenure=max(10, min(80, graph.n // 12 + k // 10)),
        stagnation_limit=max(3000, min(25000, graph.n * 18)),
        restart_limit=max(8000, min(70000, graph.n * 55)),
    )
    result = solver.solve(time_limit=time_limit, seed_colors=seed_colors)
    result.seed = seed
    return result


def write_solution(path: str, result: RunResult) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("c Advanced PartialCol solution\n")
        f.write(f"c k = {result.k}\n")
        f.write(f"c feasible = {result.feasible}\n")
        f.write(f"c uncolored = {result.uncolored}\n")
        for i, c in enumerate(result.colors, start=1):
            f.write(f"v {i} {0 if c == -1 else c + 1}\n")


def append_csv(path: str, row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "instance", "method", "seed", "k", "feasible", "uncolored",
        "iterations", "restarts", "time_seconds"
    ]
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_self_test() -> None:
    cycle_5 = Graph(
        n=5,
        edges=[(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)],
        adj=[[1, 4], [0, 2], [1, 3], [2, 4], [3, 0]],
        degree=[2, 2, 2, 2, 2],
    )
    r1 = solve_single_k(cycle_5, 3, 1.0, 1)
    assert r1.feasible
    assert verify_coloring(cycle_5, r1.colors, 3) == 0

    bip = Graph(
        n=4,
        edges=[(0, 2), (0, 3), (1, 2), (1, 3)],
        adj=[[2, 3], [2, 3], [0, 1], [0, 1]],
        degree=[2, 2, 2, 2],
    )
    r2 = solve_single_k(bip, 2, 1.0, 2)
    assert r2.feasible
    assert verify_coloring(bip, r2.colors, 2) == 0
    print("Self-test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced PartialCol Reactive Tabu Search")
    parser.add_argument("path", nargs="?", help="Path to DIMACS .col file")
    parser.add_argument("--k", type=int, default=None, help="Target number of colors")
    parser.add_argument("--time-limit", type=float, default=60.0, help="Time limit per run")
    parser.add_argument("--seed", type=int, default=0, help="Base seed")
    parser.add_argument("--runs", type=int, default=1, help="Independent runs")
    parser.add_argument("--csv", type=str, default="", help="Append run results to CSV")
    parser.add_argument("--output", type=str, default="", help="Write best solution")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not args.path:
        parser.error("You must provide a DIMACS .col file path")

    graph = Graph.from_dimacs_col(args.path)
    ub, _ = dsatur_coloring(graph)
    k = args.k if args.k is not None else ub

    print(f"Graph: n={graph.n}, m={len(graph.edges)}")
    print(f"DSATUR upper bound: {ub}")
    print(f"Target k: {k}")

    best: Optional[RunResult] = None
    for run in range(args.runs):
        seed = args.seed + run * 10007
        print(f"\nRun {run + 1}/{args.runs}, seed={seed}")
        result = solve_single_k(graph, k, args.time_limit, seed)
        print(
            f"k={result.k}, feasible={result.feasible}, uncolored={result.uncolored}, "
            f"iters={result.iterations}, restarts={result.restarts}, time={result.time_seconds:.3f}s"
        )

        if best is None or result.uncolored < best.uncolored or (result.feasible and not best.feasible):
            best = result

        if args.csv:
            append_csv(args.csv, {
                "instance": os.path.basename(args.path),
                "method": "AdvancedPartialColReactiveTabu",
                "seed": seed,
                "k": result.k,
                "feasible": int(result.feasible),
                "uncolored": result.uncolored,
                "iterations": result.iterations,
                "restarts": result.restarts,
                "time_seconds": f"{result.time_seconds:.6f}",
            })

    assert best is not None
    print("\nBest result")
    print(f"k = {best.k}")
    print(f"feasible = {best.feasible}")
    print(f"uncolored = {best.uncolored}")
    if best.feasible:
        print(f"verification conflicts = {verify_coloring(graph, best.colors, best.k)}")

    if args.output:
        write_solution(args.output, best)
        print(f"Solution written to: {args.output}")


if __name__ == "__main__":
    main()
