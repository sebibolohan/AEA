#!/usr/bin/env python3
"""
Adaptive Hybrid Graph Coloring (AHGC)
Breakout Tabu Search for fixed-k Graph Coloring on DIMACS .col files.

Core idea:
    - Complete coloring representation: every vertex always has one color in {0,...,k-1}.
    - Objective: minimize number of conflicting edges.
    - Local search: conflict-driven tabu search with incremental adj_color_count[v][c].
    - Breakout mechanism: when search stagnates, increase penalties on currently conflicting edges.
      Move scoring uses weighted conflict delta, so the search is pushed away from repeated local minima.
    - Reactive tabu tenure: increases during stagnation, decreases after progress.
    - Elite memory + strategic restarts: keeps best solutions and restarts from perturbed elites.

This is designed as a strong third method distinct from GA and ACO:
    GA  -> population/crossover
    ACO -> pheromone construction
    AHGC/BLS -> trajectory-based adaptive local search with breakout penalties

Examples:
    python ahgc_breakout.py data/raw/queen7_7.col --k 7 --time-limit 30 --runs 5 --csv results/ahgc.csv
    python ahgc_breakout.py data/raw/DSJC500.9.col --k 126 --time-limit 600 --runs 5 --csv results/ahgc.csv
    python ahgc_breakout.py data/raw/DSJC500.9.col --k-ref 126 --try-extra 2 --time-limit 600 --runs 5 --csv results/ahgc.csv
"""


import argparse
import csv
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Graph:
    n: int
    edges: List[Tuple[int, int]]
    adj: List[List[int]]
    degree: List[int]
    edge_index: Dict[Tuple[int, int], int]

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

        degree = [len(x) for x in adj]
        edge_index = {}
        for i, (u, v) in enumerate(edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i

        return cls(n=n, edges=edges, adj=adj, degree=degree, edge_index=edge_index)


@dataclass
class RunResult:
    instance: str
    method: str
    seed: int
    k: int
    feasible: bool
    conflicts: int
    weighted_conflicts: int
    iterations: int
    restarts: int
    breakouts: int
    time_seconds: float
    colors: List[int]


def verify_coloring(graph: Graph, colors: List[int], k: int) -> int:
    if len(colors) != graph.n:
        raise ValueError("Coloring length mismatch")
    for c in colors:
        if c < 0 or c >= k:
            raise ValueError("Color out of range")
    return sum(1 for u, v in graph.edges if colors[u] == colors[v])


def dsatur_coloring(graph: Graph, rng: Optional[random.Random] = None) -> Tuple[int, List[int]]:
    n = graph.n
    colors = [-1] * n
    neighbor_colors = [set() for _ in range(n)]
    uncolored = set(range(n))

    if rng is None:
        first = max(range(n), key=lambda v: graph.degree[v])
    else:
        max_deg = max(graph.degree)
        candidates = [v for v in range(n) if graph.degree[v] == max_deg]
        first = rng.choice(candidates)

    colors[first] = 0
    uncolored.remove(first)
    for nb in graph.adj[first]:
        neighbor_colors[nb].add(0)

    used = 1
    while uncolored:
        if rng is None:
            v = max(uncolored, key=lambda x: (len(neighbor_colors[x]), graph.degree[x], x))
        else:
            # Randomized tie-breaking among near-best vertices.
            best_sat = max(len(neighbor_colors[x]) for x in uncolored)
            cand = [x for x in uncolored if len(neighbor_colors[x]) == best_sat]
            best_deg = max(graph.degree[x] for x in cand)
            cand = [x for x in cand if graph.degree[x] == best_deg]
            v = rng.choice(cand)

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


def reduce_to_k_greedy(graph: Graph, seed_colors: List[int], k: int, rng: random.Random) -> List[int]:
    """Map/rebuild a coloring to exactly k colors using DSATUR order and min-conflict assignment."""
    n = graph.n
    colors = [-1] * n
    order = sorted(range(n), key=lambda v: (-graph.degree[v], seed_colors[v], v))

    # First try preferred color modulo k, then colors with minimum local conflicts.
    for v in order:
        preferred = seed_colors[v] % k
        counts = [0] * k
        for nb in graph.adj[v]:
            c = colors[nb]
            if c != -1:
                counts[c] += 1

        best_val = min(counts)
        best_cols = [c for c in range(k) if counts[c] == best_val]
        if preferred in best_cols and rng.random() < 0.70:
            chosen = preferred
        else:
            chosen = rng.choice(best_cols)
        colors[v] = chosen

    return colors


def randomized_greedy_coloring(graph: Graph, k: int, rng: random.Random) -> List[int]:
    n = graph.n
    colors = [-1] * n
    uncolored = set(range(n))
    neighbor_colors = [set() for _ in range(n)]

    while uncolored:
        best_sat = max(len(neighbor_colors[v]) for v in uncolored)
        cand = [v for v in uncolored if len(neighbor_colors[v]) == best_sat]
        cand.sort(key=lambda v: graph.degree[v], reverse=True)
        top = cand[: min(len(cand), 12)]
        v = rng.choice(top)

        counts = [0] * k
        for nb in graph.adj[v]:
            c = colors[nb]
            if c != -1:
                counts[c] += 1
        min_conf = min(counts)
        best_cols = [c for c in range(k) if counts[c] == min_conf]
        c = rng.choice(best_cols)
        colors[v] = c
        uncolored.remove(v)
        for nb in graph.adj[v]:
            if colors[nb] == -1:
                neighbor_colors[nb].add(c)

    return colors


class BreakoutTabuColoring:
    def __init__(
        self,
        graph: Graph,
        k: int,
        rng: random.Random,
        candidate_vertices: int = 96,
        base_tenure: int = 8,
        max_tenure: int = 120,
        stagnation_iters: int = 4000,
        restart_iters: int = 16000,
        elite_size: int = 5,
        verbose: bool = False,
    ) -> None:
        self.g = graph
        self.k = k
        self.rng = rng
        self.candidate_vertices = max(16, candidate_vertices)
        self.base_tenure = max(1, base_tenure)
        self.max_tenure = max(self.base_tenure + 1, max_tenure)
        self.stagnation_iters = max(200, stagnation_iters)
        self.restart_iters = max(self.stagnation_iters + 1, restart_iters)
        self.elite_size = max(1, elite_size)
        self.verbose = verbose

        # Edge penalties start at 1. Breakout increments penalties of conflict edges.
        self.edge_penalty = [1] * len(graph.edges)

    def _initialize_counts(self, colors: List[int]) -> Tuple[List[List[int]], List[int], int, int]:
        n, k, g = self.g.n, self.k, self.g
        adj_color_count = [[0] * k for _ in range(n)]
        vertex_weighted_conf = [0] * n
        conflicts = 0
        weighted_conflicts = 0

        for u, v in g.edges:
            cu = colors[u]
            cv = colors[v]
            adj_color_count[u][cv] += 1
            adj_color_count[v][cu] += 1
            if cu == cv:
                idx = g.edge_index[(u, v)]
                p = self.edge_penalty[idx]
                conflicts += 1
                weighted_conflicts += p
                vertex_weighted_conf[u] += p
                vertex_weighted_conf[v] += p

        return adj_color_count, vertex_weighted_conf, conflicts, weighted_conflicts

    def _weighted_delta(self, v: int, old_c: int, new_c: int, colors: List[int]) -> Tuple[int, int]:
        """Return (raw_delta, weighted_delta) if v changes old_c -> new_c."""
        raw_delta = 0
        weighted_delta = 0
        for nb in self.g.adj[v]:
            cnb = colors[nb]
            if cnb == old_c:
                raw_delta -= 1
                weighted_delta -= self.edge_penalty[self.g.edge_index[(v, nb)]]
            if cnb == new_c:
                raw_delta += 1
                weighted_delta += self.edge_penalty[self.g.edge_index[(v, nb)]]
        return raw_delta, weighted_delta

    def _apply_move(
        self,
        v: int,
        new_c: int,
        colors: List[int],
        adj_color_count: List[List[int]],
        vertex_weighted_conf: List[int],
        conflicts: int,
        weighted_conflicts: int,
    ) -> Tuple[int, int]:
        old_c = colors[v]
        if old_c == new_c:
            return conflicts, weighted_conflicts

        raw_delta = 0
        weighted_delta = 0

        # Update vertex weighted conflict values locally.
        # Recompute v after move; update neighbors by edge changes.
        vertex_weighted_conf[v] = 0

        for nb in self.g.adj[v]:
            idx = self.g.edge_index[(v, nb)]
            p = self.edge_penalty[idx]
            cnb = colors[nb]

            was_conf = cnb == old_c
            will_conf = cnb == new_c

            if was_conf:
                raw_delta -= 1
                weighted_delta -= p
                vertex_weighted_conf[nb] -= p
            if will_conf:
                raw_delta += 1
                weighted_delta += p
                vertex_weighted_conf[nb] += p
                vertex_weighted_conf[v] += p

            # For neighbor, v leaves old_c and enters new_c.
            adj_color_count[nb][old_c] -= 1
            adj_color_count[nb][new_c] += 1

        colors[v] = new_c

        # Rebuild row for v cheaply using adjacency.
        row = [0] * self.k
        for nb in self.g.adj[v]:
            row[colors[nb]] += 1
        adj_color_count[v] = row

        return conflicts + raw_delta, weighted_conflicts + weighted_delta

    def _conflict_vertices(self, colors: List[int], adj_color_count: List[List[int]]) -> List[int]:
        return [v for v in range(self.g.n) if adj_color_count[v][colors[v]] > 0]

    def _breakout_update(
        self,
        colors: List[int],
        vertex_weighted_conf: List[int],
    ) -> int:
        """Increase penalties on current conflict edges and return total added penalty."""
        added = 0
        for u, v in self.g.edges:
            if colors[u] == colors[v]:
                idx = self.g.edge_index[(u, v)]
                self.edge_penalty[idx] += 1
                vertex_weighted_conf[u] += 1
                vertex_weighted_conf[v] += 1
                added += 1
        # Prevent penalty explosion: periodically normalize without losing learned bias too much.
        if max(self.edge_penalty, default=1) > 100:
            for i, p in enumerate(self.edge_penalty):
                self.edge_penalty[i] = max(1, p // 2)
        return added

    def _perturb(self, colors: List[int], strength: float) -> List[int]:
        """Return perturbed copy of coloring. Bias toward high-conflict vertices later in solve."""
        new_colors = list(colors)
        n = self.g.n
        count = max(1, int(n * strength))
        vertices = list(range(n))
        self.rng.shuffle(vertices)
        for v in vertices[:count]:
            old = new_colors[v]
            if self.k <= 1:
                continue
            c = self.rng.randrange(self.k - 1)
            if c >= old:
                c += 1
            new_colors[v] = c
        return new_colors

    def _elite_insert(self, elite: List[Tuple[int, int, List[int]]], conflicts: int, weighted: int, colors: List[int]) -> None:
        item = (conflicts, weighted, list(colors))
        elite.append(item)
        elite.sort(key=lambda x: (x[0], x[1]))
        # Deduplicate by conflict + prefix pattern to avoid keeping identical solutions too often.
        trimmed: List[Tuple[int, int, List[int]]] = []
        seen = set()
        for c, w, col in elite:
            sig = (c, tuple(col[: min(50, len(col))]))
            if sig not in seen:
                seen.add(sig)
                trimmed.append((c, w, col))
            if len(trimmed) >= self.elite_size:
                break
        elite[:] = trimmed

    def solve(self, time_limit: float, initial_colors: Optional[List[int]] = None) -> RunResult:
        start = time.perf_counter()
        rng = self.rng
        n, k = self.g.n, self.k

        if initial_colors is None:
            colors = randomized_greedy_coloring(self.g, k, rng)
        else:
            colors = list(initial_colors)

        adj_color_count, vertex_weighted_conf, conflicts, weighted_conflicts = self._initialize_counts(colors)
        best_colors = list(colors)
        best_conflicts = conflicts
        best_weighted = weighted_conflicts

        elite: List[Tuple[int, int, List[int]]] = []
        self._elite_insert(elite, conflicts, weighted_conflicts, colors)

        tabu_until = [[0] * k for _ in range(n)]
        tenure = self.base_tenure
        iterations = 0
        last_improvement = 0
        last_breakout = 0
        restarts = 0
        breakouts = 0

        while time.perf_counter() - start < time_limit and best_conflicts > 0:
            iterations += 1

            conflict_vertices = self._conflict_vertices(colors, adj_color_count)
            if not conflict_vertices:
                conflicts = 0
                weighted_conflicts = 0
                best_conflicts = 0
                best_weighted = 0
                best_colors = list(colors)
                break

            # Candidate selection: sample high conflict vertices + random conflict vertices.
            conflict_vertices.sort(key=lambda v: vertex_weighted_conf[v], reverse=True)
            top_count = min(len(conflict_vertices), self.candidate_vertices // 2)
            candidates = conflict_vertices[:top_count]
            if len(conflict_vertices) > top_count:
                rest = conflict_vertices[top_count:]
                rng.shuffle(rest)
                candidates += rest[: max(0, self.candidate_vertices - len(candidates))]

            best_v = -1
            best_c = -1
            best_raw_delta = math.inf
            best_weighted_delta = math.inf
            best_score = math.inf

            for v in candidates:
                old_c = colors[v]
                old_same = adj_color_count[v][old_c]

                # Evaluate all colors. k is usually small/moderate; this is fine and robust.
                for c in range(k):
                    if c == old_c:
                        continue

                    # Fast raw delta from incremental counts.
                    raw_delta_fast = adj_color_count[v][c] - old_same

                    # Weighted delta still scans neighbors; acceptable for selected vertices.
                    # This is the part that enables breakout penalties.
                    _, weighted_delta = self._weighted_delta(v, old_c, c, colors)
                    new_conflicts = conflicts + raw_delta_fast
                    new_weighted = weighted_conflicts + weighted_delta

                    is_tabu = tabu_until[v][c] > iterations
                    aspiration = new_conflicts < best_conflicts
                    if is_tabu and not aspiration:
                        continue

                    # Score: prioritize raw conflicts, then weighted conflicts, then noise.
                    noise = rng.random() * 1e-6
                    score = (new_conflicts * 1000000) + new_weighted + noise

                    if score < best_score:
                        best_score = score
                        best_v = v
                        best_c = c
                        best_raw_delta = raw_delta_fast
                        best_weighted_delta = weighted_delta

            if best_v == -1:
                # If all candidate moves are tabu, force a random admissible move.
                best_v = rng.choice(conflict_vertices)
                old = colors[best_v]
                best_c = rng.randrange(k - 1)
                if best_c >= old:
                    best_c += 1
                best_raw_delta, best_weighted_delta = self._weighted_delta(best_v, old, best_c, colors)

            old_color = colors[best_v]
            conflicts, weighted_conflicts = self._apply_move(
                best_v,
                best_c,
                colors,
                adj_color_count,
                vertex_weighted_conf,
                conflicts,
                weighted_conflicts,
            )

            # Forbid returning this vertex to the old color.
            dynamic = tenure + rng.randrange(0, max(2, min(20, conflicts + 2)))
            tabu_until[best_v][old_color] = iterations + dynamic

            if conflicts < best_conflicts or (conflicts == best_conflicts and weighted_conflicts < best_weighted):
                best_conflicts = conflicts
                best_weighted = weighted_conflicts
                best_colors = list(colors)
                last_improvement = iterations
                tenure = max(self.base_tenure, int(tenure * 0.92))
                self._elite_insert(elite, conflicts, weighted_conflicts, colors)
                if self.verbose:
                    print(f"  iter={iterations} best_conflicts={best_conflicts} weighted={best_weighted} tenure={tenure}")
            else:
                # Mild increase on non-improvement.
                if iterations % 500 == 0:
                    tenure = min(self.max_tenure, int(tenure * 1.03) + 1)

            # Breakout: punish current local-minimum conflict edges.
            if iterations - last_improvement >= self.stagnation_iters and iterations - last_breakout >= self.stagnation_iters // 2:
                added = self._breakout_update(colors, vertex_weighted_conf)
                weighted_conflicts += added
                breakouts += 1
                last_breakout = iterations
                tenure = min(self.max_tenure, int(tenure * 1.25) + 1)

            # Strategic restart from elite if stagnation persists.
            if iterations - last_improvement >= self.restart_iters:
                restarts += 1
                if elite and rng.random() < 0.80:
                    _, _, elite_colors = rng.choice(elite[: min(len(elite), 3)])
                    strength = 0.03 if best_conflicts <= 5 else 0.06
                    colors = self._perturb(elite_colors, strength=strength)
                else:
                    colors = randomized_greedy_coloring(self.g, k, rng)

                adj_color_count, vertex_weighted_conf, conflicts, weighted_conflicts = self._initialize_counts(colors)
                last_improvement = iterations
                tenure = min(self.max_tenure, max(self.base_tenure, int(tenure * 1.15) + 1))

        elapsed = time.perf_counter() - start
        verified = verify_coloring(self.g, best_colors, k)
        if verified != best_conflicts:
            # Keep the verified value if penalty/count drift occurred somehow.
            best_conflicts = verified

        return RunResult(
            instance="",
            method="AHGC_BreakoutTabu",
            seed=-1,
            k=k,
            feasible=best_conflicts == 0,
            conflicts=best_conflicts,
            weighted_conflicts=best_weighted,
            iterations=iterations,
            restarts=restarts,
            breakouts=breakouts,
            time_seconds=elapsed,
            colors=best_colors,
        )


def solve_single_k(graph: Graph, k: int, time_limit: float, seed: int, verbose: bool = False) -> RunResult:
    rng = random.Random(seed)

    # Multi-start seed selection inside one run: DSATUR reduced + randomized greedy alternatives.
    ub, dsatur_colors = dsatur_coloring(graph, rng)
    if ub <= k:
        initial = [c % k for c in dsatur_colors]
    else:
        initial = reduce_to_k_greedy(graph, dsatur_colors, k, rng)

    solver = BreakoutTabuColoring(
        graph=graph,
        k=k,
        rng=rng,
        candidate_vertices=max(64, min(220, graph.n // 2)),
        base_tenure=max(7, min(35, graph.n // 40 + 7)),
        max_tenure=max(80, min(200, graph.n // 4 + 50)),
        stagnation_iters=max(1500, min(12000, graph.n * 8)),
        restart_iters=max(6000, min(40000, graph.n * 25)),
        elite_size=6,
        verbose=verbose,
    )
    result = solver.solve(time_limit=time_limit, initial_colors=initial)
    result.seed = seed
    return result


def solve_try_extra(graph: Graph, k_ref: int, max_extra: int, time_limit: float, seed: int, runs_per_k: int, instance: str, csv_path: str = "") -> RunResult:
    best: Optional[RunResult] = None
    for extra in range(max_extra + 1):
        k = k_ref + extra
        print(f"\nTrying k={k} (extra={extra})")
        feasible_found = False
        for run in range(runs_per_k):
            run_seed = seed + extra * 1000003 + run * 10007
            result = solve_single_k(graph, k, time_limit, run_seed)
            result.instance = instance
            print(
                f"  run {run+1}/{runs_per_k}, seed={run_seed}, "
                f"feasible={result.feasible}, conflicts={result.conflicts}, "
                f"iters={result.iterations}, breakouts={result.breakouts}, restarts={result.restarts}, "
                f"time={result.time_seconds:.3f}s"
            )
            if csv_path:
                append_csv(csv_path, result)
            if best is None or (result.feasible and not best.feasible) or (result.feasible == best.feasible and result.conflicts < best.conflicts):
                best = result
            if result.feasible:
                feasible_found = True
        if feasible_found:
            break
    assert best is not None
    return best


def write_solution(path: str, result: RunResult) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("c AHGC Breakout Tabu solution\n")
        f.write(f"c k = {result.k}\n")
        f.write(f"c feasible = {int(result.feasible)}\n")
        f.write(f"c conflicts = {result.conflicts}\n")
        for i, c in enumerate(result.colors, start=1):
            f.write(f"v {i} {c + 1}\n")


def append_csv(path: str, result: RunResult) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "instance",
        "method",
        "seed",
        "k",
        "feasible",
        "conflicts",
        "weighted_conflicts",
        "iterations",
        "breakouts",
        "restarts",
        "time_seconds",
    ]
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "instance": result.instance,
                "method": result.method,
                "seed": result.seed,
                "k": result.k,
                "feasible": int(result.feasible),
                "conflicts": result.conflicts,
                "weighted_conflicts": result.weighted_conflicts,
                "iterations": result.iterations,
                "breakouts": result.breakouts,
                "restarts": result.restarts,
                "time_seconds": f"{result.time_seconds:.6f}",
            }
        )


def run_self_test() -> None:
    cycle_5 = Graph(
        n=5,
        edges=[(0, 1), (0, 4), (1, 2), (2, 3), (3, 4)],
        adj=[[1, 4], [0, 2], [1, 3], [2, 4], [0, 3]],
        degree=[2, 2, 2, 2, 2],
        edge_index={},
    )
    cycle_5.edge_index = {}
    for i, (u, v) in enumerate(cycle_5.edges):
        cycle_5.edge_index[(u, v)] = i
        cycle_5.edge_index[(v, u)] = i
    r = solve_single_k(cycle_5, 3, 2.0, 123)
    assert r.feasible, r
    assert verify_coloring(cycle_5, r.colors, 3) == 0

    complete_4 = Graph(
        n=4,
        edges=[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        adj=[[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
        degree=[3, 3, 3, 3],
        edge_index={},
    )
    for i, (u, v) in enumerate(complete_4.edges):
        complete_4.edge_index[(u, v)] = i
        complete_4.edge_index[(v, u)] = i
    r2 = solve_single_k(complete_4, 4, 2.0, 456)
    assert r2.feasible, r2
    assert verify_coloring(complete_4, r2.colors, 4) == 0
    print("Self-test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="AHGC Breakout Tabu for fixed-k graph coloring")
    parser.add_argument("path", nargs="?", help="Path to DIMACS .col file")
    parser.add_argument("--k", type=int, default=None, help="Target number of colors")
    parser.add_argument("--k-ref", type=int, default=None, help="Reference k for try-extra mode")
    parser.add_argument("--try-extra", type=int, default=0, help="Try k-ref, k-ref+1, ..., k-ref+try_extra")
    parser.add_argument("--time-limit", type=float, default=60.0, help="Time limit per run, per k")
    parser.add_argument("--runs", type=int, default=1, help="Independent runs")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--csv", type=str, default="", help="Append detailed results to CSV")
    parser.add_argument("--output", type=str, default="", help="Write best solution")
    parser.add_argument("--verbose", action="store_true", help="Print improvements during search")
    parser.add_argument("--self-test", action="store_true", help="Run internal tests")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not args.path:
        parser.error("You must provide a DIMACS .col file path.")

    graph = Graph.from_dimacs_col(args.path)
    ub, _ = dsatur_coloring(graph)
    target_k = args.k if args.k is not None else args.k_ref
    if target_k is None:
        target_k = ub

    instance = os.path.basename(args.path)
    print(f"Graph: n={graph.n}, m={len(graph.edges)}")
    print(f"DSATUR upper bound: {ub}")
    print(f"Target k: {target_k}")

    best: Optional[RunResult] = None

    if args.k_ref is not None and args.try_extra > 0:
        best = solve_try_extra(
            graph=graph,
            k_ref=args.k_ref,
            max_extra=args.try_extra,
            time_limit=args.time_limit,
            seed=args.seed,
            runs_per_k=args.runs,
            instance=instance,
            csv_path=args.csv,
        )
    else:
        for run in range(args.runs):
            seed = args.seed + run * 10007
            print(f"\nRun {run+1}/{args.runs}, seed={seed}")
            result = solve_single_k(graph, target_k, args.time_limit, seed, verbose=args.verbose)
            result.instance = instance
            print(
                f"k={result.k}, feasible={result.feasible}, conflicts={result.conflicts}, "
                f"iters={result.iterations}, breakouts={result.breakouts}, restarts={result.restarts}, "
                f"time={result.time_seconds:.3f}s"
            )
            if args.csv:
                append_csv(args.csv, result)
            if best is None or result.conflicts < best.conflicts or (result.feasible and not best.feasible):
                best = result

    assert best is not None
    print("\nBest result")
    print(f"k = {best.k}")
    print(f"feasible = {best.feasible}")
    print(f"conflicts = {best.conflicts}")
    print(f"verification conflicts = {verify_coloring(graph, best.colors, best.k)}")

    if args.output:
        write_solution(args.output, best)
        print(f"Solution written to: {args.output}")


if __name__ == "__main__":
    main()
