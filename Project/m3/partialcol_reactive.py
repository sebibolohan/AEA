#!/usr/bin/env python3
"""
PartialCol Reactive Tabu Search for the fixed-k Graph Coloring Problem.

The method uses partial legal colorings:
    - k independent color classes
    - U = set of uncolored vertices
The objective is to minimize |U|. A feasible k-coloring is found when |U| = 0.

Input format: DIMACS .col files.

Example:
    python partialcol_reactive.py data/DSJC125.9.col --k 44 --time-limit 30 --seed 0
    python partialcol_reactive.py data/DSJC125.9.col --k-ref 44 --search-down --time-limit 20 --runs 5 --csv results.csv
"""

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict


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

        degree = [len(neighbors) for neighbors in adj]
        return cls(n=n, edges=edges, adj=adj, degree=degree)


@dataclass
class RunResult:
    k: int
    feasible: bool
    uncolored: int
    colors: List[int]
    iterations: int
    time_seconds: float
    seed: int


def verify_coloring(graph: Graph, colors: List[int], k: int) -> int:
    if len(colors) != graph.n:
        raise ValueError("Coloring length mismatch")
    conflicts = 0
    for color in colors:
        if color < 0 or color >= k:
            raise ValueError("Color out of range")
    for u, v in graph.edges:
        if colors[u] == colors[v]:
            conflicts += 1
    return conflicts


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

    used_colors = 1
    while uncolored:
        v = max(uncolored, key=lambda x: (len(neighbor_colors[x]), graph.degree[x], x))
        c = 0
        while c in neighbor_colors[v]:
            c += 1
        colors[v] = c
        used_colors = max(used_colors, c + 1)
        uncolored.remove(v)
        for nb in graph.adj[v]:
            if colors[nb] == -1:
                neighbor_colors[nb].add(c)

    return used_colors, colors


def greedy_clique_lower_bound(graph: Graph) -> int:
    order = sorted(range(graph.n), key=lambda v: graph.degree[v], reverse=True)
    clique: List[int] = []
    for v in order:
        adj_v = set(graph.adj[v])
        if all(u in adj_v for u in clique):
            clique.append(v)
    return max(1, len(clique))


class PartialColReactiveTabu:
    def __init__(
        self,
        graph: Graph,
        k: int,
        rng: random.Random,
        base_tenure: int = 7,
        candidate_sample: int = 80,
        stagnation_limit: int = 2500,
    ) -> None:
        self.graph = graph
        self.k = k
        self.rng = rng
        self.base_tenure = max(1, base_tenure)
        self.candidate_sample = max(1, candidate_sample)
        self.stagnation_limit = max(100, stagnation_limit)

    def initial_partial_coloring(self, seed_colors: Optional[List[int]] = None) -> Tuple[List[int], List[Set[int]], Set[int]]:
        """
        Builds a legal partial k-coloring.
        colors[v] = -1 means uncolored.
        Each color class is always independent.
        """
        n = self.graph.n
        colors = [-1] * n
        classes: List[Set[int]] = [set() for _ in range(self.k)]
        U: Set[int] = set()

        if seed_colors is not None:
            order = sorted(range(n), key=lambda v: (seed_colors[v], -self.graph.degree[v], v))
        else:
            # DSATUR-like randomized order: high degree first, with small noise.
            order = list(range(n))
            order.sort(key=lambda v: (self.graph.degree[v] + self.rng.random()), reverse=True)

        for v in order:
            if seed_colors is not None:
                preferred = seed_colors[v] % self.k
                color_order = [preferred] + [c for c in range(self.k) if c != preferred]
            else:
                color_order = list(range(self.k))
                self.rng.shuffle(color_order)
                color_order.sort(key=lambda c: self._class_conflicts(v, classes[c]))

            placed = False
            for c in color_order:
                if self._class_conflicts(v, classes[c]) == 0:
                    colors[v] = c
                    classes[c].add(v)
                    placed = True
                    break

            if not placed:
                U.add(v)

        return colors, classes, U

    def _class_conflicts(self, v: int, color_class: Set[int]) -> int:
        # Count neighbors of v already present in color_class.
        return sum(1 for nb in self.graph.adj[v] if nb in color_class)

    def _conflicting_vertices_in_class(self, v: int, color_class: Set[int]) -> List[int]:
        return [nb for nb in self.graph.adj[v] if nb in color_class]

    def solve(self, time_limit: float, seed_colors: Optional[List[int]] = None) -> RunResult:
        start = time.perf_counter()
        colors, classes, U = self.initial_partial_coloring(seed_colors)

        best_colors = list(colors)
        best_uncolored = len(U)

        # Tabu[v][c] means vertex v is forbidden to be moved to color c until iteration value.
        tabu_until = [[0] * self.k for _ in range(self.graph.n)]

        iteration = 0
        last_improvement = 0
        tenure = self.base_tenure

        while time.perf_counter() - start < time_limit and len(U) > 0:
            iteration += 1

            # Reactive tenure: if no progress, increase diversification.
            if iteration - last_improvement > self.stagnation_limit:
                tenure = min(80, int(tenure * 1.35) + 1)
                self._shake(colors, classes, U, strength=0.03)
                last_improvement = iteration
            elif iteration % 2000 == 0 and tenure > self.base_tenure:
                tenure = max(self.base_tenure, int(tenure * 0.95))

            candidate_vertices = list(U)
            self.rng.shuffle(candidate_vertices)
            candidate_vertices = candidate_vertices[:min(self.candidate_sample, len(candidate_vertices))]

            best_v = -1
            best_c = -1
            best_removed: List[int] = []
            best_delta = math.inf
            best_new_u_size = math.inf

            for v in candidate_vertices:
                for c in range(self.k):
                    removed = self._conflicting_vertices_in_class(v, classes[c])
                    # Move v from U to c, remove conflicting colored vertices into U.
                    new_u_size = len(U) - 1 + len(removed)
                    delta = new_u_size - len(U)

                    is_tabu = tabu_until[v][c] > iteration
                    aspiration = new_u_size < best_uncolored
                    if is_tabu and not aspiration:
                        continue

                    if (
                        delta < best_delta
                        or (delta == best_delta and new_u_size < best_new_u_size)
                        or (delta == best_delta and new_u_size == best_new_u_size and self.rng.random() < 0.10)
                    ):
                        best_v = v
                        best_c = c
                        best_removed = removed
                        best_delta = delta
                        best_new_u_size = new_u_size

            if best_v == -1:
                # Fallback if all moves were tabu.
                best_v = self.rng.choice(tuple(U))
                best_c = self.rng.randrange(self.k)
                best_removed = self._conflicting_vertices_in_class(best_v, classes[best_c])

            # Apply move: insert best_v in best_c and uncolor conflicting vertices.
            U.remove(best_v)
            colors[best_v] = best_c
            classes[best_c].add(best_v)

            for u in best_removed:
                classes[best_c].remove(u)
                colors[u] = -1
                U.add(u)
                # Avoid immediately returning removed vertices to the same class.
                tabu_until[u][best_c] = iteration + tenure + self.rng.randrange(0, 10)

            # Avoid immediately undoing current placement too aggressively.
            tabu_until[best_v][best_c] = iteration + max(1, tenure // 2)

            if len(U) < best_uncolored:
                best_uncolored = len(U)
                best_colors = list(colors)
                last_improvement = iteration
                tenure = max(self.base_tenure, int(tenure * 0.90))

        elapsed = time.perf_counter() - start

        feasible = best_uncolored == 0
        final_colors = best_colors

        if feasible:
            conflicts = verify_coloring(self.graph, final_colors, self.k)
            if conflicts != 0:
                raise RuntimeError("Internal error: feasible partial coloring has conflicts")

        return RunResult(
            k=self.k,
            feasible=feasible,
            uncolored=best_uncolored,
            colors=final_colors,
            iterations=iteration,
            time_seconds=elapsed,
            seed=-1,
        )

    def _shake(self, colors: List[int], classes: List[Set[int]], U: Set[int], strength: float) -> None:
        """
        Diversification: uncolor a small fraction of high-degree colored vertices.
        This keeps the partial coloring legal and gives the search new freedom.
        """
        colored = [v for v, c in enumerate(colors) if c != -1]
        if not colored:
            return

        count = max(1, int(len(colors) * strength))
        colored.sort(key=lambda v: self.graph.degree[v], reverse=True)
        pool = colored[:min(len(colored), max(count * 10, 20))]
        self.rng.shuffle(pool)

        for v in pool[:count]:
            c = colors[v]
            if c != -1:
                classes[c].remove(v)
                colors[v] = -1
                U.add(v)


def write_solution(path: str, result: RunResult) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"c PartialCol solution\n")
        f.write(f"c k = {result.k}\n")
        f.write(f"c feasible = {result.feasible}\n")
        f.write(f"c uncolored = {result.uncolored}\n")
        for i, c in enumerate(result.colors, start=1):
            if c == -1:
                f.write(f"v {i} 0\n")
            else:
                f.write(f"v {i} {c + 1}\n")


def solve_single_k(graph: Graph, k: int, time_limit: float, seed: int) -> RunResult:
    rng = random.Random(seed)
    ub, dsatur_colors = dsatur_coloring(graph)
    seed_colors = dsatur_colors if ub >= k else None

    solver = PartialColReactiveTabu(
        graph=graph,
        k=k,
        rng=rng,
        base_tenure=max(7, min(30, graph.n // 30 + 5)),
        candidate_sample=max(40, min(160, graph.n // 4)),
        stagnation_limit=max(1000, min(8000, graph.n * 4)),
    )
    result = solver.solve(time_limit=time_limit, seed_colors=seed_colors)
    result.seed = seed
    return result


def search_down(graph: Graph, start_k: int, time_limit: float, seed: int, max_failed_k: int) -> RunResult:
    lower_bound = greedy_clique_lower_bound(graph)
    best_feasible: Optional[RunResult] = None
    failed = 0

    for k in range(start_k, lower_bound - 1, -1):
        print(f"Trying k={k} ...")
        result = solve_single_k(graph, k, time_limit, seed + (start_k - k) * 1009)
        print(
            f"  feasible={result.feasible}, uncolored={result.uncolored}, "
            f"iters={result.iterations}, time={result.time_seconds:.3f}s"
        )

        if result.feasible:
            best_feasible = result
            failed = 0
        else:
            failed += 1
            if failed >= max_failed_k:
                break

    if best_feasible is None:
        # Return the last tried result if no feasible solution was found.
        return result
    return best_feasible


def append_csv(path: str, row: Dict[str, object]) -> None:
    fieldnames = [
        "instance", "method", "seed", "k", "feasible", "uncolored",
        "iterations", "time_seconds"
    ]
    file_exists = False
    try:
        with open(path, "r", encoding="utf-8"):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_self_test() -> None:
    cycle_5 = Graph(
        n=5,
        edges=[(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)],
        adj=[[1, 4], [0, 2], [1, 3], [2, 4], [3, 0]],
        degree=[2, 2, 2, 2, 2],
    )
    result = solve_single_k(cycle_5, 3, 1.0, 123)
    assert result.feasible
    assert verify_coloring(cycle_5, result.colors, 3) == 0

    bipartite = Graph(
        n=4,
        edges=[(0, 2), (0, 3), (1, 2), (1, 3)],
        adj=[[2, 3], [2, 3], [0, 1], [0, 1]],
        degree=[2, 2, 2, 2],
    )
    result2 = solve_single_k(bipartite, 2, 1.0, 456)
    assert result2.feasible
    assert verify_coloring(bipartite, result2.colors, 2) == 0
    print("Self-test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PartialCol Reactive Tabu Search for fixed-k graph coloring"
    )
    parser.add_argument("path", nargs="?", help="Path to DIMACS .col file")
    parser.add_argument("--k", type=int, default=None, help="Target number of colors")
    parser.add_argument("--k-ref", type=int, default=None, help="Reference k for reporting/search-down")
    parser.add_argument("--search-down", action="store_true", help="Try k-ref, k-ref-1, ... until failures")
    parser.add_argument("--time-limit", type=float, default=20.0, help="Time limit per k, per run, in seconds")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--runs", type=int, default=1, help="Independent runs")
    parser.add_argument("--max-failed-k", type=int, default=1, help="Stop search-down after this many failed k values")
    parser.add_argument("--csv", type=str, default="", help="Append results to CSV file")
    parser.add_argument("--output", type=str, default="", help="Write best solution to file")
    parser.add_argument("--self-test", action="store_true", help="Run internal tests")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not args.path:
        parser.error("You must provide a DIMACS .col file path.")

    graph = Graph.from_dimacs_col(args.path)
    ub, dsatur_colors = dsatur_coloring(graph)
    lb = greedy_clique_lower_bound(graph)

    print(f"Graph: n={graph.n}, m={len(graph.edges)}")
    print(f"DSATUR upper bound: {ub}")
    print(f"Greedy clique lower bound: {lb}")

    target_k = args.k if args.k is not None else args.k_ref
    if target_k is None:
        target_k = ub

    best_overall: Optional[RunResult] = None

    for run in range(args.runs):
        seed = args.seed + run * 10007
        print(f"\nRun {run + 1}/{args.runs}, seed={seed}")

        if args.search_down:
            result = search_down(
                graph=graph,
                start_k=target_k,
                time_limit=args.time_limit,
                seed=seed,
                max_failed_k=args.max_failed_k,
            )
        else:
            result = solve_single_k(graph, target_k, args.time_limit, seed)
            print(
                f"k={result.k}, feasible={result.feasible}, uncolored={result.uncolored}, "
                f"iters={result.iterations}, time={result.time_seconds:.3f}s"
            )

        if best_overall is None:
            best_overall = result
        else:
            # Prefer feasible with smaller k; otherwise fewer uncolored vertices.
            if result.feasible and (not best_overall.feasible or result.k < best_overall.k):
                best_overall = result
            elif result.feasible == best_overall.feasible and result.uncolored < best_overall.uncolored:
                best_overall = result

        if args.csv:
            append_csv(args.csv, {
                "instance": args.path.split("/")[-1],
                "method": "PartialColReactiveTabu",
                "seed": seed,
                "k": result.k,
                "feasible": int(result.feasible),
                "uncolored": result.uncolored,
                "iterations": result.iterations,
                "time_seconds": f"{result.time_seconds:.6f}",
            })

    assert best_overall is not None

    print("\nBest result")
    print(f"k = {best_overall.k}")
    print(f"feasible = {best_overall.feasible}")
    print(f"uncolored = {best_overall.uncolored}")

    if best_overall.feasible:
        conflicts = verify_coloring(graph, best_overall.colors, best_overall.k)
        print(f"verification conflicts = {conflicts}")

    if args.output:
        write_solution(args.output, best_overall)
        print(f"Solution written to: {args.output}")


if __name__ == "__main__":
    main()
