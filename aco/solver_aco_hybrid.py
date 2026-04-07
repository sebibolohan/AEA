#!/usr/bin/env python3

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional


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
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("c"):
                    continue
                parts = line.split()
                tag = parts[0].lower()
                if tag == "p":
                    if len(parts) < 4 or parts[1].lower() != "edge":
                        raise ValueError(f"Unsupported DIMACS line: {line}")
                    n = int(parts[2])
                elif tag == "e":
                    if len(parts) != 3:
                        raise ValueError(f"Invalid edge line: {line}")
                    u = int(parts[1]) - 1
                    v = int(parts[2]) - 1
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    edges.append((u, v))
        if n <= 0:
            raise ValueError("Invalid DIMACS .col file: missing or invalid 'p edge n m' line")

        edges = sorted(set(edges))
        adj = [[] for _ in range(n)]
        for u, v in edges:
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError("Edge endpoint out of range")
            adj[u].append(v)
            adj[v].append(u)
        degree = [len(nei) for nei in adj]
        return cls(n=n, edges=edges, adj=adj, degree=degree)


@dataclass
class ColoringResult:
    colors: List[int]
    conflicts: int
    k: int
    time_seconds: float


def verify_coloring(graph: Graph, colors: List[int], k: int) -> int:
    if len(colors) != graph.n:
        raise ValueError("Coloring length mismatch")
    for c in colors:
        if c < 0 or c >= k:
            raise ValueError("Color out of range")
    conflicts = 0
    for u, v in graph.edges:
        if colors[u] == colors[v]:
            conflicts += 1
    return conflicts


def dsatur_upper_bound(graph: Graph) -> Tuple[int, List[int]]:
    n = graph.n
    colors = [-1] * n
    neighbor_colors = [set() for _ in range(n)]
    uncolored = set(range(n))

    first = max(range(n), key=lambda x: graph.degree[x])
    colors[first] = 0
    uncolored.remove(first)
    for nb in graph.adj[first]:
        neighbor_colors[nb].add(0)

    used_colors = 1
    while uncolored:
        v = max(uncolored, key=lambda x: (len(neighbor_colors[x]), graph.degree[x], x))
        forbidden = neighbor_colors[v]
        c = 0
        while c in forbidden:
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
    clique_set = set()
    for v in order:
        ok = True
        adjv = set(graph.adj[v])
        for u in clique:
            if u not in adjv:
                ok = False
                break
        if ok:
            clique.append(v)
            clique_set.add(v)
    return max(1, len(clique_set))


def random_k_coloring(n: int, k: int, rng: random.Random) -> List[int]:
    return [rng.randrange(k) for _ in range(n)]


class FixedKHybridACO:
    def __init__(
        self,
        graph: Graph,
        k: int,
        rng: random.Random,
        ants: int = 12,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation: float = 0.20,
        q: float = 6.0,
        tabu_tenure: int = 12,
        max_stagnation_moves: int = 2000,
    ) -> None:
        self.graph = graph
        self.k = k
        self.rng = rng
        self.ants = max(1, ants)
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.q = q
        self.base_tabu_tenure = max(1, tabu_tenure)
        self.max_stagnation_moves = max_stagnation_moves
        self.pheromone = [[1.0 for _ in range(k)] for _ in range(graph.n)]

    def search(self, time_limit: float, seed_colors: Optional[List[int]] = None) -> ColoringResult:
        start = time.perf_counter()
        best_colors: Optional[List[int]] = None
        best_conflicts = math.inf

        if seed_colors is not None:
            initial = list(seed_colors)
            for i in range(len(initial)):
                initial[i] %= self.k
            improved, conf = self.tabu_search(initial, time_limit=max(0.05, min(0.5, time_limit * 0.10)))
            best_colors = improved
            best_conflicts = conf
            if conf == 0:
                return ColoringResult(improved, 0, self.k, time.perf_counter() - start)

        while time.perf_counter() - start < time_limit:
            iter_best_colors: Optional[List[int]] = None
            iter_best_conflicts = math.inf

            for _ in range(self.ants):
                remaining = time_limit - (time.perf_counter() - start)
                if remaining <= 0:
                    break
                constructed = self.construct_solution()
                local_budget = min(0.50, max(0.03, remaining / max(1, self.ants)))
                improved, conflicts = self.tabu_search(constructed, time_limit=local_budget)

                if conflicts < iter_best_conflicts:
                    iter_best_conflicts = conflicts
                    iter_best_colors = improved
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_colors = improved
                    if best_conflicts == 0:
                        return ColoringResult(best_colors, 0, self.k, time.perf_counter() - start)

            if iter_best_colors is None:
                break

            self.update_pheromone(iter_best_colors, iter_best_conflicts, best_colors, best_conflicts)

        if best_colors is None:
            best_colors = random_k_coloring(self.graph.n, self.k, self.rng)
            best_conflicts = verify_coloring(self.graph, best_colors, self.k)
        return ColoringResult(best_colors, best_conflicts, self.k, time.perf_counter() - start)

    def construct_solution(self) -> List[int]:
        n = self.graph.n
        colors = [-1] * n
        saturation = [0] * n
        seen_neighbor_colors = [set() for _ in range(n)]
        uncolored = set(range(n))

        while uncolored:
            v = max(uncolored, key=lambda x: (saturation[x], self.graph.degree[x], x))
            weights: List[float] = []
            for color in range(self.k):
                local_conf = 0
                for nb in self.graph.adj[v]:
                    if colors[nb] == color:
                        local_conf += 1
                tau = self.pheromone[v][color] ** self.alpha
                eta = (1.0 / (1.0 + local_conf)) ** self.beta
                weights.append(max(1e-12, tau * eta))
            chosen = self.weighted_choice(weights)
            colors[v] = chosen
            uncolored.remove(v)
            for nb in self.graph.adj[v]:
                if colors[nb] == -1 and chosen not in seen_neighbor_colors[nb]:
                    seen_neighbor_colors[nb].add(chosen)
                    saturation[nb] += 1

        return colors

    def weighted_choice(self, weights: List[float]) -> int:
        total = sum(weights)
        if total <= 0:
            return self.rng.randrange(len(weights))
        pick = self.rng.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if acc >= pick:
                return i
        return len(weights) - 1

    def tabu_search(self, colors: List[int], time_limit: float) -> Tuple[List[int], int]:
        graph = self.graph
        n = graph.n
        k = self.k
        start = time.perf_counter()

        cur = list(colors)
        neighbor_color_count = [[0] * k for _ in range(n)]
        for v in range(n):
            for nb in graph.adj[v]:
                neighbor_color_count[v][cur[nb]] += 1

        cur_conflicts = sum(neighbor_color_count[v][cur[v]] for v in range(n)) // 2
        best = list(cur)
        best_conflicts = cur_conflicts

        tabu_until = [[0] * k for _ in range(n)]
        iteration = 0
        stagnation = 0

        while time.perf_counter() - start < time_limit:
            if cur_conflicts == 0:
                return cur, 0

            move_v = -1
            move_color = -1
            move_delta = math.inf
            move_new_conflicts = math.inf

            conflict_vertices = [v for v in range(n) if neighbor_color_count[v][cur[v]] > 0]
            if not conflict_vertices:
                return cur, 0

            self.rng.shuffle(conflict_vertices)
            sample_size = min(len(conflict_vertices), 64)
            candidates = conflict_vertices[:sample_size]

            for v in candidates:
                old_color = cur[v]
                old_same = neighbor_color_count[v][old_color]
                for new_color in range(k):
                    if new_color == old_color:
                        continue
                    new_same = neighbor_color_count[v][new_color]
                    delta = new_same - old_same
                    new_conflicts = cur_conflicts + delta
                    tabu = tabu_until[v][new_color] > iteration
                    aspiration = new_conflicts < best_conflicts
                    if tabu and not aspiration:
                        continue
                    if (delta < move_delta) or (delta == move_delta and new_conflicts < move_new_conflicts):
                        move_delta = delta
                        move_new_conflicts = new_conflicts
                        move_v = v
                        move_color = new_color

            if move_v == -1:
                v = self.rng.choice(conflict_vertices)
                old_color = cur[v]
                choices = [c for c in range(k) if c != old_color]
                move_color = self.rng.choice(choices)
                move_v = v
                move_delta = neighbor_color_count[v][move_color] - neighbor_color_count[v][old_color]
                move_new_conflicts = cur_conflicts + move_delta

            v = move_v
            old_color = cur[v]
            new_color = move_color
            cur[v] = new_color
            cur_conflicts = move_new_conflicts

            for nb in graph.adj[v]:
                neighbor_color_count[nb][old_color] -= 1
                neighbor_color_count[nb][new_color] += 1
            row = [0] * k
            for nb in graph.adj[v]:
                row[cur[nb]] += 1
            neighbor_color_count[v] = row

            tenure = self.base_tabu_tenure + self.rng.randrange(0, 7)
            tabu_until[v][old_color] = iteration + tenure
            iteration += 1

            if cur_conflicts < best_conflicts:
                best_conflicts = cur_conflicts
                best = list(cur)
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= self.max_stagnation_moves:
                self.perturb(cur, neighbor_color_count)
                cur_conflicts = sum(neighbor_color_count[x][cur[x]] for x in range(n)) // 2
                stagnation = 0

        return best, best_conflicts

    def perturb(self, colors: List[int], neighbor_color_count: List[List[int]]) -> None:
        graph = self.graph
        conflict_vertices = [v for v in range(graph.n) if neighbor_color_count[v][colors[v]] > 0]
        if not conflict_vertices:
            return
        self.rng.shuffle(conflict_vertices)
        changes = max(1, min(len(conflict_vertices), graph.n // 20 + 1))
        for v in conflict_vertices[:changes]:
            old_color = colors[v]
            best_color = old_color
            best_same = neighbor_color_count[v][old_color]
            for c in range(self.k):
                same = neighbor_color_count[v][c]
                if same < best_same or (same == best_same and self.rng.random() < 0.25):
                    best_same = same
                    best_color = c
            if best_color == old_color and self.k > 1:
                choices = [c for c in range(self.k) if c != old_color]
                best_color = self.rng.choice(choices)
            if best_color != old_color:
                colors[v] = best_color
                for nb in graph.adj[v]:
                    neighbor_color_count[nb][old_color] -= 1
                    neighbor_color_count[nb][best_color] += 1
                row = [0] * self.k
                for nb in graph.adj[v]:
                    row[colors[nb]] += 1
                neighbor_color_count[v] = row

    def update_pheromone(
        self,
        iter_best_colors: List[int],
        iter_best_conflicts: int,
        global_best_colors: List[int],
        global_best_conflicts: int,
    ) -> None:
        n = self.graph.n
        k = self.k
        evap = 1.0 - self.evaporation
        for v in range(n):
            row = self.pheromone[v]
            for c in range(k):
                row[c] = max(1e-6, row[c] * evap)

        iter_gain = self.q / (1.0 + iter_best_conflicts)
        for v, c in enumerate(iter_best_colors):
            self.pheromone[v][c] += iter_gain

        if global_best_conflicts <= iter_best_conflicts:
            global_gain = 2.0 * self.q / (1.0 + global_best_conflicts)
            for v, c in enumerate(global_best_colors):
                self.pheromone[v][c] += global_gain


def reduce_seed_to_k(seed_colors: List[int], k: int, rng: random.Random) -> List[int]:
    if not seed_colors:
        return []
    present = sorted(set(seed_colors))
    if len(present) <= k:
        mapping = {c: i for i, c in enumerate(present)}
        out = [mapping[c] for c in seed_colors]
        if len(present) < k:
            out = [c % k for c in out]
        return out

    removable = present[k:]
    keep = present[:k]
    out = list(seed_colors)
    for idx, c in enumerate(out):
        if c in removable:
            out[idx] = rng.choice(keep)
    remap = {c: i for i, c in enumerate(sorted(set(out)))}
    out = [remap[c] for c in out]
    return [c % k for c in out]


def hybrid_solve(
    path: str,
    time_per_k: float = 10.0,
    ants: int = 12,
    seed: int = 0,
    max_no_improve_k: int = 1,
) -> ColoringResult:
    rng = random.Random(seed)
    graph = Graph.from_dimacs_col(path)

    ub, dsatur_colors = dsatur_upper_bound(graph)
    lb = greedy_clique_lower_bound(graph)

    print(f"Graph: n={graph.n}, m={len(graph.edges)}")
    print(f"Initial DSATUR upper bound: {ub}")
    print(f"Greedy clique lower bound: {lb}")

    best_valid_colors = list(dsatur_colors)
    best_valid_k = ub

    k = ub
    failures = 0
    while k >= lb:
        print(f"\nTrying k = {k}")
        solver = FixedKHybridACO(
            graph=graph,
            k=k,
            rng=rng,
            ants=ants,
            alpha=1.0,
            beta=2.0,
            evaporation=0.20,
            q=6.0,
            tabu_tenure=max(7, min(20, graph.n // 25 + 5)),
            max_stagnation_moves=max(500, min(4000, graph.n * 3)),
        )
        seed_colors = reduce_seed_to_k(best_valid_colors, k, rng)
        result = solver.search(time_limit=time_per_k, seed_colors=seed_colors)
        checked = verify_coloring(graph, result.colors, k)
        print(f"Best conflicts for k={k}: {result.conflicts}")
        if checked != result.conflicts:
            raise RuntimeError("Internal error: conflict count mismatch")

        if result.conflicts == 0:
            print(f"FOUND feasible coloring with k = {k}")
            best_valid_colors = list(result.colors)
            best_valid_k = k
            k -= 1
            failures = 0
        else:
            failures += 1
            print(f"FAILED for k = {k}")
            if failures >= max_no_improve_k:
                break
            k -= 1

    total_conflicts = verify_coloring(graph, best_valid_colors, best_valid_k)
    if total_conflicts != 0:
        raise RuntimeError("Internal error: returned best coloring is not feasible")
    return ColoringResult(best_valid_colors, total_conflicts, best_valid_k, 0.0)


def write_solution(path: str, result: ColoringResult) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"c k = {result.k}\n")
        for idx, color in enumerate(result.colors, start=1):
            f.write(f"v {idx} {color + 1}\n")


def run_self_test() -> None:
    graph = Graph(
        n=5,
        edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
        adj=[[1, 4], [0, 2], [1, 3], [2, 4], [3, 0]],
        degree=[2, 2, 2, 2, 2],
    )
    ub, colors = dsatur_upper_bound(graph)
    assert ub == 3
    assert verify_coloring(graph, colors, ub) == 0

    rng = random.Random(123)
    solver = FixedKHybridACO(graph, 3, rng, ants=8)
    result = solver.search(time_limit=1.0, seed_colors=colors)
    assert verify_coloring(graph, result.colors, 3) == 0

    bip = Graph(
        n=4,
        edges=[(0, 2), (0, 3), (1, 2), (1, 3)],
        adj=[[2, 3], [2, 3], [0, 1], [0, 1]],
        degree=[2, 2, 2, 2],
    )
    solver2 = FixedKHybridACO(bip, 2, rng, ants=8)
    result2 = solver2.search(time_limit=1.5)
    assert verify_coloring(bip, result2.colors, 2) == 0
    print("Self-test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid ACO + tabu search graph coloring solver for DIMACS .col files")
    parser.add_argument("path", nargs="?", help="Path to DIMACS .col file, for example data/raw/DSJC125.9.col")
    parser.add_argument("--time-per-k", type=float, default=10.0, help="Time budget in seconds for each k")
    parser.add_argument("--ants", type=int, default=12, help="Number of ants per ACO iteration")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", type=str, default="", help="Optional output solution file")
    parser.add_argument("--self-test", action="store_true", help="Run a small internal test and exit")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not args.path:
        parser.error("You must provide a DIMACS .col file path.")

    t0 = time.perf_counter()
    result = hybrid_solve(
        path=args.path,
        time_per_k=args.time_per_k,
        ants=args.ants,
        seed=args.seed,
        max_no_improve_k=1,
    )
    elapsed = time.perf_counter() - t0

    print("\nFinal best feasible coloring")
    print(f"k = {result.k}")
    print(f"verification conflicts = {verify_coloring(Graph.from_dimacs_col(args.path), result.colors, result.k)}")
    print(f"total runtime = {elapsed:.3f} s")

    if args.output:
        write_solution(args.output, result)
        print(f"Solution written to: {args.output}")


if __name__ == "__main__":
    main()
