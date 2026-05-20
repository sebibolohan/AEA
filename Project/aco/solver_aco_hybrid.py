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
        edges = []

        with open(path, "r", encoding="utf-8") as f:
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
class ColoringResult:
    colors: List[int]
    conflicts: int
    k: int
    time_seconds: float


def verify_coloring(graph: Graph, colors: List[int], k: int) -> int:
    if len(colors) != graph.n:
        raise ValueError("Coloring length mismatch")

    for color in colors:
        if color < 0 or color >= k:
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

    first = max(range(n), key=lambda v: graph.degree[v])
    colors[first] = 0
    uncolored.remove(first)

    for nb in graph.adj[first]:
        neighbor_colors[nb].add(0)

    used_colors = 1

    while uncolored:
        v = max(uncolored, key=lambda x: (len(neighbor_colors[x]), graph.degree[x], x))

        color = 0
        while color in neighbor_colors[v]:
            color += 1

        colors[v] = color
        used_colors = max(used_colors, color + 1)
        uncolored.remove(v)

        for nb in graph.adj[v]:
            if colors[nb] == -1:
                neighbor_colors[nb].add(color)

    return used_colors, colors


def greedy_clique_lower_bound(graph: Graph) -> int:
    order = sorted(range(graph.n), key=lambda v: graph.degree[v], reverse=True)
    clique = []

    for v in order:
        is_compatible = True
        adj_v = set(graph.adj[v])

        for u in clique:
            if u not in adj_v:
                is_compatible = False
                break

        if is_compatible:
            clique.append(v)

    return max(1, len(clique))


def random_k_coloring(n: int, k: int, rng: random.Random) -> List[int]:
    return [rng.randrange(k) for _ in range(n)]


def reduce_seed_to_k(seed_colors: List[int], k: int, rng: random.Random) -> List[int]:
    present = sorted(set(seed_colors))

    if len(present) <= k:
        mapping = {color: i for i, color in enumerate(present)}
        return [mapping[color] % k for color in seed_colors]

    frequency = {}
    for color in seed_colors:
        frequency[color] = frequency.get(color, 0) + 1

    kept_colors = sorted(present, key=lambda c: frequency[c], reverse=True)[:k]
    kept_set = set(kept_colors)
    mapping = {color: i for i, color in enumerate(kept_colors)}

    reduced = []

    for color in seed_colors:
        if color in kept_set:
            reduced.append(mapping[color])
        else:
            reduced.append(rng.randrange(k))

    return reduced


class HybridACOColoring:
    def __init__(
        self,
        graph: Graph,
        k: int,
        rng: random.Random,
        ants: int = 12,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation: float = 0.20,
        deposit_q: float = 6.0,
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
        self.deposit_q = deposit_q
        self.tabu_tenure = max(1, tabu_tenure)
        self.max_stagnation_moves = max(100, max_stagnation_moves)
        self.pheromone = [[1.0 for _ in range(k)] for _ in range(graph.n)]

    def search(self, time_limit: float, seed_colors: Optional[List[int]] = None) -> ColoringResult:
        start = time.perf_counter()

        best_colors = None
        best_conflicts = math.inf

        if seed_colors is not None:
            seed = [color % self.k for color in seed_colors]
            improved, conflicts = self.tabu_search(seed, min(0.5, max(0.05, time_limit * 0.10)))

            best_colors = improved
            best_conflicts = conflicts

            if best_conflicts == 0:
                return ColoringResult(best_colors, 0, self.k, time.perf_counter() - start)

        while time.perf_counter() - start < time_limit:
            iteration_best_colors = None
            iteration_best_conflicts = math.inf

            for _ in range(self.ants):
                remaining = time_limit - (time.perf_counter() - start)

                if remaining <= 0:
                    break

                constructed = self.construct_coloring()
                local_time = min(0.50, max(0.03, remaining / self.ants))

                improved, conflicts = self.tabu_search(constructed, local_time)

                if conflicts < iteration_best_conflicts:
                    iteration_best_conflicts = conflicts
                    iteration_best_colors = improved

                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_colors = improved

                    if best_conflicts == 0:
                        return ColoringResult(best_colors, 0, self.k, time.perf_counter() - start)

            if iteration_best_colors is None:
                break

            self.update_pheromone(
                iteration_best_colors,
                iteration_best_conflicts,
                best_colors,
                best_conflicts,
            )

        if best_colors is None:
            best_colors = random_k_coloring(self.graph.n, self.k, self.rng)
            best_conflicts = verify_coloring(self.graph, best_colors, self.k)

        return ColoringResult(best_colors, best_conflicts, self.k, time.perf_counter() - start)

    def construct_coloring(self) -> List[int]:
        n = self.graph.n
        colors = [-1] * n
        saturation_sets = [set() for _ in range(n)]
        uncolored = set(range(n))

        while uncolored:
            v = max(uncolored, key=lambda x: (len(saturation_sets[x]), self.graph.degree[x], x))

            weights = []

            for color in range(self.k):
                local_conflicts = 0

                for nb in self.graph.adj[v]:
                    if colors[nb] == color:
                        local_conflicts += 1

                tau = self.pheromone[v][color] ** self.alpha
                eta = (1.0 / (1.0 + local_conflicts)) ** self.beta

                weights.append(max(1e-12, tau * eta))

            chosen_color = self.weighted_choice(weights)

            colors[v] = chosen_color
            uncolored.remove(v)

            for nb in self.graph.adj[v]:
                if colors[nb] == -1:
                    saturation_sets[nb].add(chosen_color)

        return colors

    def weighted_choice(self, weights: List[float]) -> int:
        total = sum(weights)

        if total <= 0:
            return self.rng.randrange(len(weights))

        pick = self.rng.random() * total
        cumulative = 0.0

        for index, weight in enumerate(weights):
            cumulative += weight

            if cumulative >= pick:
                return index

        return len(weights) - 1

    def tabu_search(self, colors: List[int], time_limit: float) -> Tuple[List[int], int]:
        graph = self.graph
        n = graph.n
        k = self.k

        start = time.perf_counter()

        current = list(colors)

        neighbor_color_count = [[0] * k for _ in range(n)]

        for v in range(n):
            for nb in graph.adj[v]:
                neighbor_color_count[v][current[nb]] += 1

        current_conflicts = sum(neighbor_color_count[v][current[v]] for v in range(n)) // 2

        best = list(current)
        best_conflicts = current_conflicts

        tabu_until = [[0] * k for _ in range(n)]

        iteration = 0
        stagnation = 0

        while time.perf_counter() - start < time_limit:
            if current_conflicts == 0:
                return current, 0

            conflict_vertices = [
                v for v in range(n)
                if neighbor_color_count[v][current[v]] > 0
            ]

            if not conflict_vertices:
                return current, 0

            self.rng.shuffle(conflict_vertices)
            candidates = conflict_vertices[:min(64, len(conflict_vertices))]

            best_move_v = -1
            best_move_color = -1
            best_move_delta = math.inf
            best_move_conflicts = math.inf

            for v in candidates:
                old_color = current[v]
                old_same = neighbor_color_count[v][old_color]

                for new_color in range(k):
                    if new_color == old_color:
                        continue

                    new_same = neighbor_color_count[v][new_color]
                    delta = new_same - old_same
                    new_conflicts = current_conflicts + delta

                    is_tabu = tabu_until[v][new_color] > iteration
                    aspiration = new_conflicts < best_conflicts

                    if is_tabu and not aspiration:
                        continue

                    if delta < best_move_delta:
                        best_move_delta = delta
                        best_move_conflicts = new_conflicts
                        best_move_v = v
                        best_move_color = new_color
                    elif delta == best_move_delta and new_conflicts < best_move_conflicts:
                        best_move_conflicts = new_conflicts
                        best_move_v = v
                        best_move_color = new_color

            if best_move_v == -1:
                v = self.rng.choice(conflict_vertices)
                old_color = current[v]
                possible_colors = [color for color in range(k) if color != old_color]

                best_move_v = v
                best_move_color = self.rng.choice(possible_colors)
                best_move_delta = (
                    neighbor_color_count[v][best_move_color]
                    - neighbor_color_count[v][old_color]
                )
                best_move_conflicts = current_conflicts + best_move_delta

            v = best_move_v
            old_color = current[v]
            new_color = best_move_color

            current[v] = new_color
            current_conflicts = best_move_conflicts

            for nb in graph.adj[v]:
                neighbor_color_count[nb][old_color] -= 1
                neighbor_color_count[nb][new_color] += 1

            row = [0] * k
            for nb in graph.adj[v]:
                row[current[nb]] += 1

            neighbor_color_count[v] = row

            tenure = self.tabu_tenure + self.rng.randrange(0, 7)
            tabu_until[v][old_color] = iteration + tenure

            iteration += 1

            if current_conflicts < best_conflicts:
                best_conflicts = current_conflicts
                best = list(current)
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= self.max_stagnation_moves:
                self.perturb(current, neighbor_color_count)
                current_conflicts = sum(
                    neighbor_color_count[x][current[x]]
                    for x in range(n)
                ) // 2
                stagnation = 0

        return best, best_conflicts

    def perturb(self, colors: List[int], neighbor_color_count: List[List[int]]) -> None:
        graph = self.graph

        conflict_vertices = [
            v for v in range(graph.n)
            if neighbor_color_count[v][colors[v]] > 0
        ]

        if not conflict_vertices:
            return

        self.rng.shuffle(conflict_vertices)

        changes = max(1, min(len(conflict_vertices), graph.n // 20 + 1))

        for v in conflict_vertices[:changes]:
            old_color = colors[v]
            best_color = old_color
            best_same = neighbor_color_count[v][old_color]

            for color in range(self.k):
                same = neighbor_color_count[v][color]

                if same < best_same:
                    best_same = same
                    best_color = color
                elif same == best_same and self.rng.random() < 0.25:
                    best_color = color

            if best_color == old_color and self.k > 1:
                candidates = [color for color in range(self.k) if color != old_color]
                best_color = self.rng.choice(candidates)

            if best_color == old_color:
                continue

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
        iteration_best_colors: List[int],
        iteration_best_conflicts: int,
        global_best_colors: List[int],
        global_best_conflicts: int,
    ) -> None:
        evaporation_factor = 1.0 - self.evaporation

        for v in range(self.graph.n):
            for color in range(self.k):
                self.pheromone[v][color] *= evaporation_factor
                self.pheromone[v][color] = max(1e-6, self.pheromone[v][color])

        iteration_gain = self.deposit_q / (1.0 + iteration_best_conflicts)

        for v, color in enumerate(iteration_best_colors):
            self.pheromone[v][color] += iteration_gain

        if global_best_colors is not None:
            global_gain = 2.0 * self.deposit_q / (1.0 + global_best_conflicts)

            for v, color in enumerate(global_best_colors):
                self.pheromone[v][color] += global_gain


def hybrid_solve(
    path: str,
    time_per_k: float,
    ants: int,
    seed: int,
    max_failed_k: int,
) -> ColoringResult:
    rng = random.Random(seed)
    graph = Graph.from_dimacs_col(path)

    upper_bound, dsatur_colors = dsatur_upper_bound(graph)
    lower_bound = greedy_clique_lower_bound(graph)

    print("Graph: n={}, m={}".format(graph.n, len(graph.edges)))
    print("Initial DSATUR upper bound: {}".format(upper_bound))
    print("Greedy clique lower bound: {}".format(lower_bound))

    best_valid_colors = list(dsatur_colors)
    best_valid_k = upper_bound

    k = upper_bound
    failed_attempts = 0

    while k >= lower_bound:
        print("")
        print("Trying k = {}".format(k))

        solver = HybridACOColoring(
            graph=graph,
            k=k,
            rng=rng,
            ants=ants,
            alpha=1.0,
            beta=2.0,
            evaporation=0.20,
            deposit_q=6.0,
            tabu_tenure=max(7, min(20, graph.n // 25 + 5)),
            max_stagnation_moves=max(500, min(4000, graph.n * 3)),
        )

        seed_colors = reduce_seed_to_k(best_valid_colors, k, rng)

        result = solver.search(
            time_limit=time_per_k,
            seed_colors=seed_colors,
        )

        checked_conflicts = verify_coloring(graph, result.colors, k)

        if checked_conflicts != result.conflicts:
            raise RuntimeError("Internal error: conflict mismatch")

        print("Best conflicts for k={}: {}".format(k, result.conflicts))

        if result.conflicts == 0:
            print("FOUND feasible coloring with k = {}".format(k))

            best_valid_colors = list(result.colors)
            best_valid_k = k

            k -= 1
            failed_attempts = 0
        else:
            print("FAILED for k = {}".format(k))

            failed_attempts += 1

            if failed_attempts >= max_failed_k:
                break

            k -= 1

    final_conflicts = verify_coloring(graph, best_valid_colors, best_valid_k)

    if final_conflicts != 0:
        raise RuntimeError("Returned coloring is not feasible")

    return ColoringResult(
        colors=best_valid_colors,
        conflicts=final_conflicts,
        k=best_valid_k,
        time_seconds=0.0,
    )


def write_solution(path: str, result: ColoringResult) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("c k = {}\n".format(result.k))

        for index, color in enumerate(result.colors, start=1):
            f.write("v {} {}\n".format(index, color + 1))


def run_self_test() -> None:
    cycle_5 = Graph(
        n=5,
        edges=[(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)],
        adj=[[1, 4], [0, 2], [1, 3], [2, 4], [3, 0]],
        degree=[2, 2, 2, 2, 2],
    )

    ub, colors = dsatur_upper_bound(cycle_5)

    assert ub == 3
    assert verify_coloring(cycle_5, colors, ub) == 0

    rng = random.Random(123)

    solver = HybridACOColoring(cycle_5, 3, rng, ants=8)
    result = solver.search(time_limit=1.0, seed_colors=colors)

    assert verify_coloring(cycle_5, result.colors, 3) == 0

    bipartite = Graph(
        n=4,
        edges=[(0, 2), (0, 3), (1, 2), (1, 3)],
        adj=[[2, 3], [2, 3], [0, 1], [0, 1]],
        degree=[2, 2, 2, 2],
    )

    solver_2 = HybridACOColoring(bipartite, 2, rng, ants=8)
    result_2 = solver_2.search(time_limit=1.0)

    assert verify_coloring(bipartite, result_2.colors, 2) == 0

    print("Self-test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid ACO + tabu search solver for graph coloring on DIMACS .col files"
    )

    parser.add_argument(
        "path",
        nargs="?",
        help="Path to DIMACS .col file, example: data/raw/DSJC125.9.col",
    )

    parser.add_argument(
        "--time-per-k",
        type=float,
        default=10.0,
        help="Time budget in seconds for each k",
    )

    parser.add_argument(
        "--ants",
        type=int,
        default=12,
        help="Number of ants per ACO iteration",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )

    parser.add_argument(
        "--max-failed-k",
        type=int,
        default=1,
        help="Stop after this many failed consecutive k values",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output solution file",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run internal tests and exit",
    )

    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not args.path:
        parser.error("You must provide a DIMACS .col file path.")

    start = time.perf_counter()

    result = hybrid_solve(
        path=args.path,
        time_per_k=args.time_per_k,
        ants=args.ants,
        seed=args.seed,
        max_failed_k=args.max_failed_k,
    )

    elapsed = time.perf_counter() - start

    graph = Graph.from_dimacs_col(args.path)
    verification_conflicts = verify_coloring(graph, result.colors, result.k)

    print("")
    print("Final best feasible coloring")
    print("k = {}".format(result.k))
    print("verification conflicts = {}".format(verification_conflicts))
    print("total runtime = {:.3f} s".format(elapsed))

    if args.output:
        write_solution(args.output, result)
        print("Solution written to: {}".format(args.output))


if __name__ == "__main__":
    main()