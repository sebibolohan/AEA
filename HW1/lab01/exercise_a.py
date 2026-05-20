from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ortools.sat.python import cp_model
from sklearn import base


Color = int
Country = str
Edge = Tuple[Country, Country]


@dataclass(frozen=True)
class MapInstance:
    countries: List[Country]
    edges: List[Edge]


def _normalize_edges(edges: List[Edge]) -> List[Edge]:
    # ensure consistent ordering (a,b) with a < b, and remove duplicates
    norm = set()
    for a, b in edges:
        if a == b:
            raise ValueError("Self-loop edge is not allowed.")
        x, y = (a, b) if a < b else (b, a)
        norm.add((x, y))
    return sorted(norm)


def build_base_instance() -> MapInstance:
    # Countries: Belgium, Denmark, France, Germany, Luxembourg, Netherlands
    countries = ["Belgium", "Denmark", "France", "Germany", "Luxembourg", "Netherlands"]

    # Neighbor relations for the given 6-country instance
    # (common borders in this simplified map model):
    edges = _normalize_edges(
        [
            ("Belgium", "France"),
            ("Belgium", "Germany"),
            ("Belgium", "Luxembourg"),
            ("Belgium", "Netherlands"),
            ("Denmark", "Germany"),
            ("France", "Germany"),
            ("France", "Luxembourg"),
            ("Germany", "Luxembourg"),
            ("Germany", "Netherlands"),
        ]
    )

    return MapInstance(countries=countries, edges=edges)


def build_with_switzerland(base: MapInstance) -> MapInstance:
    if "Switzerland" in base.countries:
        raise ValueError("Switzerland already present.")

    countries = base.countries + ["Switzerland"]
    edges = _normalize_edges(base.edges + [("Switzerland", "France"), ("Switzerland", "Germany")])
    return MapInstance(countries=countries, edges=edges)

def print_neighbors(instance):
    neighbors = {c: [] for c in instance.countries}

    for a, b in instance.edges:
        neighbors[a].append(b)
        neighbors[b].append(a)

    print("\nNeighbor relations:")
    for c in instance.countries:
        print(f"{c}: {', '.join(neighbors[c])}")

def solve_map_coloring(
    instance: MapInstance,
    num_colors: int = 4,
    force_equal: Tuple[Country, Country] | None = None,
) -> Dict[Country, Color]:
    if num_colors <= 0:
        raise ValueError("num_colors must be positive.")

    model = cp_model.CpModel()
    color_names = ["blue", "white", "yellow", "green"]
    if num_colors > len(color_names):
        # keep it safe and explicit; extend if you ever change num_colors
        raise ValueError("num_colors exceeds available color_names list.")

    # Decision variables: one per country
    vars_: Dict[Country, cp_model.IntVar] = {
        c: model.NewIntVar(0, num_colors - 1, c) for c in instance.countries
    }

    # Optional constraint: force two countries same color (e.g., Germany == Denmark)
    allowed_equal_pair: Edge | None = None
    if force_equal is not None:
        a, b = force_equal
        if a not in vars_ or b not in vars_:
            raise ValueError("force_equal countries must exist in instance.")
        model.Add(vars_[a] == vars_[b])
        allowed_equal_pair = (a, b) if a < b else (b, a)

    # Constraints: neighboring countries must differ
    for a, b in instance.edges:
        # If we forced equality for this specific pair, we skip the inequality constraint for it
        if allowed_equal_pair is not None and (a, b) == allowed_equal_pair:
            continue
        model.Add(vars_[a] != vars_[b])

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No solution found for the given instance and constraints.")

    sol: Dict[Country, Color] = {c: int(solver.Value(v)) for c, v in vars_.items()}

    # Small sanity checks
    if force_equal is not None:
        a, b = force_equal
        if sol[a] != sol[b]:
            raise AssertionError("force_equal constraint violated in solution (unexpected).")
    for a, b in instance.edges:
        if allowed_equal_pair is not None and (a, b) == allowed_equal_pair:
            continue
        if sol[a] == sol[b]:
            raise AssertionError(f"Adjacency constraint violated: {a} == {b}")

    # Print solution
    print("\nSolution:")
    for c in instance.countries:
        print(f"{c}: {color_names[sol[c]]}")
    return sol


def main() -> None:
    print("A(a).1 Base instance (6 countries)")
    base = build_base_instance()

    print_neighbors(base)    
    
    solve_map_coloring(base, num_colors=4, force_equal=None)

    print("\nA(a).2 Germany and Denmark always same color")
    solve_map_coloring(base, num_colors=4, force_equal=("Germany", "Denmark"))

    print("\nA(a).3 Add Switzerland (neighbors: France, Germany) + keep Germany==Denmark")
    with_ch = build_with_switzerland(base)
    solve_map_coloring(with_ch, num_colors=4, force_equal=("Germany", "Denmark"))


if __name__ == "__main__":
    main()