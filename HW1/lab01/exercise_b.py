from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from ortools.sat.python import cp_model

Pos = Tuple[int, int]  # (row, col)


@dataclass(frozen=True)
class NQueensResult:
    n: int
    queens: List[int]  # queens[col] = row


def _validate_blocked(n: int, blocked: Sequence[Pos]) -> List[Pos]:
    seen = set()
    out: List[Pos] = []
    for r, c in blocked:
        if not (0 <= r < n and 0 <= c < n):
            raise ValueError(f"Blocked position out of range: {(r, c)} for n={n}.")
        if (r, c) not in seen:
            seen.add((r, c))
            out.append((r, c))
    return out


def print_blocked(n: int, blocked: Sequence[Pos]) -> None:
    if not blocked:
        print("Blocked positions: (none)")
        return
    print("Blocked positions:")
    for r, c in blocked:
        print(f"  (row={r}, col={c})")


def print_constraints_summary(n: int, blocked: Sequence[Pos]) -> None:
    # A short, useful summary, not dumping every pairwise diagonal constraint.
    print("\nConstraints:")
    print("- Exactly one queen per column (by variable definition).")
    print("- AllDifferent on rows (no two queens share a row).")
    print("- AllDifferent on main diagonals (q[c] + c).")
    print("- AllDifferent on anti-diagonals (q[c] - c).")
    if blocked:
        print(f"- Blocked cells: {len(blocked)} constraints of form q[col] != row.")


def print_board(n: int, queens: Sequence[int], blocked: Sequence[Pos]) -> None:
    blocked_set = set(blocked)
    print("\nBoard (Q=queen, X=blocked, .=empty):")
    for r in range(n):
        row_cells = []
        for c in range(n):
            if (r, c) in blocked_set:
                row_cells.append("X")
            elif queens[c] == r:
                row_cells.append("Q")
            else:
                row_cells.append(".")
        print(" ".join(row_cells))


def solve_n_queens(n: int, blocked: Sequence[Pos] = ()) -> Optional[NQueensResult]:
    if n <= 0:
        raise ValueError("n must be positive.")
    blocked = _validate_blocked(n, blocked)

    model = cp_model.CpModel()

    # Decision variables: one per column, value is the row index.
    q = [model.NewIntVar(0, n - 1, f"q{c}") for c in range(n)]

    # No two queens share a row.
    model.AddAllDifferent(q)

    # No two queens share a diagonal:
    # main diagonal id: q[c] + c
    # anti diagonal id: q[c] - c
    diag_main = [model.NewIntVar(0, 2 * n - 2, f"dmain{c}") for c in range(n)]
    diag_anti = [model.NewIntVar(-(n - 1), n - 1, f"danti{c}") for c in range(n)]

    for c in range(n):
        model.Add(diag_main[c] == q[c] + c)
        model.Add(diag_anti[c] == q[c] - c)

    model.AddAllDifferent(diag_main)
    model.AddAllDifferent(diag_anti)

    # Blocked positions: queen cannot be placed there.
    for r, c in blocked:
        model.Add(q[c] != r)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    queens = [int(solver.Value(v)) for v in q]

    # Sanity checks (fail fast if something is wrong)
    if len(set(queens)) != n:
        raise AssertionError("Row uniqueness violated.")
    if len(set(queens[c] + c for c in range(n))) != n:
        raise AssertionError("Main diagonal uniqueness violated.")
    if len(set(queens[c] - c for c in range(n))) != n:
        raise AssertionError("Anti-diagonal uniqueness violated.")
    blocked_set = set(blocked)
    for c in range(n):
        if (queens[c], c) in blocked_set:
            raise AssertionError("Blocked constraint violated.")

    return NQueensResult(n=n, queens=queens)


def run_instance(n: int, blocked: Sequence[Pos]) -> None:
    print(f"\nB. n-queens instance: n={n}")
    blocked = _validate_blocked(n, blocked)
    print_blocked(n, blocked)
    print_constraints_summary(n, blocked)

    res = solve_n_queens(n, blocked)
    if res is None:
        print("\nResult: UNSAT (no solution exists).")
        return

    print("\nResult: SAT (one solution found).")
    # Print as (col -> row), similar to "variables + values"
    print("Queens (col -> row):")
    for c, r in enumerate(res.queens):
        print(f"  col {c} -> row {r}")
    print_board(n, res.queens, blocked)


def main() -> None:
    # Required: solve a 4x4 instance
    run_instance(4, blocked=[])

    # Required: consider blocked positions and test on instances
    # Test 1: a small blocked set that still usually remains solvable
    run_instance(4, blocked=[(1, 1)])

    # Test 2: block one of the two known 4x4 solutions completely -> UNSAT
    # 4x4 solutions are [1,3,0,2] and [2,0,3,1] (col -> row).
    # Blocking both placements for col0 (row 1 and row 2) forces UNSAT.
    run_instance(4, blocked=[(1, 0), (2, 0)])

    # Test 3: block one of the two known 4x4 solutions partially -> still SAT, but only one solution remains.
    run_instance(4, blocked=[(1, 0)])
    




if __name__ == "__main__":
    main()