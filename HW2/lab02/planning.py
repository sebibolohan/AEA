from dataclasses import dataclass

from ortools.linear_solver import pywraplp


@dataclass(frozen=True)
class ProductionPlanResult:
    gas: float
    chloride: float
    profit: float


def build_solver() -> tuple[pywraplp.Solver, pywraplp.Variable, pywraplp.Variable]:
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        raise RuntimeError("OR-Tools GLOP solver is not available.")

    gas = solver.NumVar(0.0, solver.infinity(), "Gas")
    chloride = solver.NumVar(0.0, solver.infinity(), "Chloride")

    solver.Add(gas + chloride <= 50, "nitrogen_limit")
    solver.Add(3 * gas + 4 * chloride <= 180, "hydrogen_limit")
    solver.Add(chloride <= 40, "chlorine_limit")

    solver.Maximize(40 * gas + 50 * chloride)

    return solver, gas, chloride


def solve_production_planning() -> ProductionPlanResult:
    solver, gas, chloride = build_solver()
    status = solver.Solve()

    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("Optimal solution was not found.")

    return ProductionPlanResult(
        gas=gas.solution_value(),
        chloride=chloride.solution_value(),
        profit=solver.Objective().Value(),
    )


def main() -> None:
    result = solve_production_planning()

    print("Optimal solution found")
    print(f"Gas       = {result.gas:.2f}")
    print(f"Chloride  = {result.chloride:.2f}")
    print(f"Profit    = {result.profit:.2f}")


if __name__ == "__main__":
    main()