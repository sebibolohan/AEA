from dataclasses import dataclass
from ortools.linear_solver import pywraplp


@dataclass(frozen=True)
class Item:
    name: str
    weight: int
    value: int


@dataclass(frozen=True)
class MultiKnapsackInstance:
    name: str
    items: list[Item]
    capacities: list[int]


@dataclass(frozen=True)
class Assignment:
    item_name: str
    knapsack_index: int
    weight: int
    value: int


@dataclass(frozen=True)
class MultiKnapsackResult:
    total_value: int
    used_weights: list[int]
    assignments: list[Assignment]


def solve_multi_knapsack(instance: MultiKnapsackInstance) -> MultiKnapsackResult:
    solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    if solver is None:
        raise RuntimeError("OR-Tools CBC solver is not available.")

    num_items = len(instance.items)
    num_knapsacks = len(instance.capacities)

    # x[i, j] = 1 if item i is placed in knapsack j, 0 otherwise
    x: dict[tuple[int, int], pywraplp.Variable] = {}
    for i in range(num_items):
        for j in range(num_knapsacks):
            x[i, j] = solver.BoolVar(f"x_{i}_{j}")

    # Each item can be placed in at most one knapsack
    for i in range(num_items):
        solver.Add(sum(x[i, j] for j in range(num_knapsacks)) <= 1)

    # Capacity constraint for each knapsack
    for j in range(num_knapsacks):
        solver.Add(
            sum(instance.items[i].weight * x[i, j] for i in range(num_items))
            <= instance.capacities[j]
        )

    # Maximize total value
    solver.Maximize(
        sum(
            instance.items[i].value * x[i, j]
            for i in range(num_items)
            for j in range(num_knapsacks)
        )
    )

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("Optimal solution was not found.")

    assignments: list[Assignment] = []
    used_weights = [0] * num_knapsacks

    for i in range(num_items):
        for j in range(num_knapsacks):
            if x[i, j].solution_value() > 0.5:
                item = instance.items[i]
                assignments.append(
                    Assignment(
                        item_name=item.name,
                        knapsack_index=j,
                        weight=item.weight,
                        value=item.value,
                    )
                )
                used_weights[j] += item.weight

    return MultiKnapsackResult(
        total_value=int(round(solver.Objective().Value())),
        used_weights=used_weights,
        assignments=assignments,
    )


def validate_result(instance: MultiKnapsackInstance, result: MultiKnapsackResult) -> None:
    assigned_items: set[str] = set()

    for assignment in result.assignments:
        if assignment.item_name in assigned_items:
            raise AssertionError(f"Item assigned more than once: {assignment.item_name}")
        assigned_items.add(assignment.item_name)

    for used, capacity in zip(result.used_weights, instance.capacities):
        if used > capacity:
            raise AssertionError(
                f"Capacity exceeded: used={used}, capacity={capacity}"
            )


def print_result(instance: MultiKnapsackInstance, result: MultiKnapsackResult) -> None:
    print(f"\nInstance: {instance.name}")
    print(f"Capacities: {instance.capacities}")
    print(f"Total value: {result.total_value}")

    for j, capacity in enumerate(instance.capacities):
        print(f"\nKnapsack {j} (used {result.used_weights[j]}/{capacity}):")
        placed_items = [a for a in result.assignments if a.knapsack_index == j]

        if not placed_items:
            print("  - empty")
            continue

        for assignment in placed_items:
            print(
                f"  - {assignment.item_name}: "
                f"weight={assignment.weight}, value={assignment.value}"
            )


def build_test_instances() -> list[MultiKnapsackInstance]:
    instance_1 = MultiKnapsackInstance(
        name="Basic 2-knapsack instance",
        capacities=[5, 5],
        items=[
            Item("A", 2, 6),
            Item("B", 2, 10),
            Item("C", 3, 12),
            Item("D", 1, 7),
            Item("E", 4, 18),
        ],
    )

    instance_2 = MultiKnapsackInstance(
        name="Electronics 3-knapsack instance",
        capacities=[5, 6, 4],
        items=[
            Item("Laptop", 4, 3000),
            Item("Camera", 3, 1800),
            Item("Drone", 5, 2500),
            Item("Tablet", 2, 1400),
            Item("Lens", 1, 700),
            Item("Tripod", 3, 900),
            Item("Microphone", 2, 800),
        ],
    )

    instance_3 = MultiKnapsackInstance(
        name="Dense 3-knapsack instance",
        capacities=[8, 10, 7],
        items=[
            Item("I1", 5, 10),
            Item("I2", 4, 40),
            Item("I3", 6, 30),
            Item("I4", 3, 50),
            Item("I5", 2, 20),
            Item("I6", 7, 65),
            Item("I7", 1, 8),
            Item("I8", 4, 35),
        ],
    )
    
    instance_bad = MultiKnapsackInstance(
        name="Non-perfect fit instance",
        capacities=[5, 5],
        items=[
            Item("A", 4, 10),
            Item("B", 4, 10),
            Item("C", 4, 10),
        ],
    )

    return [instance_1, instance_2, instance_3, instance_bad]


def main() -> None:
    instances = build_test_instances()

    for instance in instances:
        result = solve_multi_knapsack(instance)
        validate_result(instance, result)
        print_result(instance, result)


if __name__ == "__main__":
    main()