import pulp
import numpy as np
import random


def solve_bin_packing(weights, bin_capacity):
    """
    Solve the bin packing problem using PuLP.

    Args:
        weights (list): List of item weights.
        bin_capacity (int): Maximum capacity of each bin.

    Returns:
        dict: A dictionary with the bin assignments for each item.
    """
    n = len(weights)  # Number of items
    bins = range(n)   # Assume the maximum number of bins needed is equal to the number of items

    # Define the problem
    problem = pulp.LpProblem("Bin_Packing_Problem", pulp.LpMinimize)

    # Variables
    x = pulp.LpVariable.dicts("x", [(i, j) for i in range(n) for j in bins], cat="Binary")
    y = pulp.LpVariable.dicts("y", bins, cat="Binary")

    # Objective: Minimize the number of bins used
    problem += pulp.lpSum([y[j] for j in bins])

    # Constraints
    # Each item must be assigned to exactly one bin
    for i in range(n):
        problem += pulp.lpSum([x[i, j] for j in bins]) == 1

    # The total weight in each bin cannot exceed its capacity
    for j in bins:
        problem += pulp.lpSum([weights[i] * x[i, j] for i in range(n)]) <= bin_capacity * y[j]

    # Solve the problem
    problem.solve()

    # Extract the solution
    if pulp.LpStatus[problem.status] == "Optimal":
        solution = {f"Item_{i}": [j for j in bins if pulp.value(x[i, j]) == 1][0] for i in range(n)}
        num_bins_used = sum(pulp.value(y[j]) for j in bins)
        return {
            "status": "Optimal",
            "num_bins_used": int(num_bins_used),
            "bin_assignments": solution,
        }
    else:
        return {"status": pulp.LpStatus[problem.status]}


def generate_random_bin_packing_data(num_items, min_weight, max_weight, bin_capacity):
    """
    Generate random data for the bin packing problem.

    Args:
        num_items (int): Number of items to generate.
        min_weight (int): Minimum weight of an item.
        max_weight (int): Maximum weight of an item.
        bin_capacity (int): Capacity of each bin.

    Returns:
        tuple: A tuple containing the list of item weights and the bin capacity.
    """
    weights = [random.randint(min_weight, max_weight) for _ in range(num_items)]
    return weights, bin_capacity


# Example usage
num_items = 100
min_weight = 1
max_weight = 10
bin_capacity = 100

weights, bin_capacity = generate_random_bin_packing_data(num_items, min_weight, max_weight, bin_capacity)
print("Generated Weights:", weights)
print("Bin Capacity:", bin_capacity)

# weights = [5, 7, 8, 3, 4, 2]
# bin_capacity = 10
result = solve_bin_packing(weights, bin_capacity)
print(result)
