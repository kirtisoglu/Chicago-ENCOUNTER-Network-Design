from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpBinary, PULP_CBC_CMD


# Multi-way partitioning problem is to partition a set of integers into k disjoint subsets 
# so that the sum of the numbers in each subset are as nearly equal as possible.

# We want to distribute populations of blocks to existing facilities as nearly as possible. 
# To do that, we partition the set of blocks into k many set of blocks such that 
# total populations of sets are as nearly equal as possible. k: number of existing facilities.

# We model an integer programming to solve this problem, and find an exact solution by pulp library. 


def multiway_number_partitioning(blocks, existing, time):


    problem = LpProblem("Multiway_Number_Partitioning", LpMinimize)

    binary_matrix = {(i, j): LpVariable(f"x_{i}_{j}", cat="Binary") for i in blocks.keys() for j in existing.keys()}

    # Constraint: Every number is assigned to one set.
    for i in blocks.keys():
        problem += lpSum(binary_matrix[i, j] for j in existing.keys()) == 1

    # Calculate the weight of a set
    def set_weight(set_index):
        return lpSum(blocks[i].get_node_population() * binary_matrix[i, set_index] for i in blocks.keys())

    # Define the max and min set weight variables
    max_set_weight = LpVariable("max_set_weight", lowBound=0)
    min_set_weight = LpVariable("min_set_weight", lowBound=0)

    # Add constraints to set the values of max_set_weight and min_set_weight
    for j in existing.keys():
        problem += min_set_weight <= set_weight(j)
        problem += max_set_weight >= set_weight(j)

    # Define the objective function: minimize the max difference between max and min set weights
    problem += max_set_weight - min_set_weight

    # Solve the problem
    problem.solve(PULP_CBC_CMD(msg=0, timeLimit=time))

    # Assign selected sets
    solution = {(i, j): int(binary_matrix[i, j].varValue) for i in blocks.keys() for j in existing.keys()}

    # Calculate weights using lpSum directly
    weights = [lpSum(blocks[i].get_node_population() * solution[i, j] for i in blocks.keys()) for j in existing.keys()]


    return problem.objective.value(), weights, solution


