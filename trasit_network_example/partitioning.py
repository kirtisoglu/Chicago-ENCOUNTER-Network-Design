from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpBinary, PULP_CBC_CMD


# Multi-way partitioning problem is to partition a set of integers into k disjoint subsets 
# so that the sum of the numbers in each subset are as nearly equal as possible.

# Example
#  Take A={1,2,3,4,5,6,7,8}. We want to partition A into 4 subsets. The sum of the numbers in each subset
# must be equal as much as possible. The following partition is an optimal partition, since the sum of the numbers in each set is 9.
# A_1 = {1,8}, A_2={2,7}, A_3={3,6}, A_4={4,5}

# Multi-way partitioning is an NP-hard problem. It means that the time complexity increases exponentially. 

# For the following code, consider that a city is a union of blocks, where each block consists of several buildings. 
# Each block has a population assigned to it.  We want to create districts (sets of blocks) in such a way that
# districts are nearly populated. To do that, we partition the set of blocks into k many set of blocks such that 
# total populations of the blocks in the sets are as nearly equal. Each district will be the serving area of a facility (primary care center)
# We have k facilities (sets/districts) in total.

# The model I have designed to solve this problem is added to the second page of our overleaf file: https://www.overleaf.com/project/657cbea83593afcf83bbf52c
# The following code uses the pulp library for finding an exact solution.

# The following is a function f(x) that calls input x=(blocks, locations, time), and returns an objective value, weights (sum of the populations in sets) and solution (sets).

# You will need to call some inputs in your function. # The first step for us is to arrange these inputs. What are the inputs we need? Once we are done with this,
# we will code up the model. 

# My inputs: Blocks is a dictionary of block locations. Key: block name, value: location of the block. I do not use block locations in the following code, you can ignore it.
# Locations is a dictionary. key: name of a facility (set), value: location of the facility. Again, you can ignore the value, I do no use it here. 
# Time is a parameter $t$ that stops the code after $t$ seconds. The algorithm behind the code creates a search tree. Each node in the search tree is a solution. 
# The algorithm explores the tree, and does not stop until checking all nodes/solutions in the tree. If you have a time parameter t, the code stops after t seconds, 
# and returns the best solution that is found so far.

# Basically, the code assigns blocks in block.keys() to facilities/sets/locations in location.keys(). 


def multiway_number_partitioning(blocks, locations, time):


    problem = LpProblem("Multiway_Number_Partitioning", LpMinimize)   # Create an empty minimization model. 

    binary_matrix = {(i, j): LpVariable(f"x_{i}_{j}", cat="Binary") for i in blocks.keys() for j in locations.keys()}  # x_{ij} variables are created.
 
    # Constraint: Each block is assigned to exactly one set.
    for i in blocks.keys():
        problem += lpSum(binary_matrix[i, j] for j in locations.keys()) == 1

    # Create a function that calculates the weight of a set
    def set_weight(set_index):
        return lpSum(blocks[i].get_node_population() * binary_matrix[i, set_index] for i in blocks.keys()) 

    # Define the variables of max and min weights of a set
    max_set_weight = LpVariable("max_set_weight", lowBound=0)
    min_set_weight = LpVariable("min_set_weight", lowBound=0)

    # Add constraints for max_set_weight and min_set_weight. For example: W_max \geq weight of S_i for all i.
    for j in locations.keys():
        problem += min_set_weight <= set_weight(j)
        problem += max_set_weight >= set_weight(j)

    # Define the objective function: minimize the max difference between max and min set weights
    problem += max_set_weight - min_set_weight

    # Solve the problem
    problem.solve(PULP_CBC_CMD(msg=0, timeLimit=time))

    # Assign selected sets
    solution = {(i, j): int(binary_matrix[i, j].varValue) for i in blocks.keys() for j in locations.keys()}

    # Calculate weights using lpSum directly
    weights = [lpSum(blocks[i].get_node_population() * solution[i, j] for i in blocks.keys()) for j in locations.keys()]


    return problem.objective.value(), weights, solution


