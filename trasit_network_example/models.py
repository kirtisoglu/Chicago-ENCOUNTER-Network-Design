import classes as cl



# Initilize variables
y = [0 for j in range(facilities)]  #  locates facility j
x = [[0 for j in range(sets)] for i in range(populations)]  # assigns Census block i to facility j

# We need to change model_one_constraint_six

def model_selection(model_name):
    
    return

"-------Model 1: Single Level Model - Inertia & Identical -------"
 

if all(constraint(neighbor_solution) for constraint in constraints):

def model_one_constraints():
    return

def model_one_objective_function(x, V_b, A, p, travel_time):  # (1) Minimizes moment-of-inertia
    return sum(sum(p[i]*x[i, j]*travel_time[(i,j)]**2 for j in A) for i in V_b)

def model_one_constaint_one(y, existing):  # (2) Locates existing PHCs 
    return [y[j] == 1 for j in existing]

def model_one_constaint_two(y, possible, max_new_phc):  # (3) Limits the number of new PHCs by m
    return sum(y[j] for j in possible) <= max_new_phc

def model_one_constaint_three(x, V_blocks, possible_union_existing):  # (4) Assigns each block to only one PHC
    return [sum(x[i, j] for j in possible_union_existing) == 1 for i in V_blocks]

def model_one_constaint_four(x, y, V_b, E, C, b):  # (5) Facility is closed -> no block is assigned
    return [sum(x[i, j] for j in E + C) <= b * y[j] for i in V_b for j in E + C]

def model_one_constaint_five(x, y, V_b, E, N):  # (6) No block is assigned -> facility is closed
    return [y[j] <= sum(x[i, j] for i in V_b) for j in E + N]

def model_one_constaint_six(x, V_b, p, alpha, p_max, N):  # (7) Populates facilities almost equally
    return [sum(p[i] * x[i, j] for i in V_b) <= (1 + alpha) * p_max * d[j] for j in E + N]

def model_one_constraint_seven(A, V_b, P, j, epsilon, p_i, x, y):  # (8) min workload. result \geq 0
    
    workload = P / sum(y[j]*(1-epsilon) for j in A) 
    
    return sum(p_i * x[i, j] for i in V_b) - workload  

def model_one_constaint_seven(y, C, c, c_T):  # (8) Cost constraint
    return [sum(c[j]*y[j] for j in C) <= c_T]

 leq sum{i in V_b} p_i x_{ij} leq frac{P}{sum{j in A} y_{j}}(1+epsilon) 
forall j in A

"-------Model 2: Block-based Model-------"


"-------Model 3: District-based Model-------"


"-------Model 4: Huff-based Model-------"