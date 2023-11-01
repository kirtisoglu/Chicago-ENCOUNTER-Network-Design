import classes as cl


max_cost = 100
traveltime_threshold = 30
max_DN_teams = 4
max_new_PHCs = 10
alpha = 0.02


"-------Model 1: Integer Programming Model-------"

def model_selection(model_name):
    
    return

def model_1_objective_function(x, N):
    return sum(x[i, j] for i in V_b for j in N)

def model_1_constraints():
    return

def model_1_constaint_1(y, E):
    return [y[j] == 1 for j in E]

def model_1_constaint_2(y, N, m):
    return sum(y[j] for j in N) <= m

def model_1_constaint_3(x, V_b, E, N):
    return [sum(x[i, j] for i in V_b) == 1 for j in E]

def model_1_constaint_4(x, V_b, E, N, b):
    return [sum(x[i, j] for j in E + N) <= b * y[j] for i in V_b for j in E + N]

def model_1_constaint_5(x, V_b, E, N):
    return [y[j] <= sum(x[i, j] for i in V_b) for j in E + N]

def model_1_constaint_6(x, V_b, p, alpha, p_max, N):
    return [sum(p[i] * x[i, j] for i in V_b) <= (1 + alpha) * p_max * d[j] for j in E + N]

def model_1_constaint_7(d, N, d_t):
    return [sum(d[j] for j in N) <= d_t]

def model_1_constaint_8(y, N, r, R, d):
    return [r * y[j] <= d[j] <= R * y[j] for j in N]

def model_1_constaint_9(x, V_b, t_r, t_epsilon, E, N):
    return [t_r[i][j] * x[i, j] <= t_epsilon for i in V_b for j in E + N]

def model_1_constaint_10(y, N, c, c_t):
    return [sum(c[j] * y[j] for j in N) <= c_t]


"-------Model 2: Block-based Model-------"


"-------Model 3: District-based Model-------"


"-------Model 4: Huff-based Model-------"