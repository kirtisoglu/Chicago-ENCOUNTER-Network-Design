import random
import math
import partitioning as pa



def find_feasible_neighbor(V_b, A, E, P, c, m, b, epsilon, p, t, tabu_list):
    while True:
        neighbor_y, neighbor_x = initialize_solution(V_b, A, E)

        # Check if the neighbor solution is feasible and not in the tabu list
        if (
            evaluate_solution(neighbor_y, neighbor_x, P, c, V_b, A, epsilon, p, t) != float('inf') and
            (neighbor_y, neighbor_x) not in tabu_list
        ):
            return neighbor_y, neighbor_x

def simulated_annealing(V_b, A, E, P, c, m, b, epsilon, p, t, iterations, initial_temperature, cooling_rate, tabu_tenure):
    current_y, current_x = initialize_solution(V_b, A, E)
    current_objective = evaluate_solution(current_y, current_x, P, c, V_b, A, epsilon, p, t)
    best_y = current_y.copy()
    best_x = current_x.copy()
    best_objective = current_objective

    temperature = initial_temperature

    # Initialize tabu list
    tabu_list = set()

    for _ in range(iterations):
        # Generate a feasible neighbor solution
        neighbor_y, neighbor_x = find_feasible_neighbor(V_b, A, E, P, c, m, b, epsilon, p, t, tabu_list)

        # Evaluate the neighbor solution
        neighbor_objective = evaluate_solution(neighbor_y, neighbor_x, P, c, V_b, A, epsilon, p, t)

        # If the neighbor solution is better, accept it
        if neighbor_objective < current_objective or random.random() < temperature:
            current_y, current_x = neighbor_y, neighbor_x
            current_objective = neighbor_objective

        # Update the best solution if needed
        if current_objective < best_objective:
            best_y, best_x = current_y, current_x
            best_objective = current_objective

        # Update the tabu list
        tabu_list.add((neighbor_y, neighbor_x))
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop()

        # Cool down the temperature
        temperature *= cooling_rate

    return best_y, best_x

# Example usage
V_b = {1, 2, 3}
A = {1, 2, 3}
E = {1, 2, 3, 4}
P = 100
c = [10, 20, 30, 40]
m = 2
b = 0.5
epsilon = 0.1

# Assuming you have a definition for p and t
p = {1: 2, 2: 3, 3: 1}
t = {(i, j): random.uniform(0, 1) for i in V_b for j in A}

best_y, best_x = simulated_annealing(V_b, A, E, P, c, m, b, epsilon, p, t, iterations=10000, initial_temperature=1.0, cooling_rate=0.99, tabu_tenure=5)
