import random
import math
import partitioning as pa
import numpy as np
import travel_time as travel


# 2) add time 90 as a variable to simulated annealing. When we call simulated annealing, it should be correlated.
# 4) Add model variable. When we call a model, SA should detect its obj func and constraints.


def generate_initial_solution(blocks, existing, possible, partitioning):

    all_facilities = {**existing, **possible}
    
    # Create y_j's setting existing facilities to 1.
    locator = {facility: 1 if facility in existing.keys() else 0 for facility in all_facilities.keys()}

    # Create x_ij's,
    assigner = {(i, j): 0 for i in blocks.keys() for j in all_facilities.keys()} 

    for i in blocks.keys():
        for j in existing.keys():
            if partitioning[(i, j)] == 1:
                assigner[(i, j)] = 1 

    return locator, assigner

# Use tabu list to save visited locations for not producing them again with our random function.

# In the case that the tabu rules will cause all neighbors to be tabu, we choose the “least tabu” neighbor.

# Can we reduce the neighborhoods and solution space using the problem structure?

## 1)  - Constraint (2): fix the components of existing facilities to 1.
## 2)  - Constraint (3): pick (|possible| - m) components from the last |possible| components of an array at random 
##                     and set them to 0.
## 3)  - Determine the values of the remaining components randomly.

#  4) - Constraint (4): For every i, pick a random j s.t. y_j = 1, set x_ij = 1. 
#  5) - Constraint (5): Set x_ij = 0 for all i, if y_j = 0.
#  6) - Constraint (6): 
#  7) - Constraint (7): Assume that costs are the same. m is chosen according to budget, and the model satisfies
#                     this constraint automatically. Otherwise, determine sum of costs of some 
#                     groups whose costs exceed the threshold together. Remove them from the solution space.

#   - We can set some components of x to 0 according to travel time or position.
#   - We can eliminate some of symmetries in x.
#   - We can eliminate some y_j's being 1 together according to travel time, population, position.

def make_neighbor_solution(blocks, existing, possible, m, epsilon, total_pop, pre_locator): # generates a neighbor solution

    # Create a neighbor of y 

    all_facilities = {**existing, **possible}  
    
    # Item (1) 
    neighbor_locator = {facility: 1 if facility in existing.keys() else 0 for facility in all_facilities.keys()}

    # Item (2)

    min_closed = len(possible) - m
    random_closed = random.sample(possible.keys(), min_closed) # Their values are already 0.
    remaining_possible = possible.copy()

    for key in random_closed:
        del remaining_possible[key]

    # Item (3)
    for facility in remaining_possible.keys():  
        neighbor_locator[facility] = random.choice(range(2)) 

    
    # Create x_ij's, binary variables of assigning block i to facility j
        
    # Item (5)
    neighbor_assigner = {(block, facility): 0 for block in blocks.keys() for facility in all_facilities.keys()}

    # Item (4): For every i, pick a random j s.t. y_j = 1, set x_ij = 1. For others, set x_ij=0.
    open_facilities = []
    for j in all_facilities.keys():
        if neighbor_locator[j] == 1:
                open_facilities.append(j)


    #  6) - Constraint (6): Equal population constraint. We consider only the upper bound.
    #                       We will consider the lower bound in the objective function.
    
    available_facilities = open_facilities.copy()
    assigned_population = {facility: 0 for facility in all_facilities.keys()}  # save population assigned to each facility
    
    lower_bound = (1 - epsilon)* total_pop / len(open_facilities)
    upper_bound = (1 + epsilon)* total_pop / len(open_facilities)
   

    for i in blocks.keys():

        while True:
            j = random.choice(available_facilities)
            node = blocks[i]
            if assigned_population[j] + node.get_node_population() >= upper_bound:
                available_facilities.remove(j) 
            if assigned_population[j] + node.get_node_population() <= upper_bound:
                break 

        neighbor_assigner[i, j] = 1
        assigned_population[j] = assigned_population[j] + node.get_node_population()

    return neighbor_locator, neighbor_assigner, assigned_population



def objective_function(blocks, existing, possible, travel, assigner): ### list, keys, values problem here ???
                                                                      ### lower bound of constraint (6) ???
    all_facilities = {**existing, **possible}  

    result = sum(sum(blocks[i].get_node_population()/20*assigner[(i, j)]*travel[(i,j)]**2 for j in all_facilities.keys()) for i in blocks.keys())  # j in opens instead of all_facilities??
    return result



def simulated_annealing(blocks, existing, possible, partitioning, travel, max_iterations, temperature, cooling_rate, m, epsilon, t_pop):
    
    initial_locator, initial_assigner = generate_initial_solution(blocks, existing, possible, partitioning)
    locators = {}
    locators[0] = initial_locator

    current_assigner = initial_assigner
 
    current_locator = initial_locator
    current_energy = objective_function(blocks, existing, possible, travel, current_assigner)
    
    
    for iteration in range(max_iterations):
        
        # Generate a neighbor solution
        neighbor_loc, neighbor_ass, assigned_pop = make_neighbor_solution(blocks, existing, possible, m, epsilon, t_pop)
        locators[iteration+1] = neighbor_loc
        # Check constraints for the neighbor solution
        #if all(constraint(neighbor_solution) for constraint in constraints):
        
        # Calculate neighbor energy (objective function value)
        neighbor_energy = objective_function(blocks, existing, possible, travel, neighbor_ass)

        # Calculate energy change (delta E)
        delta_energy = neighbor_energy - current_energy
        
        # If the neighbor solution is better or accepted based on the Metropolis criterion, update the current solution
        if delta_energy < 0 or random.random() < math.exp(-delta_energy /(delta_energy/2 *temperature)):
            current_assigner = neighbor_ass
            current_locator = neighbor_loc
            current_energy = neighbor_energy
        
        # Cool the temperature
        temperature *= 1 - cooling_rate
    
    return current_assigner, current_locator, current_energy, locators


