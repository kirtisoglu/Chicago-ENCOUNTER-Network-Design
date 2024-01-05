import random
import math
import partitioning as pa
import numpy as np
import travel_time as travel
import sorting
import random
import classes as cl
import itertools


# Add time 90 as a stopping condition to the simulated annealing. 

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

# We reduce the neighborhoods and solution space using the problem structure.

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
#   - We can eliminate some y_j's being 1 together according to travel time, district_populations, position.

def make_neighbor_solution(blocks, existing, possible, m, epsilon, total_pop, pre_assigner, travel_time): # generates a neighbor solution

    # Create a neighbor of y 

    all_facilities = {**existing, **possible}  
    
    # Item (1) 
    neighbor_locator = {facility: 1 if facility in existing.keys() else 0 for facility in all_facilities.keys()}

    # Item (2)

    ##min_closed = len(possible) - m
    ##random_closed = random.sample(possible.keys(), min_closed) # Their values are already 0.
    ##remaining_possible = possible.copy()

    ##for key in random_closed:
    ##    del remaining_possible[key]

    # Item (3)
    ##for facility in remaining_possible.keys():  
    ##   neighbor_locator[facility] = random.choice(range(2)) 

    
    # Create x_ij's, binary variables of assigning block i to facility j
        
    # Item (5)
    # neighbor_assigner = {(block, facility): 0 for block in blocks.keys() for facility in all_facilities.keys()}

    # Open facilities in neigbor_locator
    open_facilities = []
    for j in all_facilities.keys():
        if neighbor_locator[j] == 1:
                open_facilities.append(j)

    # Sort the travel times from blocks to their facilities
    assignments = {}  
    
    for block in blocks.keys():
        for facility in open_facilities:
            if pre_assigner[(block, facility)] == 1:
                assignments[block] = facility      # extracts the facilities of assigned blocks
    

    pre_assigner_travel = {}
    for block in blocks.keys():
        pre_assigner_travel[(block, assignments[block])] = travel_time[(block, assignments[block])]  # Extraction     


    sorted_pair_list = sorting.sort_dictionary(pre_assigner_travel)  # sorts the list of (key, value)'s. 
    
    facility_candiate = {}
    for facility in open_facilities:
        facility_candiate[facility] = []

    for pair in reversed(sorted_pair_list):
        facility = pair[0][1]
        facility_candiate[facility].append(pair)

    
    neighbor_assigner = pre_assigner.copy()

    for facility in open_facilities:
        travel_list = facility_candiate[facility]
        first_bad_block = travel_list[0][0][0]
        second_bad_block = travel_list[1][0][0]

        first_new_facility = min((travel_time[(first_bad_block, facility)], facility) for facility in open_facilities)[1]
        second_new_facility = min((travel_time[(second_bad_block, facility)], facility) for facility in open_facilities)[1]
         
        neighbor_assigner[(first_bad_block, facility)] = 0
        neighbor_assigner[(second_bad_block, facility)] = 0

        neighbor_assigner[(first_bad_block, first_new_facility)] = 1
        neighbor_assigner[(second_bad_block, second_new_facility)] = 1

    # Item (4): For every i, pick a random j s.t. y_j = 1, set x_ij = 1. For others, set x_ij=0


    #  6) - Constraint (6): Equal district_populations constraint. We consider only the upper bound.
    #                       We will consider the lower bound in the objective function.
    
    ##available_facilities = open_facilities.copy()

    assigned_district_populations = {facility: 0 for facility in all_facilities.keys()}

    
    
    ##lower_bound = (1 - epsilon)* total_pop / len(open_facilities)
    ##upper_bound = (1 + epsilon)* total_pop / len(open_facilities)
   
    ##for i in blocks.keys():

        ##while True:
        ##j = random.choice(available_facilities)
        ##node = blocks[i]
            ##if assigned_district_populations[j] + node.get_node_district_populations() >= upper_bound:
            ##    available_facilities.remove(j) 
            ##if assigned_district_populations[j] + node.get_node_district_populations() <= upper_bound:
            ##    break 

        ##neighbor_assigner[i, j] = 1
        ##assigned_district_populations[j] = assigned_district_populations[j] + node.get_node_district_populations()

    return neighbor_locator, neighbor_assigner, assigned_district_populations

def objective_function(blocks, existing, possible, travel, assigner): ### list, keys, values problem here ???
                                                                      ### lower bound of constraint (6) ???
    all_facilities = {**existing, **possible}  

    result = sum(sum(blocks[i].get_node_population()/20*assigner[(i, j)]*travel[(i,j)]**2 for j in all_facilities.keys()) for i in blocks.keys())  # j in opens instead of all_facilities??
    return result

def simulated_annealing(blocks, existing, possible, partitioning, travel_time, max_iterations, temperature, 
                         cooling_rate, m, epsilon, t_pop, C):
    
    initial_locator, initial_assigner = generate_initial_solution(blocks, existing, possible, partitioning)
    locators = {}
    locators[0] = initial_locator

    current_assigner = initial_assigner
 
    current_locator = initial_locator
    current_energy = objective_function(blocks, existing, possible, travel_time, current_assigner)
    
    
    for iteration in range(max_iterations):
        
        # Generate a neighbor solution
        neighbor_loc, neighbor_ass, assigned_pop = make_neighbor_solution(blocks, existing, possible, m, epsilon, t_pop, current_assigner, travel_time)
        locators[iteration+1] = neighbor_loc
        # Check constraints for the neighbor solution
        #if all(constraint(neighbor_solution) for constraint in constraints):
        
        # Calculate neighbor energy (objective function value)
        neighbor_energy = objective_function(blocks, existing, possible, travel_time, neighbor_ass)
        if iteration % 50 == 0 and iteration <= 250:
            k = iteration
            print(neighbor_energy, "-> iteration =",k)
        # Calculate energy change (delta E)
        delta_energy = neighbor_energy - current_energy
        
        # If the neighbor solution is better or accepted based on the Metropolis criterion, update the current solution
        if delta_energy < 0 or random.random() < math.exp(-1/ temperature):
            current_assigner = neighbor_ass
            current_locator = neighbor_loc
            current_energy = neighbor_energy
        
        # Cool the temperature
        #temperature *= C / iteration
    
    return current_assigner, current_locator, current_energy, locators



# -------------------- SA 2: district_populations Focused --------------------

# Initial solution = heuristic solution
# Neighbor generation = randomly pick blocks from boundaries and move them to the neighbor districts
# Objective function 1 = max total district_populations - min total district_populations
# Objective function 2 = K accessibility

def make_neighbor_pop(blocks, all_facilities, locations, m, epsilon, pre_assigner, travel_time):
    
    
    neighbor_locator = {facility: 1 if facility in locations.keys() else 0 for facility in all_facilities.keys()}

    districts = {}  # locations would be enough, though.
    boundaries = {}
    district_populations = {}
    other_districts = {}


    # Initialize districts and their total populationss.
    for j in locations.keys():
        districts[j] = []
        district_populations[j] = 0 
    
        
    # Determine districts and their total populations
    for key, value in pre_assigner.items():
        if value == 1:
            districts[key[1]].append(key[0])
            district_populations[key[1]] = district_populations[key[1]] + blocks[key[0]].get_node_population()
    
    
    # Determine the biggest and smallest districts
    biggest_district = max((district_populations[district], district) for district in districts.keys())[1]
    smallest_district = min((district_populations[district], district) for district in districts.keys())[1]


    # Determine the boundaries of districts
    # Merge the boundaries of other districts than the smallest district
    for district in districts.keys():
        boundaries[district] = {}   # key: block in the boundary, value: list of (neighbor, neighbor's district)'s
        for block in districts[district]:
            for neighbor in blocks[block].get_node_neighbors():
                if neighbor in blocks.keys() and neighbor not in districts[district]:
                    boundaries[district][block] = []
                    key = list(filter(lambda x: neighbor in districts[x], districts))[0]
                    boundaries[district][block].append((neighbor, key))

    neighbor_districts_boundaries = {}  # block : its district
    for district in districts.keys():
        if district != smallest_district:
            for block in boundaries[district].keys():
                if boundaries[district][block][0][1] == smallest_district:
                    neighbor_districts_boundaries[block] = district       



    # Sort the travel times from blocks to their facilities.
    #pre_assigner_travel = {}
    #for district in districts.keys():
    #    for block in districts[district]:
    #        pre_assigner_travel[(block, district)] = travel_time[(block, district)]                
    #sorted_pair_list = sorting.sort_dictionary(pre_assigner_travel)  # sorts the list of (key, value)'s.
                    
    #block_leaving_biggest = max((travel_time[block, biggest_district], block) for block in 
        #boundaries[biggest_district].keys())[1]
    #pair_coming_smallest = min((travel_time[block, smallest_district], block) for block in 
    #neighbor_districts_boundaries.keys())[1]

    
    
    block_leaving_biggest = random.choice(list(boundaries[biggest_district].keys()))
    block_coming_smallest = random.choice(list(neighbor_districts_boundaries.keys()))


    neighbor_assigner = pre_assigner.copy()
         
    neighbor_assigner[(block_leaving_biggest, biggest_district)] = 0
    neighbor_assigner[(block_leaving_biggest, boundaries[biggest_district][block_leaving_biggest][0][1])] = 1


    neighbor_assigner[(block_coming_smallest, neighbor_districts_boundaries[block_coming_smallest])] = 0
    neighbor_assigner[(block_coming_smallest, smallest_district)] = 1


    return neighbor_locator, neighbor_assigner

def objective_function_pop(blocks, existing, possible, travel, assigner, locator): ### list, keys, values problem here ???
                                                                      ### lower bound of constraint (6) ???
    all_facilities = {**existing, **possible} 
    district_populations = {}

    # Total district_populations in any district is zero.
    open_facilities = []
    for j in all_facilities.keys():
        if locator[j] == 1:
            open_facilities.append(j)
            district_populations[j] = 0

    # Caculate the total district_populations in each district
    for block in blocks.keys():
        for facility in open_facilities:
            if assigner[(block, facility)] == 1:
                district_populations[facility] = district_populations[facility] + blocks[block].get_node_population()
    

    # Determine the highest and lowest district_populationss of districts
    pop_min = min(district_populations[facility] for facility in open_facilities) 
    pop_max = max(district_populations[facility] for facility in open_facilities)

    #return = sum(sum(blocks[i].get_node_district_populations()/20*assigner[(i, j)]*travel[(i,j)]**2 for j in all_facilities.keys()) for i in blocks.keys())  # j in opens instead of all_facilities??
    return pop_max - pop_min

def objective_function_access(blocks, existing, possible, trav_tim, assigner, locator, beta, K, tot_pop):
                                                                      ### lower bound of constraint (6) ???
    all_facilities = {**existing, **possible} 
    district_populations = {}

    # Total district_populations in any district is zero.
    open_facilities = []
    for j in all_facilities.keys():
        if locator[j] == 1:
            open_facilities.append(j)
            district_populations[j] = 0

    # Caculate the total district_populations in each district
    for block in blocks.keys():
        for facility in open_facilities:
            if assigner[(block, facility)] == 1:
                district_populations[facility] = district_populations[facility] + blocks[block].get_node_population()
    

    # Determine the highest and lowest district_populationss of districts
    biggest_district = max((district_populations[facility], facility) for facility in open_facilities)[1]
    smallest_district = min((district_populations[facility], facility) for facility in open_facilities)[1]
    
    pop_max = district_populations[biggest_district]
    pop_min = district_populations[smallest_district]
    
    # K facilities that have the least access values. We will maximize sum of their access values. 
    value_of_acce = {}
    K_max_access_values = {}

    for facility in open_facilities:
        value_of_acce[facility] = 0
        for block in blocks.keys():
            node = blocks[block]
            value_of_acce[facility] = value_of_acce[facility] + node.get_node_population() * (trav_tim[(block, facility)] ** beta) * assigner[(block, facility)]

    sorted_accesses = sorting.sort_dictionary(value_of_acce)
    K_max_accesses_pairs = itertools.islice(reversed(sorted_accesses), K)
    
    for pair in K_max_accesses_pairs:
        facility = pair[0]
        K_max_access_values[facility] = pair[1]

    # penalize exceeding the workload (workload might be 1500 for each facility.)
    ideal_workload = tot_pop / len(open_facilities)

    squared_differences = sum((district_populations[facility]-ideal_workload)**beta for facility in open_facilities)
    sum_max_accesses = sum(K_max_access_values[facility] for facility in K_max_access_values.keys())

    return sum_max_accesses, district_populations #+ squared_differences

def simulated_annealing_pop(blocks, existing, possible, travel_time, max_iterations, temperature, cooling_rate, m, epsilon, C, locator_heu, assigner_heu, beta, K, tot_pop, all_facilities, locations):
    
    current_assigner = assigner_heu
    current_locator = locator_heu

    current_energy = objective_function_pop(blocks, existing, possible, travel_time, assigner_heu, locator_heu)
    current_energy_2, current_workload = objective_function_access(blocks, existing, possible, travel_time, assigner_heu, locator_heu, beta, K, tot_pop)
    
    assigners = []

    for iteration in range(max_iterations):
        
        # Generate a neighbor solution
        neighbor_loc, neighbor_ass = make_neighbor_pop(blocks, all_facilities, locations, m, epsilon, current_assigner, travel_time)
        
        # Check constraints for the neighbor solution
        #if all(constraint(neighbor_solution) for constraint in constraints):
        
        # Calculate neighbor energy (objective function value)
        neighbor_energy = objective_function_pop(blocks, existing, possible, travel_time, neighbor_ass, neighbor_loc)
        neigbor_energy_2, neighbor_workload = objective_function_access(blocks, existing, possible, travel_time, neighbor_ass, neighbor_loc, beta, K, tot_pop)
        
    
        # Calculate energy change (delta E)
        delta_energy = neighbor_energy - current_energy
        delta_energy_2 = neigbor_energy_2 - current_energy_2
        
        # If the neighbor solution is better or accepted based on the Metropolis criterion, update the current solution
        if delta_energy < 0 and delta_energy_2 < 0: 
            #or random.random() < math.exp(-1/ temperature):
            current_assigner = neighbor_ass
            current_locator = neighbor_loc
            current_energy = neighbor_energy
            current_energy_2 = neigbor_energy_2
            current_workload = neighbor_workload


        if iteration % 50 == 0 and iteration <= 250:
            k = iteration
            print(neighbor_energy, "-> iteration =",k)
            print(neigbor_energy_2)
            print(current_workload)

            assigners.append(current_assigner)
            
        # Cool the temperature
        #temperature *= C / iteration

    return current_assigner, current_locator, current_energy, assigners, current_workload








def make_neighbor_cont(blocks, all_facilities, locations, m, epsilon, pre_assigner, travel_time):
    
    neighbor_locator = {facility: 1 if facility in locations.keys() else 0 for facility in all_facilities.keys()}

    districts = {}  # locations would be enough, though.
    boundaries = {}
    district_populations = {}
    other_districts = {}


    # Initialize districts and their total populationss.
    for j in locations.keys():
        districts[j] = []
        district_populations[j] = 0 
    
        
    # Determine districts and their total populations
    for key, value in pre_assigner.items():
        if value == 1:
            districts[key[1]].append(key[0])
            district_populations[key[1]] = district_populations[key[1]] + blocks[key[0]].get_node_population()
    
    
    # Determine the biggest and smallest districts
    biggest_district = max((district_populations[district], district) for district in districts.keys())[1]
    smallest_district = min((district_populations[district], district) for district in districts.keys())[1]



    # Determine the boundaries of districts
    # Merge the boundaries of other districts than the smallest district

    for district in districts.keys():
        boundaries[district] = {}   # key: block in the boundary, value: list of (neighbor, neighbor's district)'s
        for block in districts[district]:
            for neighbor in blocks[block].get_node_neighbors():
                if neighbor in blocks.keys() and neighbor not in districts[district]:
                    boundaries[district][block] = []
                    key = list(filter(lambda x: neighbor in districts[x], districts))[0]
                    boundaries[district][block].append((neighbor, key))

    neighbor_districts_boundaries = {}
    for district in districts.keys():
        if district != smallest_district:
            for block in boundaries[district].keys():
                if boundaries[district][block][0][1] == smallest_district:
                    neighbor_districts_boundaries[block] = district       



    # Sort the travel times from blocks to their facilities.
    pre_assigner_travel = {}
    for district in districts.keys():
        for block in districts[district]:
            pre_assigner_travel[(block, district)] = travel_time[(block, district)]                

    sorted_pair_list = sorting.sort_dictionary(pre_assigner_travel)  # sorts the list of (key, value)'s.
                    

    #block_leaving_biggest = max((travel_time[block, biggest_district], block) for block in 
        #boundaries[biggest_district].keys())[1]
    #pair_coming_smallest = min((travel_time[block, smallest_district], block) for block in 
        #neighbor_districts_boundaries.keys())[1]

    block_leaving_biggest = random.choice(list(boundaries[biggest_district].keys()))
    block_coming_smallest = random.choice(list(neighbor_districts_boundaries.keys()))



    neighbor_assigner = pre_assigner.copy()
        
    neighbor_assigner[(block_leaving_biggest, biggest_district)] = 0
    neighbor_assigner[(block_leaving_biggest, boundaries[biggest_district][block_leaving_biggest][0][1])] = 1


    neighbor_assigner[(block_coming_smallest, neighbor_districts_boundaries[block_coming_smallest])] = 0
    neighbor_assigner[(block_coming_smallest, smallest_district)] = 1

    # Contugity: Forbid the block that has an in-neighbor with in-degree 1. 
    boundaries_biggest_copy = list(boundaries[biggest_district].copy())
    
    """while True:
        block_leaving_biggest = max((travel_time[block, biggest_district], block) for block in 
            boundaries_biggest_copy)[1]

        in_degree = {}

        for neigh_key in block_leaving_biggest.get_node_neighbors():
            if neigh_key in districts[district]:
                neigh_key_value = blocks[neigh_key]
                in_degree[neigh_key] = 0

                for neigh_neigh_key in neigh_key_value.get_node_neighbors():
                    if neigh_neigh_key in districts[biggest_district]:
                        in_degree[neigh_key] = in_degree[neigh_key] + 1
        
        for neigh_neigh_key in neigh_key_value.get_node_neighbors():
            if neigh_neigh_key in districts[biggest_district]:
                if in_degree[neigh_key] != 1:
                    break
                else: 
                    boundaries_biggest_copy.remove(block_leaving_biggest)"""

    # In-degrees of blocks in the boundaries
    # boundaries[district][block].append((neighbor, key))

    return neighbor_locator, neighbor_assigner


