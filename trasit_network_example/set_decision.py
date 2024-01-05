import heuristic as heu 
from operator import itemgetter, attrgetter
import plot


def update_smallest_biggest(districts, district_populations):
    
        # Determine the biggest and smallest districts
    biggest_district = max((district_populations[district], district) for district in districts.keys())[1]
    smallest_district = min((district_populations[district], district) for district in districts.keys())[1]

    return smallest_district, biggest_district


def update_boundaries(blocks, districts, smallest_district):

    boundaries = {}
    neighboring_districts_boundaries = {}   # block : neighboring district

    # Determine the boundaries of districts
    for district in districts.keys():
        boundaries[district] = {}   # key: block in the boundary, value: list of (neighbor, neighbor's district)'s
        for block in districts[district]:
            for neighbor in blocks[block].get_node_neighbors():
                if neighbor in blocks.keys() and neighbor not in districts[district]:
                    boundaries[district][block] = []
                    key = list(filter(lambda x: neighbor in districts[x], districts))[0]
                    boundaries[district][block].append((neighbor, key))
        
    
    # Union of the boundaries of neighboring districts of the smallest district.
    for district in districts.keys():
        if district != smallest_district:
            for block in boundaries[district].keys():
                if boundaries[district][block][0][1] == smallest_district:
                    neighboring_districts_boundaries[block] = district 

    return boundaries, neighboring_districts_boundaries


def rebalancing(m, n, existing, possible, stops, blocks, all_facilities, travel_time, located, epsilon, total_pop, max_iterations):


    # assign blocks to the closest facilities    
    neighbor_locator, neighbor_assigner = heu.assignment_travel(blocks, all_facilities, located, travel_time)

    locator = neighbor_locator
    assigner = neighbor_assigner

    # initialization
    districts = {}
    district_populations = {}


    # Initialize districts and their total populationss.
    for j in located.keys():
        districts[j] = []
        district_populations[j] = 0 


    # Determine districts and their total populations
    for key, value in neighbor_assigner.items():
        if value == 1:
            districts[key[1]].append(key[0])
            district_populations[key[1]] = district_populations[key[1]] + blocks[key[0]].get_node_population()


    # bounds on the population of a district
    lower_bound = (1 - epsilon)* total_pop / len(located) 
    upper_bound = (1 + epsilon)* total_pop / len(located)
    print("Lower bound: ", lower_bound)
    print("Upper bound: ", upper_bound)
    #print(biggest_district)
    #print(districts[biggest_district])

    for _ in range(max_iterations):
        
        # Update the biggest and smallest districts
        smallest_district, biggest_district = update_smallest_biggest(districts, district_populations)


        # Update the boundaries of districts and neighboring districts of the smallest district.
        boundaries, neighbor_districts_boundaries = update_boundaries(blocks, districts, smallest_district)


        if district_populations[smallest_district]  < lower_bound or district_populations[biggest_district]  > upper_bound:

            if district_populations[biggest_district] > upper_bound:

                #determine the block leaving biggest district
                block_leaving_biggest = max((travel_time[block, biggest_district], block) for block in boundaries[biggest_district].keys())[1]
                
                #determine new district
                new_district = boundaries[biggest_district][block_leaving_biggest][0][1]

                #move it changing the assigner accordingly
                neighbor_assigner[(block_leaving_biggest, biggest_district)] = 0
                neighbor_assigner[(block_leaving_biggest, new_district)] = 1

                #update districts
                #print("->in biggest, iteration =", _)
                #print("block_leaving_biggest: ", block_leaving_biggest, "-> iteration =", _)
                #print("biggest_district: ", biggest_district, "-> iteration =", _)
                #print("district[biggest_district]: ", districts[biggest_district], "-> iteration =", _)
                districts[biggest_district].remove(block_leaving_biggest)  # previous district
                districts[new_district].append(block_leaving_biggest)  # new district

                #update district populations
                district_populations[biggest_district] = district_populations[biggest_district] - blocks[block_leaving_biggest].get_node_population() # population of the biggest district
                district_populations[new_district] = district_populations[new_district] + blocks[block_leaving_biggest].get_node_population() # population of the smallest district        
        
              
            if district_populations[smallest_district] < lower_bound:

                #determine the block leaving smalesst district
                block_coming_smallest = min((travel_time[block, smallest_district], block) for block in neighbor_districts_boundaries.keys())[1]
                #print("->in smmlaesst, iteration =", _)
                #print("block_coming_smallest: ", block_coming_smallest, "-> iteration =", _)
                #print("smallest_district: ", smallest_district, "-> iteration =", _)
                #print("district[smallest_district]: ", districts[smallest_district], "-> iteration =", _)
                #determine old district
                old_district = neighbor_districts_boundaries[block_coming_smallest]

                #move it changing the assigner accordingly
                neighbor_assigner[(block_coming_smallest, old_district)] = 0
                neighbor_assigner[(block_coming_smallest, smallest_district)] = 1

                #update districts
                districts[old_district].remove(block_coming_smallest)  # previous district
                districts[smallest_district].append(block_coming_smallest)  # new district

                #update district populations
                district_populations[smallest_district] = district_populations[smallest_district] + blocks[block_coming_smallest].get_node_population() # population of the smallest district
                district_populations[old_district] = district_populations[old_district] - blocks[block_coming_smallest].get_node_population() # population of the old district    
                

            # Update the biggest and smallest districts
          
            print("->after loops, iteration =", _, district_populations)
            #print("biggest_district: ", biggest_district, "-> iteration =", _)
            #print("district[biggest_district]: ", districts[biggest_district], "-> iteration =", _)
            #print("smallest_district: ", smallest_district, "-> iteration =", _)
            #print("district[smallest_district]: ", districts[smallest_district], "-> iteration =", _)
             
        else:
            break

    plot.plot(m, n, blocks, existing, possible, stops, locator, assigner, "Plot 1: Initial Assignment")

    return neighbor_locator, neighbor_assigner, district_populations 


# Move 1: ReCenter Move: Pick one district and close its current center and open a new center in that district. 
#         Fix the overall solution based on this by reallocating blocks between this district and other districts using "Procedure for rebalancing population" (Move 3 as many times as possible?).

# Move 2: Split Move: Pick one district. keep its current center open, and open a second center in that district. Allocate the block containing this new center to a new district.
#         Fix the overall solution based on this by reallocating blocks between this district and other districts using "Procedure for rebalancing population" (Move 3 as many times as possible?).

# Move 3: Rebalancing Move: Pick a border block and move it to a neighboring district while keeping population constraint satisfied. Do this k times.


def move_1():
    return


# Located sets are the solutions. 
# Make the sets neighbors over their rebalanced assignments.
# make_neighbors will produce new sets.
# Objective function, K accessibility, will measure the quality of a set.
# SA will pick a set as optimum.


def make_neighbors(located_set, rebalanced_assigner, rebalanced_locator):




    return


      
def objective_function():
    return
def simulated_annealing():
    return

"""def initial_solution(grid, blocks, all_facilities, travel, located, epsilon, total_pop):
    
    locator, assigner = heu.assignment_travel(blocks, all_facilities, located, travel)

    districts = {}
    boundaries = {}
    district_populations = {}
    district_populations_list = []
    relationships = {}

    # Initialize districts and their total populationss.
    for j in located.keys():
        districts[j] = []
        district_populations[j] = 0

    # Determine districts and their total populations Sort the populations. 
    for block in blocks.keys():
        for facility in located:
            if assigner[(block, facility)] == 1:
                district_populations[facility] = district_populations[facility] + blocks[block].get_node_population()
                districts[facility].add(block)
            district_populations_list.append((facility, district_populations[facility]))  # (district, population)
        sorted(district_populations_list, key=itemgetter(1), reverse=True)


    # sort travel times from blocks to center for each district
    travel_time_list = {}
    for source in located:
        travel_time_list[source] = []
        travel_time = tt.travel_time_to_source(grid, source)    # dictionary.  "node" : "time" for node in grid.keys()
        for node, time in travel_time.items():
            travel_time_list[source].append((node, time))
        sorted(travel_time_list[source], key=itemgetter(1), reverse=True)
    

    # Determine the boundaries of districts. Sort blocks in boundary
    ## Merge the boundaries of other districts than the smallest district
    for district in districts.keys():
        boundaries[district] = {}   # key: block in the boundary, value: list of (neighbor, neighbor's district)'s
        for block in districts[district]:
            for neighbor in blocks[block].get_node_neighbors():
                if neighbor in blocks.keys() and neighbor not in districts[district]:
                    boundaries[district][block] = []
                    key = list(filter(lambda x: neighbor in districts[x], districts))[0]
                    boundaries[district][block].append((neighbor, key))



    return 
"""

