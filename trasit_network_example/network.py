import classes as cl
import numpy as np
import random 


grid = {}  # dictionary for grid
V_blocks = {}  # set of vertices correspoding to blocks 
V_stops = {}  # set of vertices corresponding to stops
V_existing = {}  # set of vertices corresponding to existing PHCs
V_possible = {}  # set of vertices corresponding to possible PHCs
pop = [] # list of populations

def create_id(name,n,m):
    if name in create_stops(n, m):      # stop nodes  
        result = 1    
    else: 
        result = 0     
    return result

def create_x_axis(name):
    i = name[0]
    noise_x = np.random.normal(0,0.1)   
    pos_x = i + noise_x
    return pos_x

def create_y_axis(name):
    j = name[1]
    noise_y = np.random.normal(0,0.1)
    pos_y = j + noise_y
    return pos_y

def create_population(name,n,m,p_min,p_max):
    i = name[0]
    j = name[1]

    if name in create_stops(n,m):
        result = 0  # population of a stop node is zero.
    else:            
        result =  random.randrange(p_min,p_max)
    
    return result

def create_neighbors(name,n,m):
    i = name[0]
    j = name[1]
    neighbors = set()   
    if i == 0 or i == n - 1 or j == 0 or j == m - 1:   # corners
        if i != 0:
                neighbors.add((i - 1,j))  # bottom neighbor
        if j != m - 1:
                neighbors.add((i,j + 1))  # right neighbor
        if i != n - 1:
                neighbors.add((i + 1,j))  # top neighbor
        if j != 0:
                neighbors.add((i,j - 1))  # left neighbor
    else:
        neighbors.add((i - 1,j))
        neighbors.add((i + 1,j))
        neighbors.add((i,j - 1))
        neighbors.add((i,j + 1))   # middle nodes
    return neighbors

def create_stops(n, m):
    stops = set()
    for k in range(min(n, m)):
        i, j = 1 + k, k   # (k+1,k)
        stops.add((i, j))
    return stops

def create_EPHC(n,m,e):
     
    pairs = {(i, j) for i in range(1, n + 1) for j in range(1, m + 1)}  # all possible pairs (i, j)
    candidates = pairs - create_stops(n,m)   # subway nodes cannot be a PHC.

    random_EPHCs = random.sample(list(candidates), e)  # e many random elements as existing PHCs
    existing = set(random_EPHCs)  

    return existing

def create_PPHC(n,m,e,p):
    pairs = {(i, j) for i in range(1, n + 1) for j in range(1, m + 1)}  # all possible pairs (i, j)
    candidates = pairs - create_stops(n, m)   # subway nodes cannot be a PHC.
    
    remaining_elements = candidates - create_EPHC(n, m, e)
    random_PPHCs = random.sample(list(remaining_elements), p)  # Choose p random elements as possible PHCs
    possible = set(random_PPHCs)

    return possible

def travel_time(name): # shortest path algorithm hesaplayacak.
    result = 1
    return result

def create_grid(n, m, e, p, p_min, p_max):

    # Properties: name, ID, x_axis, y_axis, population, travel_time, neighbors, EPHC, PPHC

    for i in range(n):
        for j in range(m):
            
            # Property EPHC
            EPHC = 0  
            if (i,j) in create_EPHC(n,m,e):
              EPHC = 1
            else: EPHC = 0 

            # Property PPHC
            PPHC = 0   
            if (i,j) in create_PPHC(n,m,e,p):
              PPHC = 1
            else: PPHC = 0 
            
            # Property name
            name = (i,j)

            # Property ID
            id = create_id(name,n,m)

            # Property x_axis
            x_axis = create_x_axis(name)

            # Property y_axis
            y_axis = create_y_axis(name)

            # Property population
            population = create_population(name,n,m,p_min,p_max)

            # Property neighbors
            neighbors = create_neighbors(name,n,m)

            # Property travel time
            # Call it from traveltime.py

            # Assign properties of each class.
            node = cl.Node(name, id, x_axis, y_axis, population, 
                        [], neighbors, EPHC, PPHC) 

            # All nodes
            grid[(i,j)] = node
            
            # V_stops and V_blocks
            if id == 1:
                V_stops[(i,j)] = node  
            else:
                V_blocks[(i,j)] = node

            # V_existing
            if EPHC == 1:
                V_existing[(i,j)] = node

            #V_possible
            if PPHC == 1:
                V_possible[(i,j)] = node

            # Population list            
            pop.append(population)
    
    return grid, V_stops, V_blocks, V_existing, V_possible, pop