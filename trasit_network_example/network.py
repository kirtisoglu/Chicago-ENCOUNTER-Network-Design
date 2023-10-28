import classes as cl
import numpy as np
import random 


grid = {}  # dictionary for grid
V_blocks = {}  # set of vertices correspoding to blocks 
V_EPHC = {}  # set of vertices corresponding to existing PHCs
V_PPHC = {}  # set of vertices corresponding to possible PHCs


# Columns of a node are name, ID, pos_x, pos_y, population (int, 10-50), travel_time, 
#                       neighbors (set), EPHC (int, 0-1), PPHC
" time = define travel_time column somehow. It will be a list of travel times to all PHCs."


def id(t,V_s,V_b):
    if t in V_b:      # block nodes
        result = 0
    if t in V_s:      # stop nodes  
        result = 1    
    return result

def x_axis(t):
    i = t[0]
    noise_x = np.random.normal(0,0.1)   
    pos_x = i + noise_x
    return pos_x

def y_axis(t):
    j = t[1]
    noise_y = np.random.normal(0,0.1)
    pos_y = j + noise_y
    return pos_y

def population(t, V_b, V_s):
    i = t[0]
    j = t[1]

    if t in V_s:
        result = 0  # population of stop nodes is zero.
    if t in V_b:            
        result =  random.randrange(10,50)
    
    return result


def neighbors(t,n,m):
    i = t[0]
    j = t[1]
    neighbors = {}    
    if i == 0 or i == n - 1 or j == 0 or j == m - 1:   # corners
        if i != 0:
                neighbors.update((i - 1,j))  # bottom neighbor
        if j != m - 1:
                neighbors.update((i,j + 1))  # right neighbor
        if i != n - 1:
                neighbors.update((i + 1,j))  # top neighbor
        if j != 0:
                neighbors.update((i,j - 1))  # left neighbor
    else:
        neighbors.update((i - 1,j), (i + 1,j), (i,j - 1), (i,j + 1))   # middle nodes
    return neighbors

def neighbors_subway(n,m,subway_line_num,grid):
    
    neighbors = {} 

    sub_line = 0
    while sub_line < subway_line_num:
        rand_i = random.randint(0,n)
        rand_j = random.randint(0,m)
        neighbors[(rand_i,rand_j)] = [grid[(rand_i,rand_j)], 0]  # add a node at i,j position with transition cost = 0.


        sub_line += 1
    return neighbors

def subway_nodes(n,m):
    V_s = {}
    Origin = (1,0)
    Destination = (n-1,m) 
    k = 0

    while k < n-1:
        v_k = (1,0) + (k,k)

        k += 1
        
    return


def travel_time(t):
    result = 1
    return result


def grid(n,m,e,p):

    # number of subway lines
    subway_line_num = 2

    # Generate all possible pairs (i, j) 
    pairs = {(i, j) for i in range(1, n + 1) for j in range(1, m + 1)}

    # Define subway stops

    # Choose e random elements as existing PHCs
    random_EPHCs = random.sample(list(pairs), e)
    existing = set(random_EPHCs)

    # Calculate remaining possible PHCs
    remaining_elements = pairs - existing

    # Choose p random elements as possible PHCs
    random_PPHCs = random.sample(list(remaining_elements), p)
    possible = set(random_PPHCs)
    
    EPHC = 0
    PPHC = 0

    for i in range(n):
        for j in range(m):
            if (i,j) in existing:
              EPHC = 1
            else: EPHC = 0 

        if (i,j) in possible:
             PPHC = 1
        else: PPHC = 0 

        name = (i,j)

        node = cl.Node(name, id(name), x_axis(name), y_axis(name), population(name), 
                        travel_time(name), neighbors(name,n,m), neighbors_subway(n,m,subway_line_num),
                         EPHC, PPHC) # Call property functions of nodes
        
        grid[(i,j)] = node

        if EPHC == 1:
            V_EPHC[(i,j)] = node  
        if PPHC == 1:
            V_PPHC = [(i,j)] = node

        return 

def grid_edges():
    return 

def create_grid(n,m,e,p):
    return

# Columns of an edge: ID, type, line, tail, head, travel_time
E_blocks = {}  # initialize a dictionary for edges between blocks

for (i,j) in range(n,m):
    for (s,r) in range (n,m):
        ID = ((i,j),(s,r))
        "line = define this after adding a transportation network"
        "type = define this after adding a transportation network"
        tail = (i,j)    #Should I take this as a dictionary and connect to the node?
        head = (s,r)
        # travel_time = Do I need that?
        edge = cl.Edge(ID, [], [], tail, head, [])
        E_blocks[(i,j),(r,s)] = edge