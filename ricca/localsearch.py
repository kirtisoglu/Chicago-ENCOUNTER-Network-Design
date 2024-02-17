import networkx as nx
import random
import copy
import collections
import sample

import random
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Union,
    Hashable,
    Sequence,
    Tuple,
)


# A path from a vertex to the root of a random spanning tree is called loop-erased random walk (Kulkarni'90).
# Wilson'96 uses these walks to sample a spanning tree. The algorithm also works for directed graphs.
# For simple graphs, it has two versions: RandomTreeWithRoot(r) and RandomTree. We need the rooted version.


#             Wilson's Algorithm for Sampling a Rooted Spanning Tree

#  for i = 1,...,n
#  
#
#
#
#
#
#
#
#
#
def uniform_spanning_tree(graph: nx.Graph, choice: Callable = random.choice) -> nx.Graph:
    """
    Builds a spanning tree chosen uniformly from the space of all
    spanning trees of the graph. Uses Wilson's algorithm.

    :param graph: Networkx Graph
    :type graph: nx.Graph
    :param choice: :func:`random.choice`. Defaults to :func:`random.choice`.
    :type choice: Callable, optional

    :returns: A spanning tree of the graph chosen uniformly at random.
    :rtype: nx.Graph
    """

    new_graph = graph.copy(as_view=False)

    # remove the edges between stops before sampling.
    for edge in new_graph.edges:
        endpoint1, endpoint2 = edge
        if new_graph.nodes[endpoint1]["id"] == 1 and new_graph.nodes[endpoint2]["id"] == 1:
            new_graph.remove_edge(endpoint1, endpoint2)

    root = choice(list(new_graph.nodes))
    tree_nodes = set([root])
    next_node = {root: None}

    for node in new_graph.nodes:
        u = node
        while u not in tree_nodes:
            next_node[u] = choice(list(new_graph.neighbors(u)))
            u = next_node[u]

        u = node
        while u not in tree_nodes:
            tree_nodes.add(u)
            u = next_node[u]

    G = nx.Graph()
    for node in tree_nodes:
        if next_node[node] is not None:
            G.add_edge(node, next_node[node])

    # re-assign the attributes of the nodes.
    for node in G.nodes:
        G.nodes[node].update(graph.nodes[node])

    return G


# Explain how to sample the tree.

"""def sample_tree(G):

    new_graph = G.copy(as_view=False)

    # remove the edges between stops before sampling.
    for edge in new_graph.edges:
        endpoint1, endpoint2 = edge
        if new_graph.nodes[endpoint1]["id"] == 1 and new_graph.nodes[endpoint2]["id"] == 1:
            new_graph.remove_edge(endpoint1, endpoint2)

    # sample a spanning tree. Probability is calculated over the multiplication of edge weights: travel time of the edge.
    tree = nx.random_spanning_tree(new_graph, weight=None, multiplicative=True, seed=None)

    # re-assign the attributes of the nodes.
    for node in tree.nodes:
        tree.nodes[node].update(G.nodes[node])
    
    return tree"""



# Generate an initial partition / solution

#     Initialize set of clusters: $C = \{ \}$.  (Dictionary. key: cluster name, value: list of vertices)
#     Determine the set of the connected components in $G$, $\mathcal{G}=\{G_1, G_2, ..., G_k\}$. 
#     Repeat until $\mathcal{G} = \emptyset$.
#
#        For each component $G_i$, do:
#
#            If there is only one facility in the vertex set:
#                Put $G_i$ to $C$ and remove it from $G$.
#
#            else:   
#                Choose a random facility $u$ from $G_i$. 
#                Calculate lengths of the shortest paths from $u$ to all other facilities. 
#                Pick the minimum length. 
#                Let $P$ be one of the shortest paths corresponding to the minimum length. 
#                Choose an edge from $P$ uniformly at random and remove the edge.
#                Update $G$.
#        
#        Update $\mathcal{G}$.

def generate_initial_partition(G, grid, open_facilities):  # cuts random p-1 edges.

    clusterim = {}  #  key: cluster name,  value: list of vertices
    populations = {}

    spanning_tree = uniform_spanning_tree(G)
    # spanning_tree = sample_tree(G)


    tree = spanning_tree.copy(as_view=False)
    setim = list(nx.connected_components(tree)) # list of sets of nodes

    while len(tree.nodes) > 0:  # neden nodelarin sayisi sifirdan buyukse
        
        #print("Num of components in Tree: ", len(S))

        for component in setim: # component is a set of nodes 
            #print("component nodes:", component)
            facilities_in_component = [value for value in open_facilities if value in list(component)]

            if len(facilities_in_component) == 1:
                #print("num of facilities_in_component", len(facilities_in_component))
                #print("facilities_in_component", facilities_in_component)
                clusterim[facilities_in_component[0]] = list(component)
                #print("component", C[facilities_in_component[0]])
                tree.remove_nodes_from(component)
                #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            else:  
                #print("facilities in compoenent (else)", facilities_in_component) 
                #print("num of facilities_in_component (else)", len(facilities_in_component))
                u = random.choice(facilities_in_component)
                #print("u:", u)
                paths = {v: nx.shortest_path(tree, source=u, target=v) for v in facilities_in_component if v != u} # dictionary. key: target node v, value: list of nodes 
                #print("paths", paths)
                lengths = {}
                k = 500
                closest = (0,0)
                for v in paths.keys():   
                   if len(paths[v]) < k:
                        k = len(paths[v])
                        closest = v
                #closest = min((len(paths[v]), v) for v in paths.keys())[1]  # ?
                
            
                #print("closest:", closest)
                path = tree.subgraph(paths[closest])
                #print("path nodes", path.nodes)
                edge = random.choice(list(path.edges))
                #print("removed edge:", edge)
                tree.remove_edge(*edge)  # one edge is removed from the tree. 
            #print("####################################################") 

        setim = list(nx.connected_components(tree)) # list of sets of nodes

    for center in clusterim.keys():
        populations[center] = 0

        for node in clusterim[center]:
            populations[center] = populations[center] + grid[node].get_node_population()

    # Create the partitioned graph
    component = []
    for center in clusterim.keys():
        component.append(nx.induced_subgraph(spanning_tree, clusterim[center]))
    
    partitioned_tree = nx.union(component[0], component[1])
    for index, graph in enumerate(component):
        if index > 1:
            partitioned_tree = nx.union(partitioned_tree, graph)

    return partitioned_tree, spanning_tree, clusterim, populations


def generate_initial_partition_BFS(G, grid, open_facilities):
    clusterim = {}
    populations = {}

    spanning_tree = uniform_spanning_tree(G)

    tree = spanning_tree.copy(as_view=False)
    setim = list(nx.connected_components(tree))

    # Function to find the closest facility to node u in a given component
    def find_closest_facility(tree, u, facilities_in_component):
        visited = set()
        queue = collections.deque([(u, 0)])  # (node, distance)
        closest_distance = float('inf')
        closest_facility = None

        while queue:
            node, distance = queue.popleft()
            if node in facilities_in_component and node != u:
                closest_distance = distance
                closest_facility = node
                break  # Found the closest facility
            visited.add(node)
            neighbors = tree.neighbors(node)
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))

        return closest_facility

    while len(tree.nodes) > 0:
        for component in setim:
            facilities_in_component = [value for value in open_facilities if value in list(component)]

            if len(facilities_in_component) == 1:
                clusterim[facilities_in_component[0]] = list(component)
                tree.remove_nodes_from(component)
            else:
                u = random.choice(facilities_in_component)
                closest = find_closest_facility(tree, u, facilities_in_component)
                path = nx.shortest_path(tree, source=u, target=closest)
                path_edges = list(zip(path[:-1], path[1:]))
                edge = random.choice(path_edges)
                tree.remove_edge(*edge)

        setim = list(nx.connected_components(tree))

    for center in clusterim.keys():
        populations[center] = 0

        for node in clusterim[center]:
            populations[center] += grid[node].get_node_population()

    component = []
    for center in clusterim.keys():
        component.append(nx.induced_subgraph(spanning_tree, clusterim[center]))

    partitioned_tree = nx.union(component[0], component[1])
    for index, graph in enumerate(component):
        if index > 1:
            partitioned_tree = nx.union(partitioned_tree, graph)

    return partitioned_tree, spanning_tree, clusterim, populations

# 0. Initialize 
#               clusters: 
#               partitioned_tree: Currently empty. Will be the final subgraph with p components.
#               populations: measures the populations of clusters
#               components:

# 1. Generate a spanning tree T of graph G. Let T' be a copy of T. 
# 2. Put T in the list of components. Currently, |components| = 1.
# 3. While |components| < |open_facilities|, do:
#       for every component in compinents, do:

# 


def generate_initial_solution1(G, grid, open_facilities):
    
    clusters = {}  #  key: cluster name,  value: list of vertices
    partitioned_tree = nx.empty_graph(0) 
    populations = {}  # key: center node,  value: total population of blocks assigned to that center
    components = []

    spanning_tree = uniform_spanning_tree(G)
    tree = spanning_tree.copy(as_view=False)
    components.append(tree)
    
    while len(list(nx.connected_components(partitioned_tree))) < len(open_facilities):

        for component in components:
            bad_component = True

            while bad_component == True:
                edge = random.choice(list(component.edges))
                component.remove_edge(*edge)
                
                S = [component.subgraph(component_component).copy() for component_component in nx.connected_components(component)]

                valid = all(any(facility in component_component.nodes for facility in open_facilities) for component_component in S)
                if valid == False:
                    component.add_edge(*edge)
                    bad_component = True
                else:
                    bad_component = False 
            
            components.remove(component)

            for component in S:
                facilities = list(set(open_facilities).intersection(component.nodes))
                #print(facilities)

                if len(facilities) <= 1:
                    partitioned_tree = nx.union(partitioned_tree, component)
                    clusters[facilities[0]] = list(component.nodes)

                else: 
                    components.append(component)

    for center in clusters.keys():
        populations[center] = 0

        for node in clusters[center]:
            populations[center] = populations[center] + grid[node].get_node_population()

    return clusters, populations, partitioned_tree, spanning_tree

# A clustering is a solution. A neighbor of a solution is a clustering (solution) in which exactly one node's cluster differs.
# For contiguity, this node must in a (node) boundary. In other words, a migrating node
#
#    (a) must be adjacent to some node of the destination-district.
#
# This was an obvious condition. The following condition is not that obvious but necessary and sufficient with (a) for having connected subgraphs in neighboring clustering.
#
#   (b) must not be a cut-node for the origin-district.
#
# The origin-district is chosen at random. WHY?
# We define the boundary of the origin district. Then we check if removing a node in the boundary makes the origin district disconnected.
# If the district is still connected, we move the migrating node safely and save the new clustering as a neighbor.
# The exact neighborhood (all neighbors) is determined. 
#
# The choice of the migrating node within the boundary of the origin-district depends on the heuristic at hand. The Descent algorithm and 
# the first of the Tabu Search implementations make use of exact neighborhoods, that is, they look at all the feasible solutions in the neighborhood 
# (equivalently, all the nodes in the boundary) in order to find the best one.

def exact_neighborhood(graph, clusters):  # exact neighborhood is defined for Descent Search and the first Tabu Search implementation.  

    #  migrating node: origin district --> destination district
    #print("--------- ENTERING EXACT NEIGHBORHOOD ---------")
    boundaries = {}  # key: (origin's center, a neighboring cluster's center),  value: nodes in the boundary of origin, and adjacent to the neighboring cluster. Origin is fixed.
    neighborhood = {}  # key: migrating node, value: a dictionary of clusters (clusters -> key: center, value: list of nodes in the cluster of center) 
    
    centers = [center for center in clusters.keys() if len(clusters[center]) > 1] # list of district centers such that discrits have more than 1 node. (Origin district must have at least 2 nodes.)
    #centers = [center for center, nodes in clusters.items() if len(nodes) > 1]
    origin = random.choice(centers)    # choose origin district of migrating node uniformly random.
    #print("origin:", origin)
    #print("clusters[origin] right after choosing origin randomly:", clusters[origin])


    for cluster in clusters.keys():  # each cluster is a candidate of destination. 
        if cluster != origin:
            boundaries[(origin, cluster)] = list(nx.node_boundary(graph, clusters[cluster], clusters[origin])) # return a list of nodes in clusters[cluster] that are adjacent to some nodes in clusters[origin]

            for migrating_node in boundaries[(origin, cluster)]:
                if migrating_node != origin:
                    #print("migrating_node=", migrating_node)
                    new_clusters = copy.deepcopy(clusters)
                    new_clusters[origin].remove(migrating_node)
                    new_clusters[cluster].append(migrating_node)
                    subgraph = nx.induced_subgraph(graph, new_clusters[origin])
                    #print("number of components in origin= ", len(list(nx.connected_components(subgraph))))
                    if len(list(nx.connected_components(subgraph))) == 1:       # remove migrating node from its cluster and check if the cluster is connected
                        #print("subgraph has only 1 component")
                        #print("clusters[origin]:", clusters[origin])
                        #print("migrating node:", migrating_node)
                        #print("new_clusters[origin]:", new_clusters[origin])
                        #print("number of components in origin - after = ", len(list(nx.connected_components(subgraph))))

                        neighborhood[migrating_node] = new_clusters
                    else:
                        continue
    #                    print("nodes=", subgraph.nodes)
    #                    if all(nx.has_path(subgraph, u, v)==True for u, v in subgraph.nodes):
    #                        new_clusters[cluster].append(migrating_node)
    #                        neighborhood[migrating_node] = new_clusters
        else: 
            continue

    #print("--------- LEAVING EXACT NEIGHBORHOOD ---------")

    return origin, boundaries, neighborhood

# Note: Migrating note might be a facility in a district with 2 nodes. If the other node is a stop node, new cluster will not have any population and facility.



def random_neighbor(graph, clusters):

    #  migrating node: origin district --> destination district
    boundaries = {}  # key: a center,  value: list of nodes in the boundary of center's cluster that are not cut-vertex in their clusters
    neighborhood = {} # key: migrating node, value: a dictionary of clusters (clusters -> key: center, value: list of nodes in the cluster of center) 
    candidates = []

    seeds = list(clusters.keys())
 
    for center in seeds:
        if  len(clusters[center]) > 1:
            for center2 in seeds:
                if center2 != center:
                    boundaries[(center, center2)] = list(nx.node_boundary(graph, clusters[center], clusters[center2])) # return a list of nodes in clusters[center] that are adjacent to some nodes in other clusters
                    for node in boundaries[(center, center2)]:
                        new_cluster = copy.deepcopy(clusters[center])
                        new_cluster.remove(node)
                        subgraph = nx.induced_subgraph(graph, new_cluster)
                        if len(list(nx.connected_components(subgraph))) == 1:      
                            candidates.append((center, center2, node))


    random_pair = random.choice(candidates)
    migrating_node = random_pair[2]
    origin = random_pair[0]
    destination = random_pair[1]

    clusters[origin].remove(migrating_node)
    clusters[destination].append(migrating_node)


    return clusters

# Ricca uses three objective functions. Population equality, compactness, conformity to administrative boundaries. 
# We take only one of them: populuation equality. We need one more objective function: accessibility. 
# Let $\Pi_k(G)$ be the collection of all connected partitions of $G$.
#
# For a clustering $\pi in \Pi_k(G)$, let $p_i, \bar{p}$ be the population of district $i$ and average district population, respectively.
# The objective function for the population equality condition is $f_1(\pi) = \dfrac{\sum\limits_{j=1}^{k} |p_i - \bar{p}|}{k\bar{p}}$,
# where $k$ is the number of clusters in $\pi$. $f_1(\pi)$ takes values in the interval $[0, \dfrac{2(k-1)}{k}].$
#
# Let $d_{ij}$ be the travel time betweens$i$ and $j$. For a given district $C$ centered at $c$, let $d(C) = \text{max}\limits_{i \in C} d_{ic}$
# be the travel radius of the district. Denote average travel radius by $\bar{d}$. The objective function for accessibility is 
# $f_2(\pi) = \dfrac{\sum\limits_{C \in \pi} |d(C) - \bar{d}|}{k\bar{d}}$.
#
# We want to solve the vector-minimization problem $\text{min}\limits_{\pi \in \Pi_k(G)} \{f_1(\pi), f_2(\pi)\}$.
# The following objective function returns $f_1(\pi)$ and $f_2(\pi)$  functions for a given $\pi$.

def objective_function(grid, clusters, travel, graph):

    #print("--------- ENTERING OBJECTIVE FUNCTION ---------")

    populations = {}
    travel_radius = {}
    graph_radius = {}
    num_districts = len(clusters)

    for cluster in clusters.keys():
        #print("picked a cluster with the center", cluster)
        #print("nodes in the cluster:", clusters[cluster])
        populations[cluster] = sum(grid[node].get_node_population() for node in clusters[cluster])
        #print("total popuation of the cluster is", populations[cluster])
        travel_times = [travel[node, cluster] for node in clusters[cluster]]
        #print("travel times from the center are", travel_times)
        travel_radius[cluster] = max(travel[node, cluster] for node in clusters[cluster])
        #print("travel radius of the cluster is", travel_radius[cluster])
        #print("end of the loop for the cluster")
        
        ##subgraph = nx.induced_subgraph(graph, clusters[cluster])
        ##graph_radius[cluster] = nx.radius(subgraph)

    pop_average = sum(populations.values()) / num_districts
    radius_average = sum(travel_radius.values()) / num_districts
    ##graph_radius_average = sum(graph_radius.values()) / num_districts
    ##+ abs(graph_radius[cluster] - graph_radius_average)**2 / (num_districts * graph_radius_average)

    f_1 = sum((abs(populations[cluster] - pop_average) / (num_districts * pop_average)) for cluster in clusters.keys())
    f_2 = sum((abs(travel_radius[cluster] - radius_average)**2 / (num_districts * radius_average)) for cluster in clusters.keys())

    #print("--------- LEAVING OBJECTIVE FUNCTION ---------")

    return f_1, f_2


def objective_function_ricca(grid, clusters, travel, graph):

    #print("--------- ENTERING OBJECTIVE FUNCTION ---------")

    populations = {}
    travel_radius = {}
    graph_radius = {}
    num_districts = len(clusters)

    for cluster in clusters.keys():
        #print("picked a cluster with the center", cluster)
        #print("nodes in the cluster:", clusters[cluster])
        populations[cluster] = sum(grid[node].get_node_population() for node in clusters[cluster])
        #print("total popuation of the cluster is", populations[cluster])
        travel_times = [travel[node, cluster] for node in clusters[cluster]]
        #print("travel times from the center are", travel_times)
        travel_radius[cluster] = max(travel[node, cluster] for node in clusters[cluster])
        #print("travel radius of the cluster is", travel_radius[cluster])
        #print("end of the loop for the cluster")
        
        subgraph = nx.induced_subgraph(graph, clusters[cluster])
        graph_radius[cluster] = nx.radius(subgraph)

    pop_average = sum(populations.values()) / num_districts
    radius_average = sum(travel_radius.values()) / num_districts
    graph_radius_average = sum(graph_radius.values()) / num_districts


    f_1 = sum((abs(populations[cluster] - pop_average) / (num_districts * pop_average)) for cluster in clusters.keys())
    f_2 = sum((abs(travel_radius[cluster] - radius_average)**2 / (num_districts * radius_average)) + abs(graph_radius[cluster] - graph_radius_average)**2 / (num_districts * graph_radius_average) for cluster in clusters.keys())

    #print("--------- LEAVING OBJECTIVE FUNCTION ---------")

    return f_1, f_2
# Multi-start Descent Search

#   Solving this problem consists in finding some Pareto-optimal connected $k$-partition of $G$. 
#   Given a multiobjective program with $m$ obj functions, a solution is locally Pareto-optimal 
#   if there is no local perturbation that makes an objective improve without worsening some other objective.
#
#   We adjust the local search for the multiobjective minimization as follows. Consider $f_1$ as the "target" function 
#   (we do not minimize $f_2$ directly). $\alpha \geq 0$ of the max acceptable worsening for $f_2$ is fixed. 
#   If a solution $\pi$ changes into $\pi^{\prime}$, then $f_{2}(\pi^{\prime}) \geq (1-\alpha)f_2(\pi)$ must hold.
#
#   Let $N(\pi)$ be the set of solutions in the neighborhood of $\pi$ and $D(\pi^{\prime}) = f(\pi^{\prime}) - f(\pi)$, $\pi^{\prime} \in N(\pi)$. 
#   We call $\pi^{\star}$ a best solution in $N(\pi)$ if we have $D(\pi^{\star}) = \text{min}\limits_{\pi^{\prime} \in $N(\pi)$} D(\pi^{\prime})$.
#
#   Starting from an initial feasible solution, at each iteration the Descent algorithm searches for a best solution in the neighborhood of the current solution. 
#   The Descent algorithm stops when there are no more feasible moves that make the objective function improve. The algorithm often remains entrapped in a bad local optima. 
#   Hence, we use a multiple start approach (Multistart Descent algorithm) according to which the search is re-started from a new initial (random) solution.
#
#   Let $M$ denote the total number of iterations. In general, when $M$ is increased, the performance of the algorithm improves due to the fact that many additional solutions 
#   are evaluated. Indeed, if $M$ is very large, the algorithm can occasionally perform even better than other local search algorithms. Ricca et. al. takes $M = 80,000$.
#   Because, $M = 80,000$ has shown to be sufficient to guarantee that all the heuristic achieve their best solution.

def multistart_descent_search(grid, graph, travel, open_facilities, alpha, num_iterations):

    iteration_results = {}

    iteration = 0

    while iteration < num_iterations:

                #tree, initial_solution, population = generate_initial_partition(G, grid, open_facilities)
        partitioned_tree, spanning_tree, initial_solution, populations = generate_initial_partition(graph, grid, open_facilities)
        initial_energy_pop, initial_energy_access = objective_function(grid, initial_solution, travel, graph)
        
        current_solution = initial_solution
        current_energy_pop = initial_energy_pop
        current_energy_access = initial_energy_access

        if iteration % 500 == 0:
            print("iteration = ", iteration)


        # Descent algorithm for the current iteration. As long as we find a better solution, while loop continues. 
        while True:  

            # Generate the exact neighborhood of the current solution
            origin, boundaries, neighborhood = exact_neighborhood(graph, current_solution)
            #print("neighborhood= ", neighborhood)
            
            # Save the objective values f_1 and f_2 of the neighbors that have max acceptable worsening for f_2 (access function).
            neighbor_energies = {}
            for migrating_node in neighborhood.keys():
                neighbor_energy_pop, neighbor_energy_access = objective_function(grid, neighborhood[migrating_node], travel, graph)
                if neighbor_energy_access <= (1 + alpha) * current_energy_access:
                    neighbor_energies[migrating_node] = (neighbor_energy_pop, neighbor_energy_access)
            
            if len(neighbor_energies) == 0:
                break 

            # Take the best solution in the neighborhood, i.e., migrating node corresponding to the minimum f_1 (population function).
            best_migration = min((neighbor_energies[migrating_node][0], migrating_node) for migrating_node in neighbor_energies.keys())[1]

            # if the best solution in the neighborhood is better than the current solution, accept it.
            if neighbor_energies[best_migration][0] < current_energy_pop:
                current_solution = neighborhood[best_migration]
                current_energy_pop = neighbor_energies[best_migration][0]
                current_energy_access = neighbor_energies[best_migration][1]
            
            # Otherwise, stop the iteration. 
            else:
                break
    
        # Save the result of the curent iteration. 
        iteration_results[iteration] = (initial_solution, current_solution, current_energy_pop, current_energy_access, initial_energy_pop, initial_energy_access)

        iteration += 1

    # Initialize the best iteration as the first iteration
    current_iteration_initial = iteration_results[0][0]
    current_iteration_solution = iteration_results[0][1]
    current_iteration_energy_pop = iteration_results[0][2]
    current_iteration_energy_access = iteration_results[0][3]
    current_iteration_initial_pop = iteration_results[0][4]
    current_iteration_initial_access = iteration_results[0][5]

    # Compare the results of iterations
    for iteration in range(num_iterations - 1):

        if iteration_results[iteration + 1][3] <= (1 + alpha) * current_iteration_energy_access and iteration_results[iteration + 1][2] < current_iteration_energy_pop:
            current_iteration_initial = iteration_results[iteration + 1][0]
            current_iteration_solution = iteration_results[iteration + 1][1]
            current_iteration_energy_pop = iteration_results[iteration + 1][2]
            current_iteration_energy_access = iteration_results[iteration + 1][3]
            current_iteration_initial_pop = iteration_results[iteration + 1][4]
            current_iteration_initial_access = iteration_results[iteration + 1][5]

    # Obtain required information for plotting
    #current_iteration_initial_origin, current_iteration_initial_boundaries, current_iteration_initial_neighborhood = exact_neighborhood(graph, current_iteration_initial)
    #current_iteration_solution_origin, current_iteration_solution_boundaries, current_iteration_solution_neighborhood = exact_neighborhood(graph, current_iteration_initial)

    # Create partitioned graph for initial of current solution
    #initial_component = []
    #for center in current_iteration_initial.keys():
    #    initial_component.append(nx.induced_subgraph(spanning_tree, current_iteration_initial[center]))
    
    #initial_partitioned_tree = nx.union(initial_component[0], initial_component[1])
    #for index, graph in enumerate(initial_component):
    #    if index > 1:
    #        initial_partitioned_tree = nx.union(initial_partitioned_tree, graph)

    # Create partitioned graph for the current solution
    #    solution_component = []
    #for center in current_iteration_solution.keys():
    #    solution_component.append(nx.induced_subgraph(spanning_tree, current_iteration_solution[center]))
    
    #solution_partitioned_tree = nx.union(solution_component[0], solution_component[1])
    #for index, graph in enumerate(solution_component):
    #    if index > 1:
    #        solution_partitioned_tree = nx.union(solution_partitioned_tree, graph)

    return iteration_results, current_iteration_initial, current_iteration_solution, current_iteration_energy_pop, current_iteration_energy_access, current_iteration_initial_pop, current_iteration_initial_access


def multistart_descent_search_obj2(grid, graph, travel, open_facilities, alpha, num_iterations):

    iteration_results = {}

    iteration = 0

    while iteration < num_iterations:

                #tree, initial_solution, population = generate_initial_partition(G, grid, open_facilities)
        partitioned_tree, spanning_tree, initial_solution, populations = generate_initial_partition(graph, grid, open_facilities)
        initial_energy_pop, initial_energy_access = objective_function_ricca(grid, initial_solution, travel, graph)
        
        current_solution = initial_solution
        current_energy_pop = initial_energy_pop
        current_energy_access = initial_energy_access

        if iteration % 500 == 0:
            print("iteration = ", iteration)


        # Descent algorithm for the current iteration. As long as we find a better solution, while loop continues. 
        while True:  

            # Generate the exact neighborhood of the current solution
            origin, boundaries, neighborhood = exact_neighborhood(graph, current_solution)
            #print("neighborhood= ", neighborhood)
            
            # Save the objective values f_1 and f_2 of the neighbors that have max acceptable worsening for f_2 (access function).
            neighbor_energies = {}
            for migrating_node in neighborhood.keys():
                neighbor_energy_pop, neighbor_energy_access = objective_function_ricca(grid, neighborhood[migrating_node], travel, graph)
                if neighbor_energy_access <= (1 + alpha) * current_energy_access:
                    neighbor_energies[migrating_node] = (neighbor_energy_pop, neighbor_energy_access)
            
            if len(neighbor_energies) == 0:
                break 

            # Take the best solution in the neighborhood, i.e., migrating node corresponding to the minimum f_1 (population function).
            best_migration = min((neighbor_energies[migrating_node][0], migrating_node) for migrating_node in neighbor_energies.keys())[1]

            # if the best solution in the neighborhood is better than the current solution, accept it.
            if neighbor_energies[best_migration][0] < current_energy_pop:
                current_solution = neighborhood[best_migration]
                current_energy_pop = neighbor_energies[best_migration][0]
                current_energy_access = neighbor_energies[best_migration][1]
            
            # Otherwise, stop the iteration. 
            else:
                break
    
        # Save the result of the curent iteration. 
        iteration_results[iteration] = (initial_solution, current_solution, current_energy_pop, current_energy_access, initial_energy_pop, initial_energy_access)

        iteration += 1

    # Initialize the best iteration as the first iteration
    current_iteration_initial = iteration_results[0][0]
    current_iteration_solution = iteration_results[0][1]
    current_iteration_energy_pop = iteration_results[0][2]
    current_iteration_energy_access = iteration_results[0][3]
    current_iteration_initial_pop = iteration_results[0][4]
    current_iteration_initial_access = iteration_results[0][5]

    # Compare the results of iterations
    for iteration in range(num_iterations - 1):

        if iteration_results[iteration + 1][3] <= (1 + alpha) * current_iteration_energy_access and iteration_results[iteration + 1][2] < current_iteration_energy_pop:
            current_iteration_initial = iteration_results[iteration + 1][0]
            current_iteration_solution = iteration_results[iteration + 1][1]
            current_iteration_energy_pop = iteration_results[iteration + 1][2]
            current_iteration_energy_access = iteration_results[iteration + 1][3]
            current_iteration_initial_pop = iteration_results[iteration + 1][4]
            current_iteration_initial_access = iteration_results[iteration + 1][5]

    # Obtain required information for plotting
    #current_iteration_initial_origin, current_iteration_initial_boundaries, current_iteration_initial_neighborhood = exact_neighborhood(graph, current_iteration_initial)
    #current_iteration_solution_origin, current_iteration_solution_boundaries, current_iteration_solution_neighborhood = exact_neighborhood(graph, current_iteration_initial)

    # Create partitioned graph for initial of current solution
    #initial_component = []
    #for center in current_iteration_initial.keys():
    #    initial_component.append(nx.induced_subgraph(spanning_tree, current_iteration_initial[center]))
    
    #initial_partitioned_tree = nx.union(initial_component[0], initial_component[1])
    #for index, graph in enumerate(initial_component):
    #    if index > 1:
    #        initial_partitioned_tree = nx.union(initial_partitioned_tree, graph)

    # Create partitioned graph for the current solution
    #    solution_component = []
    #for center in current_iteration_solution.keys():
    #    solution_component.append(nx.induced_subgraph(spanning_tree, current_iteration_solution[center]))
    
    #solution_partitioned_tree = nx.union(solution_component[0], solution_component[1])
    #for index, graph in enumerate(solution_component):
    #    if index > 1:
    #        solution_partitioned_tree = nx.union(solution_partitioned_tree, graph)

    return iteration_results, current_iteration_initial, current_iteration_solution, current_iteration_energy_pop, current_iteration_energy_access, current_iteration_initial_pop, current_iteration_initial_access




# Tabu Search 

#  1.   Select an initial feasible solution s
#  2.   repeat:
#          generate the set of all feasible moves producing the corresponding
#          set of feasible solutions in the neighborhood N(s) of the current solution s.
#  2.1.    If there is at least a feasible non-tabu move
#              select a feasible non-tabu move leading to a best solution s' ∈ N(s).
#              update the tabu-list.
#  2.2.    else [all possible moves are either infeasible or tabu]
#              STOP (a local optimum is found)
#  2.3.    end if
#  3.   until the stopping condition is met
#  4.   The final solution is the best local optimum found s*


def tabu_search_random():
    return


# Old Bachelor Acceptance 

# At each step a threshold value specifies the maximum acceptable change in the objective function.
# When the algorithm goes from a solution s to a new solution s' in the neighborhood of s, 
# the objective function may improve, otherwise it may worsen within the threshold limit as in Descent Method. 
# However, each time the threshold is automatically updated in a non-monotonic way, with even the possibility that it reaches negative values.

# In particular, the threshold decreases after an improvement in the objective function and increases when the objective function worsens. 
# Such an updating strategy has been shown to be an efficient way to avoid premature arrests in bad local optima.
# When the threshold changes, the search has the possibility of finding new promising Descent directions, especially when the current solution is far from a local optimum.

# The algorithm has the ability to escape from bad local minima by increasing the threshold value, while, during the ambitious phase, 
# successive threshold decreases cause the algorithm to accelerate along steep Descents towards a local minimum point.

# Δ^{+}(i) and Δ^{-}(i) are both positive and they are involved in the threshold updating, while the stopping condition is a fixed number of total iterations.
# In our application, Δ^{+}(i) and Δ^{-}(i) are linear functions of the quantity (1 - i/M), where i denotes the current iteration. 
# When the last steps of the algorithm are performed – that means a low value for (1 - i/M) – both Δ^{+}(i) and Δ^{-}(i) become small. 
# This corresponds to a strategy of maximum exploitation of the last iterations in order to find some additional local optima.
# Updating of the thresholds D+(i) and D􏰀(i) is symmetric in order to avoid undesired unbalanced threshold adjustments.

# After each failure, the criterion for acceptability is relaxed by slightly increasing the threshold (this motivates the name Old Bachelor Acceptance). 
# After sufficiently many consecutive failures, the threshold will become large enough for OBA to escape the current local minimum. 
# The converse of dwindling expectations is what we call ambition, whereby after each acceptance of s', the threshold is lowered so that 
# OBA becomes more aggressive in moving toward a local minimum.

# An extremely high threshold prevents from any move and may provoke the algorithm to find bad local optima. 
# On the other hand, a threshold which decreases too fast tends to become negative, thus producing a premature stop of the algorithm. 
# This is similar to what happens with the Descent algorithm, since in this case it becomes very difficult to find good moves.
# We need to calibrate Δ^{+}(i) and Δ^{-}(i) in order to allow a long and diversified search, increasing the chance of finding many different local optima. 
# If the final set of encountered local optima is large, the final solution is likely to be a good one.

# The algorithms start from a random initial solution which is generated by selecting a spanning tree for the graph G at random 
# and obtaining k subtrees by the random selection of their roots as in Descent Algorithm. They explore the trees in breadth-first search order.
# They also consider a mixture of the objective functions. In this case, the objective function is given by a convex combination of the objectives 
# with weights equal to 0.5 for population equality, 0.3 for compactness and 0.2 for conformity, according to the relative importance.

# Old Bachelor Acceptance is characterized by a parameter called granularity. We observed that very low values are necessary to guarantee a good performance.
# If the target is compactness, granularity takes values of magnitude 10^{-2} or 10^{-3}. When conformity to administrative boundaries is considered, 
# granularity becomes much smaller 10^{-7}, but it reaches the smallest value when population equality is the target criterion 10^{-8}. 
# For the target mixture we obtain the best performance with a granularity value of magnitude 10^{-6}.

# To guarantee the robustness of these results, one must ensure a good quality of the final solution not only in a single (lucky) run, 
# but in a large percentage of cases. In this sense, the robustness of the results is guaranteed by the fact that 
# all the variances associated with these repeated runs are always very small (order of 10^{-3}).

#  1.    Select an initial feasible solution s 
#  2.    Define an initial threshold T_1
#  3.    repeat:
#           select at random a feasible move producing a neighborhood feasible solution s' of the current solution s
# 3.1.      if Δ(s') < T_i
#               perform the move
# 3.1.1.        if Δ(s') < 0  decrease the threshold: T_{i+1}:= T_i − Δ^{-}(i)
# 3.2.      otherwise [ Δ(s') ≥ T_i ]
#               increase the threshold: T_{i+1}:= T_i + Δ^{+}(i) 
# 3.3.      end if
# 4.    until the stopping condition is met
# 5.    The final solution is the best local optimum found s *


def old_bachelor(graph, grid, open_facilities, travel, num_iterations, granularity, a, b, c, alpha):


    # 1. Generate the same initial solution as in Descent. Change the objective ?
    partitioned_tree, spanning_tree, initial_solution, initial_populations = generate_initial_partition(graph, grid, open_facilities)
    initial_energy_pop, initial_energy_access = objective_function(grid, initial_solution, travel, graph) 
    current_solution = initial_solution
    current_energy_pop = initial_energy_pop
    current_energy_access = initial_energy_access

    # 2. Define an initial threshold T_0.
    threshold = 0
    age = 0
    iteration = 0

    # 3. num_iterations = M in the explonation above. Note that i /leq M-1, and so (1 - i/M) > 0.
    while iteration < num_iterations:

        if iteration % 500 == 0:
            print("iteration = ", iteration)
        
        # Select at random a feasible move
        origin, boundaries, neighborhood = exact_neighborhood(graph, current_solution)
        migrating = random.choice(list(neighborhood.keys()))
        neighbor = neighborhood[migrating]
        #neighbor = random_neighbor(graph, current_solution) 
        neighbor_energy_pop, neighbor_energy_access = objective_function(grid, neighbor, travel, graph)

        # 3.1. if energy change < T_i, perform the move.
        if neighbor_energy_access <= (1 + alpha) * current_energy_access and neighbor_energy_pop - current_energy_pop < threshold:  
            current_solution = neighbor
            current_energy_pop = neighbor_energy_pop
            current_energy_access = neighbor_energy_access

            # 3.1.1. if energy change < 0  decrease the threshold: T_{i+1}:= T_i − Δ^{-}(i). --> Only close bad moves will be accepted, since we just found a good solution.
            if neighbor_energy_pop - current_energy_pop < 0:
                age = 0
                threshold = ( ( age / a ) ** b - 1) * granularity * (1 - iteration / num_iterations ) ** c 

        # 3.2. Otherwise, Δ >= T_i, increase the threshold: T_{i+1}:= T_i + Δ^{+}(i) --> We should accept worst solutions increasing T_i, since we may be trapped at a local min.
        else: 
            age += 1
            threshold = ( (age / a ** b - 1) * granularity * (1 - iteration / num_iterations ) ** c )
        
        # increase the iteration number 
        iteration += 1

    # 5. The final solution is the best local optimum found s *
    return initial_solution, initial_populations, initial_energy_pop, initial_energy_access, current_solution, current_energy_pop, current_energy_access





def multi_old_bachelor(graph, grid, open_facilities, travel, num_iterations, granularity, a, b, c, alpha, num_inner_iterations):

    iteration_results = {}
    iteration = 0

    # 3. num_iterations = M in the explonation above. Note that i /leq M-1, and so (1 - i/M) > 0.
    while iteration < num_iterations:

        
            # 1. Generate the same initial solution as in Descent. Change the objective ?
        partitioned_tree, spanning_tree, initial_solution, initial_populations = generate_initial_partition(graph, grid, open_facilities)
        initial_energy_pop, initial_energy_access = objective_function(grid, initial_solution, travel, graph) 
        current_solution = initial_solution
        current_energy_pop = initial_energy_pop
        current_energy_access = initial_energy_access

        # 2. Define an initial threshold T_0.
        threshold = 0
        age = 0

        inner_iteration = 0


        while inner_iteration < num_inner_iterations:

            if inner_iteration % 250 == 0:
                print("inner_iteration = ", inner_iteration)
            # Select at random a feasible move
            origin, boundaries, neighborhood = exact_neighborhood(graph, current_solution)
            migrating = random.choice(list(neighborhood.keys()))
            neighbor = neighborhood[migrating]
            #neighbor = random_neighbor(graph, current_solution) 
            neighbor_energy_pop, neighbor_energy_access = objective_function(grid, neighbor, travel, graph)

            # 3.1. if energy change < T_i, perform the move.
            if neighbor_energy_access <= (1 + alpha) * current_energy_access and neighbor_energy_pop - current_energy_pop < threshold:   # worsening?
                current_solution = neighbor
                current_energy_pop = neighbor_energy_pop
                current_energy_access = neighbor_energy_access

                # 3.1.1. if energy change < 0  decrease the threshold: T_{i+1}:= T_i − Δ^{-}(i). --> Only close bad moves will be accepted, since we just found a good solution.
                if neighbor_energy_pop - current_energy_pop < 0:
                    age = 0
                    threshold = ( ( age / a ) ** b - 1) * granularity * (1 - iteration / num_iterations ) ** c 

            # 3.2. Otherwise, Δ >= T_i, increase the threshold: T_{i+1}:= T_i + Δ^{+}(i) --> We should accept worst solutions increasing T_i, since we may be trapped at a local min.
            else: 
                age += 1
                threshold = ( (age / a ** b - 1) * granularity * (1 - iteration / num_iterations ) ** c )

            inner_iteration += 1
        
        # Save the result of the curent iteration. 
        iteration_results[iteration] = (initial_solution, current_solution, current_energy_pop, current_energy_access, initial_energy_pop, initial_energy_access)

        # increase the iteration number 
        iteration += 1


    # Initialize the best iteration as the first iteration
    current_iteration_initial = iteration_results[0][0]
    current_iteration_solution = iteration_results[0][1]
    current_iteration_energy_pop = iteration_results[0][2]
    current_iteration_energy_access = iteration_results[0][3]
    current_iteration_initial_pop = iteration_results[0][4]
    current_iteration_initial_access = iteration_results[0][5]

    # Compare the results of iterations
    for iteration in range(num_iterations - 1):

        if iteration_results[iteration + 1][3] <= (1 + alpha) * current_iteration_energy_access and iteration_results[iteration + 1][2] < current_iteration_energy_pop:
            current_iteration_initial = iteration_results[iteration + 1][0]
            current_iteration_solution = iteration_results[iteration + 1][1]
            current_iteration_energy_pop = iteration_results[iteration + 1][2]
            current_iteration_energy_access = iteration_results[iteration + 1][3]
            current_iteration_initial_pop = iteration_results[iteration + 1][4]
            current_iteration_initial_access = iteration_results[iteration + 1][5]

    # 5. The final solution is the best local optimum found s *
    return iteration_results, current_iteration_initial, current_iteration_solution, current_iteration_energy_pop, current_iteration_energy_access, current_iteration_initial_pop, current_iteration_initial_access