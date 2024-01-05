import networkx as nx
import random



# Explain how to sample the tree.

def sample_tree(G):

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
    
    return tree



# Generate an initial partition / solution

#     Initialize set of clusters: $C = \{ \}$.  (Dictionary. key: cluster name, value: list of vertices)
#    Determine the set of the connected components in $G$, $\mathcal{G}=\{G_1, G_2, ..., G_k\}$. 
#    Repeat until $\mathcal{G} = \emptyset$.
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

    C = {}  #  key: cluster name,  value: list of vertices
    populations = {}

    spanning_tree = sample_tree(G)
    tree = spanning_tree.copy(as_view=False)

    while len(tree.nodes) > 0:  
        S = list(nx.connected_components(tree))
        
        for component in S:
            facilities_in_component = list(set(open_facilities).intersection(component))

            if len(facilities_in_component) == 1:
                C[facilities_in_component[0]] = list(component)
                tree.remove_nodes_from(component)
            else:   
                u = random.choice(facilities_in_component)
                paths = {v: nx.shortest_path(tree, source=u, target=v) for v in facilities_in_component if v != u}
                closest = min((len(paths[v]), v) for v in paths.keys())[1]
                path = tree.subgraph(paths[closest])
                edge = random.choice(list(path.edges))
                tree.remove_edge(*edge)
    
    for center in C.keys():
        populations[center] = 0

        for node in C[center]:
            populations[center] = populations[center] + grid[node].get_node_population()

    # Create the partitioned graph
    component = []
    for center in C.keys():
        component.append(nx.induced_subgraph(spanning_tree, C[center]))
    
    partitioned_tree = nx.union(component[0], component[1])
    for index, graph in enumerate(component):
        if index > 1:
            partitioned_tree = nx.union(partitioned_tree, graph)


    return partitioned_tree, spanning_tree, C, populations


def generate_initial_solution1(G, grid, open_facilities):
    
    clusters = {}  #  key: cluster name,  value: list of vertices
    partitioned_tree = nx.empty_graph(0)
    populations = {}
    components = []

    spanning_tree = sample_tree(G)
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
                print(facilities)

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
# The origin-district is chosen at random.
# We define the boundary of the origin district. Then we check if removing a node in the boundary makes the origin district disconnected.
# If the district is still connected, we move the migrating node safely and save the new clustering as a neighbor.
# The exact neighborhood (all neighbors) is determined. 
#
# The choice of the migrating node within the boundary of the origin-district depends on the heuristic at hand. The Descent algorithm and 
# the first of the Tabu Search implementations make use of exact neighborhoods, that is, they look at all the feasible solutions in the neighborhood 
# (equivalently, all the nodes in the boundary) in order to find the best one.

def exact_neighborhood(tree, clusters):  # exact neighborhood is defined for Descent Search and the first Tabu Search implementation.  

    #  migrating node: origin district --> destination district

    boundaries = {}  # key: (origin, a cluster),  value: nodes in the boundary of origin, and adjacent to the cluster. Origin is fixed.
    neighborhood = {}  # key: migrating node, value: a dictionary of clusters (clusters -> key: center, value: list of nodes in the cluster of center) 
    
    #centers = [center for center in clusters.keys() if len(clusters[center]) > 1] # list of district centers such that discrits have more than 1 node.
    centers = [center for center, nodes in clusters.items() if len(nodes) > 1]
    origin = random.choice(centers)    # choose origin district of migrating node uniformly random.


    for cluster in clusters.keys():  # each cluster is a candidate of destination. 
        if cluster != origin:
            boundaries[(origin, cluster)] = list(nx.node_boundary(tree, clusters[cluster], clusters[origin]))

            for migrating_node in boundaries[(origin, cluster)]:
                print("migrating_node=", migrating_node)
                new_clusters = clusters.copy()
                new_clusters[origin].remove(migrating_node)
                subgraph = nx.induced_subgraph(tree, new_clusters[origin])
                print("number of components in origin= ", len(list(nx.connected_components(subgraph))))
                if len(list(nx.connected_components(subgraph))) == 1:       # remove migrating node from its cluster and check if the cluster is connected
                    print("number of components in origin - after = ", len(list(nx.connected_components(subgraph))))
                    new_clusters[cluster].append(migrating_node)

                    neighborhood[migrating_node] = new_clusters
#                else:
#                    print("nodes=", subgraph.nodes)
#                    if all(nx.has_path(subgraph, u, v)==True for u, v in subgraph.nodes):
#                        new_clusters[cluster].append(migrating_node)
#                        neighborhood[migrating_node] = new_clusters

    return origin, boundaries, neighborhood



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

def objective_function(grid, clusters, travel):

    populations = {}
    travel_radius = {}
    num_districts = len(clusters)

    for cluster in clusters.keys():
        populations[cluster] = sum(grid[node].get_node_population() for node in clusters[cluster])
        travel_radius[cluster] = max(travel[node, cluster] for node in clusters[cluster])

    pop_average = sum(populations.values()) / num_districts
    radius_average = sum(travel_radius.values()) / num_districts

    f_1 = sum((abs(populations[cluster] - pop_average) / (num_districts * pop_average)) for cluster in clusters.keys())
    f_2 = sum((abs(travel_radius[cluster] - radius_average) / (num_districts * radius_average)) for cluster in clusters.keys())
    
    return f_1, f_2



# Solving this problem consists in finding some Pareto-optimal connected $k$-partition of $G$. 
# Given a multiobjective program with $m$ obj functions, a solution is locally Pareto-optimal 
# if there is no local perturbation that makes an objective improve without worsening some other objective.
#
# We adjust the local search for the multiobjective minimization as follows. Consider $f_1$ as the "target" function 
# (we do not minimize $f_2$ directly). $\alpha \geq 0$ of the max acceptable worsening for $f_2$ is fixed. 
# If a solution $\pi$ changes into $\pi^{\prime}$, then $f_{2}(\pi^{\prime}) \geq (1-\alpha)f_2(\pi)$ must hold.
#
# Let $N(\pi)$ be the set of solutions in the neighborhood of $\pi$ and $D(\pi^{\prime}) = f(\pi^{\prime}) - f(\pi)$, $\pi^{\prime} \in N(\pi)$. 
# We call $\pi^{\star}$ a best solution in $N(\pi)$ if we have $D(\pi^{\star}) = \text{min}\limits_{\pi^{\prime} \in $N(\pi)$} D(\pi^{\prime})$.
#
# Starting from an initial feasible solution, at each iteration the Descent algorithm searches for a best solution in the neighborhood of the current solution. 
# The Descent algorithm stops when there are no more feasible moves that make the objective function improve. The algorithm often remains entrapped in a bad local optima. 
# Hence, we use a multiple start approach (Multistart Descent algorithm) according to which the search is re-started from a new initial (random) solution.
#
# Let $M$ denote the total number of iterations. In general, when $M$ is increased, the performance of the algorithm improves due to the fact that many additional solutions 
# are evaluated. Indeed, if $M$ is very large, the algorithm can occasionally perform even better than other local search algorithms. Ricca et. al. takes $M = 80,000$.
# Because, $M = 80,000$ has shown to be sufficient to guarantee that all the heuristic achieve their best solution.

def multistart_descent_search(grid, G, travel, open_facilities, alpha, num_iterations):

    iteration_results = {}

    for iteration in range(num_iterations):

        if iteration % 500 == 0:
            print("iteration = ", iteration)

        tree, initial_solution, population = generate_initial_partition(G, grid, open_facilities)
        current_solution = initial_solution
        current_energy_pop, current_energy_access = objective_function(grid, current_solution, travel)

        # Descent algorithm for the current iteration
        while True:  

            # Generate the exact neighborhood of the current solution
            neighborhood = exact_neighborhood(G, current_solution)
            print("neighborhood= ", neighborhood)
            
            # Save the objective value f_1 of the neighbors that have max acceptable worsening for f_2.
            neighbor_energies = {}
            for migrating_node in neighborhood.keys():
                neighbor_energy_pop, neighbor_energy_access = objective_function(grid, neighborhood[migrating_node], travel)
                if neighbor_energy_access <= (1 + alpha) * current_energy_access:
                    neighbor_energies[migrating_node] = (neighbor_energy_pop, neighbor_energy_access)
            
            if len(neighbor_energies) == 0:
                break 

            # Take the best solution in the neighborhood
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
        iteration_results[iteration] = (initial_solution, current_solution, current_energy_pop, current_energy_access)

    # Initialize the best iteration as the first iteration
    current_iteration_initial = iteration_results[0][0]
    current_iteration_solution = iteration_results[0][1]
    current_iteration_energy_pop = iteration_results[0][2]
    current_iteration_energy_access = iteration_results[0][3]

    # Compare the results of iterations
    for iteration in range(num_iterations - 1):

        if iteration_results[iteration + 1][3] <= (1 + alpha) * current_iteration_energy_access and iteration_results[iteration + 1][2] < current_iteration_energy_pop:
            current_iteration_initial = iteration_results[iteration + 1][0]
            current_iteration_solution = iteration_results[iteration + 1][1]
            current_iteration_energy_pop = iteration_results[iteration + 1][2]
            current_iteration_energy_access = iteration_results[iteration + 1][3]

    return iteration_results, current_iteration_initial, current_iteration_solution, current_iteration_energy_pop, current_iteration_energy_access



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