import networkx as nx
import travel_time as tt
import random
import numpy as np
import sys
import math
import heuristic as heu
import sorting


# We build a graph for the grid we have. Future advice for complexity:
# i) We can directly create a graph instead of creating a grid.
# ii) Or we can find a way to sample a spanning tree from the grid.

def build_graph(grid, blocks, stops, existings, possibles):
    
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(list(blocks.keys()), id=0)     # block attribute -> id = 0
    G.add_nodes_from(list(stops.keys()), id=1)      # stop attribute -> id = 1
    G.add_nodes_from(list(existings.keys()), id=2)  # existing attribute -> id = 2
    G.add_nodes_from(list(possibles.keys()), id=3)  # possible attribute -> id = 3
    
    # Add edges
    for node in grid.keys():

        value = grid[node]
        neighbors = value.get_node_neighbors()
        distances = value.get_node_distance()

        for neighbor in neighbors:
            distance = distances[neighbor]
            if G.has_edge(neighbor, node) == False:
                G.add_edge(node, neighbor, weight=distance)  # edge attribute --> weight=distance between endpoints

    return G


def random_centers(existing, possible, p):

    openings = random.sample(list(possible.keys()), k=p)
    centers = list(existing.keys()) + openings

    return centers



def distance(p1, p2):  # Euclidean distance
    return np.sum(np.square(p1 - p2))
  
def choose_centers_kmeans2(grid, existing, possibles, p):  

    possible_centroids = np.array([np.array(tuple) for tuple in possibles.keys()])

    centroids = list(existing.keys())
    if len(possible_centroids) >= p:

        for _ in range(p):
            # Calculate distances from each point to the current centroids
            dist = [min(distance(point, centroid) for centroid in centroids) for point in possible_centroids]
            # Choose the next centroid from possible_centroids based on the maximum distance
            next_centroid_index = int(np.argmax(dist))
            next_centroid = possible_centroids[next_centroid_index]
            centroids.append(next_centroid)
    else:
        print("Error: Not enough possible centroids for k-means")

    list_of_tuples = [tuple(arr) for arr in centroids]

    return list_of_tuples

def choose_centers_kmeans_3(grid, existing, possibles, p): 
    possible_centroids = np.array([np.array(tuple) for tuple in possibles.keys()])  # 10 locations

    centroids = list(existing.keys())  # currently 5 locations. 5 will come from possibles.
    
    if len(possible_centroids) >= p:

        for _ in range(p):  # repeat the following procedure p times
            # Compute distances from points to their nearest centroids. |distances| = |possible_centroids|
            distances = [min(distance(point, centroid) for centroid in centroids) for point in possible_centroids]
            # Choose the next centroid with probability proportional to squared distance
            probabilities = np.array(distances) / np.sum(distances)
            next_centroid_index = np.random.choice(len(possible_centroids), p=probabilities)
            next_centroid = possible_centroids[next_centroid_index]
            centroids.append(next_centroid)
            possible_centroids = np.delete(possible_centroids, next_centroid_index, axis=0)

    else:



        print("Error: Not enough possible centroids for k-means")

    # Convert each centroid to a tuple before creating the final list
    return [tuple(centroid) for centroid in centroids]

def create_partition(tree, grid, existings, centers, total_pop, epsilon):

    exists_list = list(existings.keys())
    centerss = exists_list + centers

    lower_bound = (1 - epsilon)* total_pop / len(centerss) 
    upper_bound = (1 + epsilon)* total_pop / len(centerss)


    subgraphs = {}
    populations = {}
    k= 0 

    for center in centerss:
        populations[center] = 0
        
        while True:

            k = k + 1
            if populations[center] < lower_bound:

                graph = nx.ego_graph(tree, center, radius=k, undirected = False)    # We can use edge weight (travel time) in the radius.
                subgraphs[center] = list(graph.nodes)
        
                for node in subgraphs[center]:
                    value = grid[node]
                    populations[center] = populations[center] + value.get_node_population()

            else: 
                break

    return subgraphs, populations


def choose_centers(tree, grid, existing, possibles, p):     # According to graph distances, incomplete.
          
    # The k-mean++ algorithm - (1.a)    (We do not choose it from the whole data set. We choose it from the set of possibles.)
    possible_centroids = np.array([np.array(tuple) for tuple in possibles.keys()])

    centroids = list(existing.keys())
    if len(possible_centroids) >= p:

        for _ in range(p):
            # Compute distances from points to the nearest centroid
            distances = [min(distance(point, centroid) for centroid in centroids) for point in possible_centroids]
            # Choose the next centroid with probability proportional to squared distance
            probabilities = np.array(distances) / np.sum(distances)
            next_centroid_index = np.random.choice(len(possible_centroids), p=probabilities)
            next_centroid = possible_centroids[next_centroid_index]
            centroids.append(next_centroid)
            possible_centroids = np.delete(possible_centroids, next_centroid_index, axis=0)

    else:
        print("Error: Not enough possible centroids for k-means")

    # Convert each centroid to a tuple before creating the final list
    locations = [tuple(centroid) for centroid in centroids]

        # Calculate graph distances between vertices in the tree.
    length = nx.all_pairs_shortest_path_length(tree)             ##### Is it costly on the spanning tree? We may need to change this.

    dictionary = {}
    dict_for_node = {}
    final_dict = {}
    b = list(length)

    for i in range(len(b)):
        node = b[i][0]
        dictionary[node] = b[i][1]

    for node in tree.nodes:
        dict_for_node[node] = []
        for facility in locations: # existingi degistir
            distance = dictionary[node][facility]
            dict_for_node[node].append((facility, distance))
    
    for node in tree.nodes:
        sorting.merge_sort(dict_for_node[node])  
    
    return 


def heu_cluster(graph, grid, stops, blocks, all_facilities, centers, travel):

    locations = {}
    for center in centers:
        value = all_facilities[center]
        locations[center] = value

    location, allocation = heu.assignment_travel(grid, stops, blocks, all_facilities, locations, travel)
    clusters = {}  # key: center , value: list of nodes

    for center in locations.keys():
        clusters[center] = []
        for node in grid.keys():

            if allocation[(node, center)] == 1:
                clusters[center].append(node)
                clusters[center].append(center)

    return clusters





