import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches


## Open Possible Facilities (dot ,each has different color)
## Existing Facilities (mark, each has different color)
## Closed Facilities  (Dot, no color)
## Stops   (Cross, no color)
# Blocks  (No dot, facility color)


def plot(m, n, V_blocks, V_existing, V_possible, V_stops, location, allocation, text):  # possible and existing phcs are also assigned to clusters with a population. didn't change it here.


    all_facilities = {**V_existing, **V_possible}

    open_facilities = []
    closed_facilities = []

    for j in V_possible.keys():
        if location[j] == 1:
            open_facilities.append(j)
        else:
            closed_facilities.append(j)    

    size_m = m + 0.5
    size_n = n + 0.5

    grid_size = 1
    x1 = np.arange(-0.5, size_n, grid_size)
    y1 = np.arange(-0.5, size_m, grid_size)


    # Define your 15 colors
    open_colors = ['pink', 'brown', 'gold', 'indigo', 'cyan', 'orange', 'gray', 'olive', 'lime', 'teal', 'magenta']
    existing_colors = ['red', 'blue', 'green', 'yellow', 'purple']

    center_color = {}

    for stop in V_stops.keys():
        x = stop[0]
        y = stop[1]
        plt.scatter(x, y, c='black',s=50, facecolors=None)    # stops are marked using cross.(no ceil color) 

    for close in closed_facilities:
        x = close[0]
        y = close[1] 
        plt.scatter(x, y, c='black', marker="x", facecolors=None)   # Closed Facilities: marker = x  (no ceil color)

    for k, exist in enumerate(V_existing.keys()):   # Existing Facilities: marker = square.  (each ceil has different color)
        x = exist[0]
        y = exist[1]
        center_color[exist] = existing_colors[k]
        plt.scatter(x, y, c='black', marker="s", facecolors='none')  

    for k, open in enumerate(open_facilities):   # Open Possible Facilities: marker = +. (each ceil has different color)
        x = open[0]
        y = open[1]
        center_color[open] = open_colors[k]
        plt.scatter(x, y, c='black', marker="P", facecolors='none')  

    for block in V_blocks.keys():
        for facility in all_facilities:
            if allocation[block, facility] == 1:  # We are assuming every block is assigned to one facility.
                    center_color[block] = center_color[facility]

    # Draw vertical lines
    for i in range(len(x1)):
        plt.vlines(x1[i], ymin=y1[0], ymax=y1[-1], color='black')

    # Draw horizontal lines
    for j in range(len(y1)):
        plt.hlines(y1[j], xmin=x1[0], xmax=x1[-1], color='black')

    for key in center_color.keys():
        x = key[0]
        y = key[1]
        x_coords = [x - 0.5, x + 0.5, x + 0.5, x - 0.5]
        y_coords = [y - 0.5, y - 0.5, y + 0.5, y + 0.5]

        plt.fill_between(x_coords, y_coords, color=center_color[key])

    for stop in V_stops.keys():
        x = stop[0]
        y = stop[1]
        plt.scatter(x, y, c='black',s=50, facecolors=None)    # stops are marked using cross.(no ceil color) 

    for close in closed_facilities:
        x = close[0]
        y = close[1]
        plt.scatter(x, y, c='black', marker="x", facecolors=None)   # Closed Facilities: marker = x  (no ceil color)

    for k, exist in enumerate(V_existing.keys()):   # Existing Facilities: marker = square.  (each ceil has different color)
        x = exist[0]
        y = exist[1]
        center_color[exist] = existing_colors[k]
        plt.scatter(x, y, c='black', marker="s", facecolors='none')  

    for k, open in enumerate(open_facilities):   # Open Possible Facilities: marker = +. (each ceil has different color)
        x = open[0]
        y = open[1]
        center_color[open] = open_colors[k]
        plt.scatter(x, y, c='black', marker="P", facecolor='none')  

    plt.yticks(y1)
    plt.xticks(x1, rotation=90)
    plt.title(text)
    plt.show()


def plot_graph(graph, stops, possibles, existings, all_facilities, centers, node_size, alpha_node, alpha_edge):

    # block attribute -> id = 0
    # stop attribute -> id = 1
    # existing attribute -> id = 2
    # possible attribute -> id = 3
    # edge attribute --> weight=distance between endpoints
    

    list_nodes = list(graph.nodes)
    list_stops = list(stops.keys())
    list_not_stop = [i for i in list_nodes if i not in list_stops]
    list_possibles =  list(possibles.keys())
    closed_possibles = [item for item in list_possibles if item not in centers]

    node_positions = {}
    for i in list_nodes:
        node_positions[i]=i


    options = {"node_size": node_size, "alpha": 0.7}

    nx.draw_networkx_nodes(graph, node_positions, nodelist=list_stops, node_color="tab:red", **options)
    nx.draw_networkx_nodes(graph, node_positions, nodelist=list_not_stop, node_color="tab:blue", **options)
    
    induced_stops= graph.subgraph(list_stops)
    remaining_edges = []

    for edge in graph.edges:
        endpoint1, endpoint2 = edge
        if graph.nodes[endpoint1]["id"]!=1 or graph.nodes[endpoint2]["id"]!=1:
            remaining_edges.append(edge)


    # edges
    nx.draw_networkx_edges(graph, node_positions, width=1.0, alpha=0.5)
    nx.draw_networkx_edges(
        graph,
        node_positions,
        edgelist=induced_stops.edges,
        width=1,
        alpha=0.5,
        edge_color='red'
    )
    nx.draw_networkx_edges(
        graph,
        node_positions,
        edgelist=remaining_edges,
        width=1,
        alpha=0.5,
        edge_color='blue'
    )


    # some math labels
    labels = {}
    
    for node in graph.nodes:
        if graph.nodes[node]["id"]== 0:
            labels[node] = ""
        elif graph.nodes[node]["id"]== 1:
            labels[node] = r"$s$"      
        elif graph.nodes[node]["id"]== 2:
            labels[node] = r"$e$"
        elif node in centers:
            labels[node] = r"$o$"
        elif node in closed_possibles:
            labels[node] = r"$x$"

    nx.draw_networkx_labels(graph, node_positions, labels, font_size=15, font_color="whitesmoke")

    plt.tight_layout()
    plt.axis("off")
    plt.show()


    return

def plot_graph_clusters(graph, stops, possibles, existings, all_facilities, centers, node_size, subgraphs):

    # block attribute -> id = 0
    # stop attribute -> id = 1
    # existing attribute -> id = 2
    # possible attribute -> id = 3
    # edge attribute --> weight=distance between endpoints


    list_nodes = list(graph.nodes)
    list_stops = list(stops.keys())
    list_possibles =  list(possibles.keys())
    closed_possibles = [item for item in list_possibles if item not in centers]


    node_positions = {}
    for i in list_nodes:
        node_positions[i]=i


    open_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'pink', 'brown', 'gold', 'indigo', 'cyan', 'orange', 'gray', 'olive', 'lime', 'teal', 'magenta', 'red', 'blue', 'green', 'yellow', 'purple']

    community_subgraph = {}
    for center in subgraphs.keys():
        community_subgraph[center] = nx.induced_subgraph(graph, subgraphs[center])

    
    list_keys = list(subgraphs.keys())


    for center in list_keys:
        
        options = {"node_size": node_size, "alpha": 0.7}
        index = list_keys.index(center)
        nx.draw_networkx_nodes(graph, node_positions, nodelist=subgraphs[center], node_color=open_colors[index], **options)

    

    induced_stops= graph.subgraph(list_stops)
    remaining_edges = []

    for edge in graph.edges:
        endpoint1, endpoint2 = edge
        if graph.nodes[endpoint1]["id"]!=1 or graph.nodes[endpoint2]["id"]!=1:
            remaining_edges.append(edge)


    # edges
    nx.draw_networkx_edges(graph, node_positions, width=1.0, alpha=0.5)
    nx.draw_networkx_edges(
        graph,
        node_positions,
        edgelist=induced_stops.edges,
        width=1,
        alpha=0.5,
        edge_color='red'
    )
    nx.draw_networkx_edges(
        graph,
        node_positions,
        edgelist=remaining_edges,
        width=1,
        alpha=0.5,
        edge_color='blue'
    )


    # some math labels
    labels = {}
    
    for node in graph.nodes:
        if graph.nodes[node]["id"]== 0:
            labels[node] = ""
        elif graph.nodes[node]["id"]== 1:
            labels[node] = r"$s$"      
        elif graph.nodes[node]["id"]== 2:
            labels[node] = r"$e$"
        elif node in centers:
            labels[node] = r"$o$"
        elif node in closed_possibles:
            labels[node] = r"$x$"

    nx.draw_networkx_labels(graph, node_positions, labels, font_size=15, font_color="whitesmoke")

    plt.tight_layout()
    plt.axis()
    plt.show()


    return


def plot_graph_clusters_boundaries(graph, stops, possibles, existings, all_facilities, centers, node_size, subgraphs, boundaries, origin):
    print("clusters:", subgraphs)
    # block attribute -> id = 0
    # stop attribute -> id = 1
    # existing attribute -> id = 2
    # possible attribute -> id = 3
    # edge attribute --> weight=distance between endpoints


    list_nodes = list(graph.nodes)
    list_stops = list(stops.keys())
    list_possibles =  list(possibles.keys())
    closed_possibles = [item for item in list_possibles if item not in centers]


    node_positions = {}
    for i in list_nodes:
        node_positions[i]=i


    open_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'pink', 'brown', 'gold', 'indigo', 'cyan', 'orange', 'gray', 'olive', 'lime', 'teal', 'magenta', 'red', 'blue', 'green', 'yellow', 'purple']

    community_subgraph = {}
    for center in subgraphs.keys():
        community_subgraph[center] = nx.induced_subgraph(graph, subgraphs[center])

    
    list_keys = list(subgraphs.keys())


    for center in list_keys:

        if center == origin:
            nodes_boundary = []
            index_origin = list_keys.index(origin)

            for pair in boundaries.keys():  # # boundaries -> key: (origin, a cluster),  value: nodes in the boundary of origin, and adjacent to the cluster. Origin is fixed.
                nodes_boundary = list(set(nodes_boundary) | set(boundaries[pair])) 
                cluster = pair[1]    # cluster = neighboring cluster of the origin
                index_cluster = list_keys.index(cluster)
                options = {"edgecolors": open_colors[index_cluster], "node_size": node_size, "alpha": 0.7}
                nx.draw_networkx_nodes(graph, node_positions, nodelist=boundaries[(origin, cluster)], node_color=open_colors[index_origin], **options)
            
            nodes_nonboundary = [i for i in subgraphs[origin] if i not in nodes_boundary]
            options = {"node_size": node_size, "alpha": 0.7}
            nx.draw_networkx_nodes(graph, node_positions, nodelist=nodes_nonboundary, node_color=open_colors[index_origin], **options)


        else:
            options = {"node_size": node_size, "alpha": 0.7}
            index = list_keys.index(center)
            nx.draw_networkx_nodes(graph, node_positions, nodelist=subgraphs[center], node_color=open_colors[index], **options)
    

    induced_stops= graph.subgraph(list_stops)
    remaining_edges = []

    for edge in graph.edges:
        endpoint1, endpoint2 = edge
        if graph.nodes[endpoint1]["id"]!=1 or graph.nodes[endpoint2]["id"]!=1:
            remaining_edges.append(edge)


    # edges
    nx.draw_networkx_edges(graph, node_positions, width=1.0, alpha=0.5)
    nx.draw_networkx_edges(
        graph,
        node_positions,
        edgelist=induced_stops.edges,
        width=1,
        alpha=0.5,
        edge_color='red'
    )
    nx.draw_networkx_edges(
        graph,
        node_positions,
        edgelist=remaining_edges,
        width=1,
        alpha=0.5,
        edge_color='blue'
    )


    # some math labels
    labels = {}
    
    for node in graph.nodes:
        if graph.nodes[node]["id"]== 0:
            labels[node] = ""
        elif graph.nodes[node]["id"]== 1:
            labels[node] = r"$s$"      
        elif graph.nodes[node]["id"]== 2:
            labels[node] = r"$e$"
        elif node in centers:
            labels[node] = r"$o$"
        elif node in closed_possibles:
            labels[node] = r"$x$"

    nx.draw_networkx_labels(graph, node_positions, labels, font_size=15, font_color="whitesmoke")

    plt.tight_layout()
    plt.axis()
    plt.show()


    return
