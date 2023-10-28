class Node():
    def __init__(self, name, ID, pos_x, pos_y, population, travel_time, neighbors, neighbors_subway, EPHC, PPHC):
        self.name = name
        self.ID = ID
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.population = population
        self.travel_time = travel_time
        self.neighbors = neighbors
        self.neighbors_subway = neighbors_subway
        self.EPHC = EPHC
        self.PPHC = PPHC

    def get_node_position(self):
        return self.posX, self.posY
    
    def get_node_ID(self):
        return self.ID

    def get_node_neighbors(self):
        return self.neighbors
    def get_node_neighbors_subway(self):
        return self.neighbors_subway
#n1 = Node(1,2,10,[1,2,3,4],[])
# print(n1.get_position())


class Edge():
    def __init__(self, name, ID, type, line, tail, head, travel_time):
        self.name = name
        self.ID = ID
        self.type = type
        self.line = line
        self.tail = tail
        self.head = head
        self.travel_time = travel_time
    
    def get_edge_tail(self):
        return self.tail 

    def get_edge_head(self):
        return self.head    

    def get_edge_travel_time(self):
        return self.travel_time 