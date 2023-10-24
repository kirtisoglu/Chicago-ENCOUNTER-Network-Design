class Node():
    def __init__(self, ID, posX, posY, population, travel_time, neighbors, EPHC, PPHC):
        self.ID = ID
        self.posX = posX
        self.posY = posY
        self.population = population
        self.travel_time = travel_time
        self.neighbors = neighbors
        self.EPHC = EPHC
        self.PPHC = PPHC

    def get_node_position(self):
        return self.posX, self.posY
    
    def get_node_ID(self):
        return self.ID

    def get_node_neighbors(self):
        return self.neighbors

#n1 = Node(1,2,10,[1,2,3,4],[])
# print(n1.get_position())


class Edge():
    def __init__(self, ID, type, line, tail, head, travel_time):
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