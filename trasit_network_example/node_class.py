class Node():
    def __init__(self, posX, posY, population, travel_time, neighbours):
        self.posX = posX
        self.posY = posY
        self.population = population
        self.travel_time = travel_time
        self.neigbours = neighbours

    def get_position(self):
        return self.posX, self.posY



n1 = Node(1,2,10,[1,2,3,4],[])

print(n1.get_position())