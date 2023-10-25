import classes as cl
import network as net


max_cost = 100
traveltime_threshold = 30
max_DN_teams = 4
max_new_PHCs = 10
alpha = 0.02


n2 = cl.Node(1,2,10,[1,2,3,4],[])

print(n2.get_position())