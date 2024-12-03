import networkx as nx
import numpy as np

# Define G as global

def init():
    init_instance()
    solve()


def init_instance():
    # placeholder
    f = open("instances/Abdulkader/vrpnc13b.txt")

    data = f.readline().strip().split()
    n = int(data[5])
    global G
    G = nx.Graph(
        capacity = [floatstr_to_int(data[3]), floatstr_to_int(data[4])],
        max_time = float(data[6]),
        drop_time = float(data[7])
    )

    # Add nodes
    G.add_node(int(data[0]), coords = [int(data[1]), int(data[2])], demand = [0, 0], visited = True)
    for l in f:
        line = l.strip().split()
        G.add_node(int(line[0]), coords = [int(line[1]), int(line[2])], demand = [floatstr_to_int(line[3]), floatstr_to_int(line[4])], visited = False)
    
    f.close()

    # Add edges
    for i in range(n):
        for j in range(i+1, n+1):
            distance = get_distance(G.nodes[i], G.nodes[j])
            G.add_edge(i, j, cost=distance)


def floatstr_to_int(str):
    return int(str.replace('.', ''))


def get_distance(u, v):
    p1 = u['coords']
    p2 = v['coords']
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def can_satisfy(capacity, demand):
    return (capacity[0] >= demand[0] and capacity[1] >= demand[1])



def solve():
    # Nearest neighbour algorithm
    sol = []
    cost = 0

    for _ in range(G.number_of_nodes()):
        start_node = 0
        capacity = G.graph['capacity'].copy()
        time_limit = G.graph['max_time']
        path = [start_node]
        nearest = nearest_neighbour(G, start_node, capacity, time_limit)
        while nearest != -1:
            G.nodes[nearest]['visited'] = True
            cost += G[start_node][nearest]['cost']
            time_limit -= G[start_node][nearest]['cost']  
            # print(time_limit)   # I don't know how this works
            path.append(nearest)
            capacity[0] -= G.nodes[nearest]['demand'][0]
            capacity[1] -= G.nodes[nearest]['demand'][1]
            start_node = nearest
            nearest = nearest_neighbour(G, start_node, capacity, time_limit)
            if nearest == -1:
                path.append(0)
                cost += G[start_node][0]['cost']
                sol.append(path)

        non_visited_nodes = [node for node in G.nodes() if not G.nodes[node]['visited']]
        if (len(non_visited_nodes) == 0):
            break
    
    non_visited_nodes = [node for node in G.nodes() if not G.nodes[node]['visited']]
    if (len(non_visited_nodes) > 0):
        print('No solution found')
        return

    print('Paths:', sol)
    print('Total cost:', cost)


def nearest_neighbour(G, start_node, capacity, time_limit):
    neighbors = [
        (neighbour, G[start_node][neighbour]['cost'])
        for neighbour in G.neighbors(start_node)
        if (
            not G.nodes[neighbour]['visited']
            and can_satisfy(capacity, G.nodes[neighbour]['demand'])
            and G[start_node][neighbour]['cost'] <= time_limit
        )
    ]
    if (len(neighbors) == 0):
        return -1
    
    neighbors.sort(key=lambda x: x[1])
    return neighbors[0][0]


init()
