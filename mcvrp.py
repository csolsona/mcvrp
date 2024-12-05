import sys
import copy
import time

import networkx as nx
import numpy as np

def solve():
    start_time = time.time()
    init()

    sol, cost = nearest_neighbour_alg()
    print_sol(sol, cost)
    if (not sol):
        exit(-1)

    new_cost = 0
    i = 0
    while (cost != new_cost):
        i += 1
        cost = new_cost
        sol, new_cost = intraroute_swap(sol)
        sol, new_cost = interroute_swap(sol)
    end_time = time.time()
    print_sol(sol, cost)
    print('i:', i)
    print("--- %s seconds ---" % (end_time - start_time))


def init():
    if (len(sys.argv) < 2):
        print("Not enough parameters")
        exit(1)
    
    file_name = sys.argv[1]
    f = open("instances/Abdulkader/" + file_name)

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



def nearest_neighbour_alg():
    # Nearest neighbour algorithm
    sol = []
    cost = 0

    for _ in range(G.number_of_nodes()):
        start_node = 0
        capacity = G.graph['capacity'].copy()
        time_limit = G.graph['max_time']
        path = [start_node]
        nearest = get_nearest_neighbour(G, start_node, capacity, time_limit)
        while nearest != -1:
            G.nodes[nearest]['visited'] = True
            cost += G[start_node][nearest]['cost']
            time_limit -= G[start_node][nearest]['cost']  
            # print(time_limit)   # I don't know how this works
            path.append(nearest)
            capacity[0] -= G.nodes[nearest]['demand'][0]
            capacity[1] -= G.nodes[nearest]['demand'][1]
            start_node = nearest
            nearest = get_nearest_neighbour(G, start_node, capacity, time_limit)
            if nearest == -1:
                path.append(0)
                cost += G[start_node][0]['cost']
                sol.append(path)

        non_visited_nodes = [node for node in G.nodes() if not G.nodes[node]['visited']]
        if (len(non_visited_nodes) == 0):
            break
    
    non_visited_nodes = [node for node in G.nodes() if not G.nodes[node]['visited']]
    if (len(non_visited_nodes) > 0):
        return False

    return (sol, cost)


def get_nearest_neighbour(G, start_node, capacity, time_limit):
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


def get_path_cost(path):
    cost = 0
    for i in range(len(path) - 1):
        cost += G[path[i]][path[i+1]]['cost']
    return cost

def get_neighbour_cost(path, i):
    return G[path[i-1]][path[i]]['cost'] + G[path[i]][path[i+1]]['cost']


def intraroute_swap(paths):

    sol = []
    for path in paths:
        finished = False
        while (not finished):
            for i in range(1, len(path) - 1):
                path_cost_i = get_neighbour_cost(path, i)
                for j in range(i+1, len(path) - 1):
                    path_cost = path_cost_i + get_neighbour_cost(path, j)
                    swap_path = copy.deepcopy(path)
                    swap_path[i], swap_path[j] = swap_path[j], swap_path[i]
                    swap_cost = get_neighbour_cost(swap_path, i) + get_neighbour_cost(swap_path, j)
                    if (swap_cost < path_cost):
                        path = swap_path
                        cost = swap_cost
                        break
                else:
                    continue
                break   # Break out of the nested loop as well
            finished = True

        sol.append(path)

    cost = 0
    for path in sol:
        cost += get_path_cost(path)

    return (sol, cost)


def interroute_swap(paths):

    def loop():
        for n in range(len(paths) - 1):
            for m in range(n + 1, len(paths)):
                for i in range(1, len(paths[n]) - 1):
                    for j in range(1, len(paths[m]) - 1):
                        path_cost = get_neighbour_cost(paths[n], i) + get_neighbour_cost(paths[m], j)
                        swap_path_n, swap_path_m = copy.deepcopy(paths[n]), copy.deepcopy(paths[m])
                        swap_path_n[i], swap_path_m[j] = swap_path_m[j], swap_path_n[i]
                        swap_cost = get_neighbour_cost(swap_path_n, i) + get_neighbour_cost(swap_path_m, j)
                        if (swap_cost < path_cost):
                            paths[n], paths[m] = swap_path_n, swap_path_m
                            return True
        return False
    
    while loop():
        pass

    cost = 0
    for path in paths:
        cost += get_path_cost(path)

    return paths, cost 


def print_sol(sol, cost):
    if not sol:
        print('No solution found')
        return

    print('Paths:', sol)
    print('Total cost:', cost)


solve()
