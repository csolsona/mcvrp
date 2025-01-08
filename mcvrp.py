import sys
import copy
import math
import random
import time

import networkx as nx
import numpy as np


def init():
    if (len(sys.argv) < 2):
        print("Not enough parameters")
        exit(1)
    
    file_name = sys.argv[1]
    f = open("instances/Abdulkader/" + file_name)

    data = f.readline().strip().split()
    n = int(data[5])
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
    
    return G


def solve(G, k_max, n):

    sol, cost = nearest_neighbour_alg(G)
    if (not sol):
        exit(-1)
    print_sol(sol, cost)

    sol, cost = vns(sol, cost, k_max, n)
    
    for path in sol:
        print(is_path_feasible(path), end=", ")
    print()

    return sol, cost

# Variable neighborhood search (VNS)
def vns(sol, cost, k_max, n):

    for _ in range(n):
        k = 5
        cost = get_sol_cost(sol)
        while (k <= k_max):
            shaken_sol = shake(sol, k)
            new_sol, new_cost = local_search(shaken_sol)
            if (new_cost < cost):
                print(new_cost)
                k = 5
                sol = new_sol
                cost = new_cost
            else:
                k += 5
    
    return sol, cost


def local_search(sol):

    cost = get_sol_cost(sol)
    i = 0

    new_cost = cost
    while (True):
        i += 1
        cost = new_cost
        sol, new_cost = intraroute_swap(sol)
        if (new_cost < cost): continue
        sol, new_cost = interroute_swap(sol)
        if (new_cost < cost): continue
        sol, new_cost = two_opt_swap(sol)
        if (new_cost < cost): continue
        sol, new_cost = intraroute_insertion(sol)
        if (new_cost < cost): continue
        sol, new_cost = interroute_insertion(sol)
        if (new_cost < cost): continue
        break

    # print('i:', i)
    
    return sol, cost


def shake(sol, k):
    print('k:', k)

    def generate_indexes(sol, visited_nodes):
        indexes = []
        for i, path in enumerate(sol):
            for j, node in enumerate(path):
                if (node == 0) or (node in visited_nodes):
                    continue
                indexes.append((i, j))
        return indexes

    nodes_to_change = math.ceil(len(generate_indexes(sol, [])) * (k / 100))
    
    def loop():
        shaken_sol = copy.deepcopy(sol)
        visited_nodes = []
        nodes_changed = 0
        while (nodes_changed < nodes_to_change):
            indexes_orig = generate_indexes(shaken_sol, visited_nodes)
            indexes_dest = generate_indexes(shaken_sol, [])
            random.shuffle(indexes_orig)
            random.shuffle(indexes_dest)

            x, i = indexes_orig.pop()
            item = shaken_sol[x].pop(i)
            while True:
                if (not indexes_dest):
                    return False
                y, j = indexes_dest.pop()
                # print(shaken_sol)
                shaken_sol[y].insert(j, item)
                if (is_path_feasible(shaken_sol[x]) and is_path_feasible(shaken_sol[y])):
                    visited_nodes.append(item)
                    nodes_changed += 1
                    break
                else:
                    shaken_sol[y].pop(j)
        
        return shaken_sol


    shaken_sol = loop()
    while not shaken_sol:
        shaken_sol = loop()
    
    return shaken_sol


def floatstr_to_int(str):
    return int(str.replace('.', ''))


def get_distance(u, v):
    p1 = u['coords']
    p2 = v['coords']
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def can_satisfy(capacity, demand):
    return (capacity[0] >= demand[0] and capacity[1] >= demand[1])



def nearest_neighbour_alg(G):
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


# Sum of nodes does not exceed the maximum capacity
def is_path_feasible(path):
    return (
        sum(G.nodes[a]['demand'][0] for a in path) <= G.graph['capacity'][0]
        and
        sum(G.nodes[a]['demand'][1] for a in path) <= G.graph['capacity'][1]
    )


def get_sol_cost(sol):
    return sum(get_path_cost(path) for path in sol)


def get_path_cost(path):
    return sum(G[a][b]['cost'] for a, b in zip(path, path[1:]))


def get_neighbour_cost(path, i):
    return G[path[i-1]][path[i]]['cost'] + G[path[i]][path[i+1]]['cost']


def intraroute_swap(paths):
    # first improvement

    sol = []
    for path in paths:
        finished = False
        while (not finished):
            indexes = list(range(1, len(path) - 1))
            random.shuffle(indexes)
            for i, n in enumerate(indexes):
                path_cost_n = get_neighbour_cost(path, n)
                for m in indexes[i+1:]:
                    path_cost = path_cost_n + get_neighbour_cost(path, m)
                    swap_path = copy.deepcopy(path)
                    swap_path[n], swap_path[m] = swap_path[m], swap_path[n]
                    swap_cost = get_neighbour_cost(swap_path, n) + get_neighbour_cost(swap_path, m)
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


def intraroute_insertion(paths):

    sol = []
    for path in paths:
        finished = False
        while (not finished):
            indexes = list(range(1, len(path) - 1))
            random.shuffle(indexes)
            path_cost = get_path_cost(path)     # lo he cambiado, si funciona bien borrar el comentario
            for n in indexes:
                for m in indexes:
                    if n == m:
                        continue
                    # print(path)
                    # print(n, m)
                    insertion_path = copy.deepcopy(path)
                    item = insertion_path.pop(n)
                    insertion_path.insert(m, item)
                    # print(insertion_path)
                    # print('-----------')
                    insertion_cost = get_path_cost(insertion_path)
                    if (insertion_cost < path_cost):
                        path = insertion_path
                        cost = insertion_cost
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
        for x in range(len(paths) - 1):
            indexes_x = list(range(1, len(paths[x]) - 1))
            random.shuffle(indexes_x)
            for y in range(x + 1, len(paths)):
                indexes_y = list(range(1, len(paths[y]) - 1))
                random.shuffle(indexes_y)
                for n in indexes_x:
                    for m in indexes_y:
                        path_cost = get_neighbour_cost(paths[x], n) + get_neighbour_cost(paths[y], m)
                        swap_path_x, swap_path_y = copy.deepcopy(paths[x]), copy.deepcopy(paths[y])
                        swap_path_x[n], swap_path_y[m] = swap_path_y[m], swap_path_x[n]
                        swap_cost = get_neighbour_cost(swap_path_x, n) + get_neighbour_cost(swap_path_y, m)
                        if (swap_cost < path_cost and
                            is_path_feasible(swap_path_x) and is_path_feasible(swap_path_y)
                            ):
                            paths[x], paths[y] = swap_path_x, swap_path_y
                            return True
        return False
    
    while loop():
        pass

    cost = 0
    for path in paths:
        cost += get_path_cost(path)

    return paths, cost 


def interroute_insertion(paths):

    def loop():
        for x in range(len(paths)):
            path_cost_x = get_path_cost(paths[x])
            indexes_x = list(range(1, len(paths[x]) - 1))
            random.shuffle(indexes_x)
            for y in range(len(paths)):
                if x == y:
                    continue
                path_cost = path_cost_x + get_path_cost(paths[y])
                indexes_y = list(range(1, len(paths[y]) - 1))
                random.shuffle(indexes_y)
                for n in indexes_x:
                    for m in indexes_y:
                        insertion_path_x, insertion_path_y = copy.deepcopy(paths[x]), copy.deepcopy(paths[y])
                        item = insertion_path_x.pop(n)
                        insertion_path_y.insert(m, item)
                        swap_cost = get_path_cost(insertion_path_x) + get_path_cost(insertion_path_y)
                        if (swap_cost < path_cost and
                            is_path_feasible(insertion_path_x) and is_path_feasible(insertion_path_y)
                            ):
                            paths[x], paths[y] = insertion_path_x, insertion_path_y
                            return True
        return False
    
    while loop():
        pass

    cost = 0
    for path in paths:
        cost += get_path_cost(path)

    return paths, cost 


def two_opt_swap(paths):
    sol = []
    for path in paths:
        finished = False
        while not finished:
            indexes = list(range(1, len(path) - 1))
            random.shuffle(indexes)
            path_cost = get_path_cost(path)
            for i in indexes:
                for j in indexes:
                    if i == j:
                        continue
                    x, y = i, j
                    if i > j:
                        x, y = j, i
                    # i, j = path.index(n), path.index(m)
                    swap_path = (
                        path[:x] +
                        path[x:y][::-1] +
                        path[y:]
                    )
                    # print(x, y)
                    # print(path, swap_path)
                    # print('---------')
                    swap_cost = get_path_cost(swap_path)
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


def print_sol(sol, cost):
    if not sol:
        print('No solution found')
        return

    print('Paths:', sol)
    print('Total cost:', round(cost, 1))


start_time = time.time()

global G
G = init()

sol, cost = solve(G, 20, 1)

end_time = time.time()
print_sol(sol, cost)
print("--- %s seconds ---" % (end_time - start_time))