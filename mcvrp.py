import copy
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

import networkx as nx
import numpy as np


# Size of the per-customer neighbourhood used to restrict candidate moves in the local search operators.
NEIGHBOURHOOD_SIZE = 20

# Minimum delta for a move to count as an improvement, so float round-off on zero-cost moves isn't mistaken for a gain.
IMPROVEMENT_TOLERANCE = 1e-9


def init(file_name):
    
    try:
        f = open("instances/" + file_name)
    except FileNotFoundError as e:
        print(e)
        exit(1)

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

    # Precompute, for each customer, the NEIGHBOURHOOD_SIZE nearest customers
    # (depot 0 excluded), sorted by ascending distance. The local search
    # operators restrict their candidate partners to these lists.
    neighbours = {}
    customer_ids = list(range(1, G.number_of_nodes()))
    for i in customer_ids:
        order = sorted((j for j in customer_ids if j != i), key=lambda j: G[i][j]['cost'])
        neighbours[i] = order[:NEIGHBOURHOOD_SIZE]
    G.graph['neighbours'] = neighbours

    return G

        

def solve(G, k_max, num_threads):

    sol, cost = nearest_neighbour_alg(G)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(execute, G, sol, k_max) for _ in range(num_threads)]

    results = []
    for future in as_completed(futures):
        results.append(future.result())

    all_sol = list(map(lambda x: x[0], results))

    valid_sol = all([is_path_feasible(G, path) for sol in all_sol for path in sol])
    if (not valid_sol):
        print('Error. Solution not valid')
        exit(1)
    
    print('Valid solution')

    return results


def execute(G, sol, k_max):
        start_time = time.perf_counter()
        sol, cost = gvns(G, sol, k_max)
        end_time = time.perf_counter()
        execute_time = end_time - start_time
        return (sol, cost, execute_time)


# General Variable neighborhood search (GVNS)
def gvns(G, sol, k_max):
    k = 5
    cost = get_sol_cost(G, sol)
    while (k <= k_max):
        shaken_sol = shake(G, sol, k)
        new_sol, new_cost = vnd(G, shaken_sol)
        if (new_cost < cost):
            k = 5
            sol = new_sol
            cost = new_cost
        else:
            k += 5

    return sol, cost


# Variable Neighborhood Descent (VND)
def vnd(G, sol):

    cost = get_sol_cost(G, sol)
    i = 0

    new_cost = cost
    
    while (True):
        i += 1
        cost = new_cost
        sol, new_cost = intraroute_swap(G, sol)
        if (new_cost < cost): continue
        sol, new_cost = intraroute_insertion(G, sol)
        if (new_cost < cost): continue
        sol, new_cost = two_opt_swap(G, sol)
        if (new_cost < cost): continue
        sol, new_cost = interroute_swap(G, sol)
        if (new_cost < cost): continue
        sol, new_cost = interroute_insertion(G, sol)
        if (new_cost < cost): continue
        
        
        break

    return sol, cost


def shake(G, sol, k):

    def generate_indexes(sol, visited_nodes):
        indexes = []
        for i, path in enumerate(sol):
            if (len(path) == 2):
                indexes.append((i, 1))
            else:
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
            while (len(shaken_sol[x]) < 3):
                x, i = indexes_orig.pop()

            item = shaken_sol[x].pop(i)
            while True:
                if (not indexes_dest):
                    return False
                y, j = indexes_dest.pop()
                shaken_sol[y].insert(j, item)
                if (is_path_feasible(G, shaken_sol[x]) and is_path_feasible(G, shaken_sol[y])):
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



def random_alg(G):

    def can_append(path, prev_node, node, capacity, time_limit):
        path.append(node)
        path.append(0)
        capacity[0] -= G.nodes[node]['demand'][0]
        capacity[1] -= G.nodes[node]['demand'][1]
        time_limit -= G[prev_node][node]['cost']
        time_limit -= G[node][0]['cost']
        time_limit -= G.graph['drop_time']
        path.pop()
        path.pop()
        return (capacity[0] >= 0) and (capacity[1] >= 0) and (time_limit >= 0)


    sol = []
    start_node = 0
    prev_node = start_node
    nodes = list(range(1, G.number_of_nodes()))
    random.shuffle(nodes)
    capacity = G.graph['capacity'].copy()
    time_limit = G.graph['max_time']
    path = [start_node]
    while nodes:
        node = nodes.pop()
        if can_append(path, prev_node, node, capacity, time_limit):
            path.append(node)
            capacity[0] -= G.nodes[node]['demand'][0]
            capacity[1] -= G.nodes[node]['demand'][1]
            time_limit -= G[prev_node][node]['cost']
            time_limit -= G.graph['drop_time']
            prev_node = node
        else:
            path.append(start_node)
            sol.append(path)
            prev_node = start_node
            capacity = G.graph['capacity'].copy()
            time_limit = G.graph['max_time']
            nodes.insert(0, node)
            path = [start_node]

    # Close and append the last open route, otherwise its customers are lost.
    if len(path) > 1:
        path.append(start_node)
        sol.append(path)

    cost = get_sol_cost(G, sol)

    return (sol, cost)


def nearest_neighbour_alg(G):
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
            time_limit -= G.graph['drop_time']
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
    
    cost = get_sol_cost(G, sol)

    return (sol, cost)


def is_appendable(start_node, node, capacity, time_limit):
    return (
        can_satisfy(capacity, G.nodes[node]['demand'])
        and G[start_node][node]['cost'] + G.graph['drop_time'] <= time_limit
    )


def get_nearest_neighbour(G, start_node, capacity, time_limit):
    neighbors = [
        (neighbour, G[start_node][neighbour]['cost'])
        for neighbour in G.neighbors(start_node)
        if (
            not G.nodes[neighbour]['visited']
            and can_satisfy(capacity, G.nodes[neighbour]['demand'])
            and G[start_node][neighbour]['cost'] + G.graph['drop_time'] + G[neighbour][0]['cost'] <= time_limit
        )
    ]
    if (len(neighbors) == 0):
        return -1
    
    neighbors.sort(key=lambda x: x[1])
    return neighbors[0][0]


def intraroute_swap(G, paths):
    neighbours = G.graph['neighbours']

    def loop(path):
        pos_of = {c: idx for idx, c in enumerate(path) if c != 0}
        indexes = list(range(1, len(path) - 1))
        random.shuffle(indexes)
        for n in indexes:
            cust_n = path[n]
            path_cost_n = get_neighbour_cost(G, path, n)
            for nbr in neighbours[cust_n]:
                m = pos_of.get(nbr)
                if m is None or m == n:
                    continue
                path_cost = path_cost_n + get_neighbour_cost(G, path, m)
                path[n], path[m] = path[m], path[n]
                swap_cost = get_neighbour_cost(G, path, n) + get_neighbour_cost(G, path, m)
                if (swap_cost < path_cost):
                    return path
                else:
                    path[n], path[m] = path[m], path[n]
        
        return False
    
    for path in paths:
        while path:
            path = loop(path)

    cost = get_sol_cost(G, paths)

    return (paths, cost)


def intraroute_insertion(G, paths):
    neighbours = G.graph['neighbours']

    def loop(path):
        pos_of = {c: idx for idx, c in enumerate(path) if c != 0}
        indexes = list(range(1, len(path) - 1))
        random.shuffle(indexes)
        for n in indexes:
            X = path[n]
            prev = path[n-1]
            nxt = path[n+1]
            delta_remove = G[prev][nxt]['cost'] - G[prev][X]['cost'] - G[X][nxt]['cost']
            for nbr in neighbours[X]:
                m = pos_of.get(nbr)
                if m is None or m == n:
                    continue
                # pop(n) then insert(m): X ends up between a and b
                if m < n:
                    a, b = path[m-1], path[m]
                else:
                    a, b = path[m], path[m+1]
                delta = delta_remove + G[a][X]['cost'] + G[X][b]['cost'] - G[a][b]['cost']
                if delta < -IMPROVEMENT_TOLERANCE:
                    item = path.pop(n)
                    path.insert(m, item)
                    return path

        return False
    
    for path in paths:
        while path:
            path = loop(path)

    cost = get_sol_cost(G, paths)

    return (paths, cost)


def interroute_swap(G, paths):
    neighbours = G.graph['neighbours']

    def loop():
        # Global customer -> (route_idx, position) map, rebuilt each pass
        pos_of = {}
        for xi, p in enumerate(paths):
            for k in range(1, len(p) - 1):
                pos_of[p[k]] = (xi, k)

        for x in range(len(paths)):
            indexes_x = list(range(1, len(paths[x]) - 1))
            random.shuffle(indexes_x)
            for n in indexes_x:
                cust_n = paths[x][n]
                path_cost_x = get_neighbour_cost(G, paths[x], n)
                for nbr in neighbours[cust_n]:
                    pos = pos_of.get(nbr)
                    if pos is None:
                        continue
                    y, m = pos
                    if y == x:
                        continue
                    path_cost = path_cost_x + get_neighbour_cost(G, paths[y], m)
                    paths[x][n], paths[y][m] = paths[y][m], paths[x][n]
                    swap_cost = get_neighbour_cost(G, paths[x], n) + get_neighbour_cost(G, paths[y], m)
                    if (swap_cost < path_cost and
                        is_path_feasible(G, paths[x]) and is_path_feasible(G, paths[y])
                        ):
                        return True
                    else:
                        paths[x][n], paths[y][m] = paths[y][m], paths[x][n]

        return False
    
    while loop():
        pass

    cost = get_sol_cost(G, paths)

    return (paths, cost)


def interroute_insertion(G, paths):
    neighbours = G.graph['neighbours']

    def loop():
        # Global customer -> (route_idx, position) map, rebuilt each pass
        pos_of = {}
        for xi, p in enumerate(paths):
            for k in range(1, len(p) - 1):
                pos_of[p[k]] = (xi, k)

        for x in range(len(paths)):
            indexes_x = list(range(1, len(paths[x]) - 1))
            random.shuffle(indexes_x)
            for n in indexes_x:
                X = paths[x][n]
                x_prev = paths[x][n-1]
                x_next = paths[x][n+1]
                delta_remove = G[x_prev][x_next]['cost'] - G[x_prev][X]['cost'] - G[X][x_next]['cost']
                for nbr in neighbours[X]:
                    pos = pos_of.get(nbr)
                    if pos is None:
                        continue
                    y, m = pos
                    if y == x:
                        continue
                    # insert(m) into route y: X ends up between y_prev and y_cur
                    y_prev = paths[y][m-1]
                    y_cur = paths[y][m]
                    delta = (delta_remove
                             + G[y_prev][X]['cost'] + G[X][y_cur]['cost'] - G[y_prev][y_cur]['cost'])
                    if delta < -IMPROVEMENT_TOLERANCE:
                        item = paths[x].pop(n)
                        paths[y].insert(m, item)
                        if is_path_feasible(G, paths[x]) and is_path_feasible(G, paths[y]):
                            return True
                        item = paths[y].pop(m)
                        paths[x].insert(n, item)
        return False
    
    while loop():
        pass

    cost = get_sol_cost(G, paths)

    return (paths, cost) 


def two_opt_swap(G, paths):
    neighbours = G.graph['neighbours']
    sol = []

    def loop(path):
        pos_of = {c: idx for idx, c in enumerate(path) if c != 0}
        indexes = list(range(1, len(path) - 1))
        random.shuffle(indexes)
        for i in indexes:
            cust_i = path[i]
            for nbr in neighbours[cust_i]:
                j = pos_of.get(nbr)
                if j is None or j == i:
                    continue
                x, y = (i, j) if i < j else (j, i)
                if y - x < 2:
                    continue
                # Reversing path[x:y] only swaps the two boundary edges.
                delta = (
                    G[path[x-1]][path[y-1]]['cost'] + G[path[x]][path[y]]['cost']
                    - G[path[x-1]][path[x]]['cost'] - G[path[y-1]][path[y]]['cost']
                )
                if delta < -IMPROVEMENT_TOLERANCE:
                    return path[:x] + path[x:y][::-1] + path[y:]

        sol.append(path)
        return False
    
    for path in paths:
        while path:
            path = loop(path)

    cost = get_sol_cost(G, sol)

    return (sol, cost)


# Sum of nodes does not exceed the maximum capacity
def is_path_feasible(G, path):
    num_customers = sum(1 for a in path if a != 0)
    return (
        sum(G.nodes[a]['demand'][0] for a in path) <= G.graph['capacity'][0]
        and
        sum(G.nodes[a]['demand'][1] for a in path) <= G.graph['capacity'][1]
        and
        get_path_cost(G, path) + num_customers * G.graph['drop_time'] <= G.graph['max_time']
    )


def get_sol_cost(G, sol):
    return sum(get_path_cost(G, path) for path in sol)


def get_path_cost(G, path):
    if (len(path) < 3):
        return 0
    return sum(G[a][b]['cost'] for a, b in zip(path, path[1:]))


def get_neighbour_cost(G, path, i):
    return G[path[i-1]][path[i]]['cost'] + G[path[i]][path[i+1]]['cost']



def print_results(results, num_threads, k_max):

    best_sol, best_cost, best_ex_time = min(results, key=lambda x: x[1])

    print()
    print('Runs:', num_threads)
    print('Max neighborhood:', k_max)
    print()

    print('Best sol:', best_sol)
    print('Best cost:', round(best_cost, 1))
    print('Best time:', round(best_ex_time, 3), 's')
    print()

    average_cost = sum([x[1] for x in results]) / len(results)
    print('Average cost:', round(average_cost, 1))
    average_time = round((sum([x[2] for x in results]) / len(results)), 3)
    print('Average time:', average_time, 's')
    print('Dev:', round(abs(best_cost - average_cost) / best_cost * 100, 1), '%')

    return



def check_parameters():
    parser = argparse.ArgumentParser(description="Routing optimization with solution shaking to escape local optima.")

    parser.add_argument(
        "--instance", 
        type = str,
        required = True,
        help = "Path to the file containing the instance to be executed."
    )

    parser.add_argument(
        "--max-neighborhood", 
        type = int,
        default = 20,
        help = "Maximum neighborhood size to shake the solution and escape local optima (max 50)."
    )

    parser.add_argument(
        "--runs", 
        type = int,
        default = 1,
        help = "Number of times the program will be executed for each instance."
    )

    args = parser.parse_args()

    if args.max_neighborhood > 50:
        parser.error("--max-neighborhood must be less than or equal to 50.")
    
    return args

if __name__ == '__main__':

    args = check_parameters()

    G = init(args.instance)

    runs = int(args.runs)
    k_max = int(args.max_neighborhood)
    
    results = solve(G, k_max, runs)

    print_results(results, runs, k_max)