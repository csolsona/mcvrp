import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

import networkx as nx
import numpy as np


# Size of the per-customer neighbourhood used to restrict candidate moves in the local search operators.
NEIGHBOURHOOD_SIZE = 50

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

    # Precomputed distance matrix as a list of lists of native Python floats.
    # Scalar lookups dist[a][b] in the hot loops are far faster than networkx
    # edge access (and faster than numpy scalar indexing). Built vectorised.
    num_nodes = G.number_of_nodes()
    coords = np.array([G.nodes[i]['coords'] for i in range(num_nodes)], dtype=float)
    diffs = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diffs ** 2).sum(axis=2)).tolist()
    G.graph['dist'] = dist

    # Demands as a plain list, for fast feasibility checks without node access.
    G.graph['demands'] = [list(G.nodes[i]['demand']) for i in range(num_nodes)]

    # Precompute, for each customer, the other customers sorted by ascending
    # distance (depot 0 excluded). The active candidate lists used by the
    # operators (G.graph['neighbours']) are a prefix of these, set by
    # set_neighbourhood_size so the size can be varied without rebuilding.
    neighbours_sorted = {}
    customer_ids = list(range(1, num_nodes))
    for i in customer_ids:
        neighbours_sorted[i] = sorted((j for j in customer_ids if j != i), key=lambda j: dist[i][j])
    G.graph['neighbours_sorted'] = neighbours_sorted
    set_neighbourhood_size(G, NEIGHBOURHOOD_SIZE)

    return G


def set_neighbourhood_size(G, size):
    # Restrict each customer's candidate list to its `size` nearest customers,
    # re-slicing from the full sorted lists so no graph rebuild is needed.
    full = G.graph['neighbours_sorted']
    G.graph['neighbours'] = {i: full[i][:size] for i in full}

        

def solve(G, max_shake, num_threads):

    sol, cost = nearest_neighbour_alg(G)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(execute, G, sol, max_shake) for _ in range(num_threads)]

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


def execute(G, sol, max_shake):
        start_time = time.perf_counter()
        sol, cost = gvns(G, sol, max_shake)
        end_time = time.perf_counter()
        execute_time = end_time - start_time
        return (sol, cost, execute_time)


# General Variable neighborhood search (GVNS)
def gvns(G, sol, max_shake):
    shake_intensity = 5
    cost = get_sol_cost(G, sol)
    while (shake_intensity <= max_shake):
        shaken_sol = shake(G, sol, shake_intensity)
        new_sol, new_cost = vnd(G, shaken_sol)
        if (new_cost < cost):
            shake_intensity = 5
            sol = new_sol
            cost = new_cost
        else:
            shake_intensity += 5

    return sol, cost


# Variable Neighborhood Descent (VND)
#
# operators is the ordered list of local-search operators to explore. None uses
# DEFAULT_VND_ORDER (defined after the operators). Passing a different order
# (e.g. reversed) lets the experiments compare orderings without editing this.
def vnd(G, sol, operators=None):
    if operators is None:
        operators = DEFAULT_VND_ORDER

    cost = get_sol_cost(G, sol)
    new_cost = cost

    while (True):
        cost = new_cost
        improved = False
        for operator in operators:
            sol, new_cost = operator(G, sol)
            if (new_cost < cost):
                improved = True
                break
        if not improved:
            break

    return sol, cost


def shake(G, sol, shake_intensity):

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

    nodes_to_change = math.ceil(len(generate_indexes(sol, [])) * (shake_intensity / 100))
    
    def loop():
        # Shallow-copy each route: the elements are ints, so no deepcopy needed.
        shaken_sol = [route[:] for route in sol]
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


def can_satisfy(capacity, demand):
    return (capacity[0] >= demand[0] and capacity[1] >= demand[1])



def random_alg(G):
    dist = G.graph['dist']

    def can_append(path, prev_node, node, capacity, time_limit):
        path.append(node)
        path.append(0)
        capacity[0] -= G.nodes[node]['demand'][0]
        capacity[1] -= G.nodes[node]['demand'][1]
        time_limit -= dist[prev_node][node]
        time_limit -= dist[node][0]
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
            time_limit -= dist[prev_node][node]
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
    dist = G.graph['dist']
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
            cost += dist[start_node][nearest]
            time_limit -= dist[start_node][nearest]
            time_limit -= G.graph['drop_time']
            path.append(nearest)
            capacity[0] -= G.nodes[nearest]['demand'][0]
            capacity[1] -= G.nodes[nearest]['demand'][1]
            start_node = nearest
            nearest = get_nearest_neighbour(G, start_node, capacity, time_limit)
            if nearest == -1:
                path.append(0)
                cost += dist[start_node][0]
                sol.append(path)

        non_visited_nodes = [node for node in G.nodes() if not G.nodes[node]['visited']]
        if (len(non_visited_nodes) == 0):
            break
    
    non_visited_nodes = [node for node in G.nodes() if not G.nodes[node]['visited']]
    if (len(non_visited_nodes) > 0):
        return False
    
    cost = get_sol_cost(G, sol)

    return (sol, cost)


def get_nearest_neighbour(G, start_node, capacity, time_limit):
    dist = G.graph['dist']
    drop_time = G.graph['drop_time']
    neighbors = [
        (neighbour, dist[start_node][neighbour])
        for neighbour in range(1, G.number_of_nodes())
        if (
            neighbour != start_node
            and not G.nodes[neighbour]['visited']
            and can_satisfy(capacity, G.nodes[neighbour]['demand'])
            and dist[start_node][neighbour] + drop_time + dist[neighbour][0] <= time_limit
        )
    ]
    if (len(neighbors) == 0):
        return -1

    neighbors.sort(key=lambda x: x[1])
    return neighbors[0][0]


def intraroute_swap(G, paths):
    neighbours = G.graph['neighbours']
    dist = G.graph['dist']

    def loop(path):
        pos_of = {c: idx for idx, c in enumerate(path) if c != 0}
        indexes = list(range(1, len(path) - 1))
        random.shuffle(indexes)
        for n in indexes:
            cust_n = path[n]
            path_cost_n = get_neighbour_cost(dist, path, n)
            for nbr in neighbours[cust_n]:
                m = pos_of.get(nbr)
                if m is None or m == n:
                    continue
                path_cost = path_cost_n + get_neighbour_cost(dist, path, m)
                path[n], path[m] = path[m], path[n]
                swap_cost = get_neighbour_cost(dist, path, n) + get_neighbour_cost(dist, path, m)
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
    dist = G.graph['dist']

    def loop(path):
        pos_of = {c: idx for idx, c in enumerate(path) if c != 0}
        indexes = list(range(1, len(path) - 1))
        random.shuffle(indexes)
        for n in indexes:
            X = path[n]
            prev = path[n-1]
            nxt = path[n+1]
            # prev == nxt (both depot) when X is the route's only customer:
            # removing it collapses the route, so the closing edge costs 0.
            close = 0 if prev == nxt else dist[prev][nxt]
            delta_remove = close - dist[prev][X] - dist[X][nxt]
            for nbr in neighbours[X]:
                m = pos_of.get(nbr)
                if m is None or m == n:
                    continue
                # pop(n) then insert(m): X ends up between a and b
                if m < n:
                    a, b = path[m-1], path[m]
                else:
                    a, b = path[m], path[m+1]
                delta = delta_remove + dist[a][X] + dist[X][b] - dist[a][b]
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
    dist = G.graph['dist']

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
                path_cost_x = get_neighbour_cost(dist, paths[x], n)
                for nbr in neighbours[cust_n]:
                    pos = pos_of.get(nbr)
                    if pos is None:
                        continue
                    y, m = pos
                    if y == x:
                        continue
                    path_cost = path_cost_x + get_neighbour_cost(dist, paths[y], m)
                    paths[x][n], paths[y][m] = paths[y][m], paths[x][n]
                    swap_cost = get_neighbour_cost(dist, paths[x], n) + get_neighbour_cost(dist, paths[y], m)
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
    dist = G.graph['dist']

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
                # x_prev == x_next (both depot) when X is the route's only
                # customer: removing it collapses the route (closing edge = 0).
                close = 0 if x_prev == x_next else dist[x_prev][x_next]
                delta_remove = close - dist[x_prev][X] - dist[X][x_next]
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
                             + dist[y_prev][X] + dist[X][y_cur] - dist[y_prev][y_cur])
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
    dist = G.graph['dist']
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
                    dist[path[x-1]][path[y-1]] + dist[path[x]][path[y]]
                    - dist[path[x-1]][path[x]] - dist[path[y-1]][path[y]]
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


# Default order in which the VND explores the operators: the cheap intra-route
# neighbourhoods first, the expensive inter-route ones last.
DEFAULT_VND_ORDER = [
    intraroute_swap,
    intraroute_insertion,
    two_opt_swap,
    interroute_swap,
    interroute_insertion,
]


# Single pass over every operator, in order, without restarting from the first
# one on improvement (unlike the VND). Baseline "fixed sequence" for tab:4.
def all_local_searches(G, sol, operators=None):
    if operators is None:
        operators = DEFAULT_VND_ORDER
    for operator in operators:
        sol, cost = operator(G, sol)
    return sol, get_sol_cost(G, sol)


# Sum of nodes does not exceed the maximum capacity
def is_path_feasible(G, path):
    demands = G.graph['demands']
    capacity = G.graph['capacity']
    num_customers = sum(1 for a in path if a != 0)
    return (
        sum(demands[a][0] for a in path) <= capacity[0]
        and
        sum(demands[a][1] for a in path) <= capacity[1]
        and
        get_path_cost(G, path) + num_customers * G.graph['drop_time'] <= G.graph['max_time']
    )


def get_sol_cost(G, sol):
    return sum(get_path_cost(G, path) for path in sol)


def get_path_cost(G, path):
    if (len(path) < 3):
        return 0
    dist = G.graph['dist']
    return sum(dist[a][b] for a, b in zip(path, path[1:]))


def get_neighbour_cost(dist, path, i):
    return dist[path[i-1]][path[i]] + dist[path[i]][path[i+1]]



def print_results(results, num_threads, max_shake):

    best_sol, best_cost, best_ex_time = min(results, key=lambda x: x[1])

    print()
    print('Runs:', num_threads)
    print('Max shake:', max_shake)
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
        "--max-shake",
        type = int,
        default = 50,
        help = "Maximum shake intensity as a percentage of customers relocated (rho_max, max 50)."
    )

    parser.add_argument(
        "--neighbourhood-size",
        type = int,
        default = NEIGHBOURHOOD_SIZE,
        help = "Number of nearest customers each local-search operator considers as move candidates (k)."
    )

    parser.add_argument(
        "--runs",
        type = int,
        default = 1,
        help = "Number of times the program will be executed for each instance."
    )

    args = parser.parse_args()

    if args.max_shake > 50:
        parser.error("--max-shake must be less than or equal to 50.")

    return args

if __name__ == '__main__':

    args = check_parameters()

    G = init(args.instance)
    set_neighbourhood_size(G, args.neighbourhood_size)

    runs = int(args.runs)
    max_shake = int(args.max_shake)

    results = solve(G, max_shake, runs)

    print_results(results, runs, max_shake)