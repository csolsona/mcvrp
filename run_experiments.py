#!/usr/bin/env python3
"""Configurable experiment runner for the MCVRP results tables.

Every "method" (a constructive heuristic, a single local-search operator, the
VND, or the GVNS) is run `--runs` times and reported as one raw-data block per
instance:

    <label> [best <cost> | avg <cost> | dev <%> | time <s>]

Timing is measured per single run with no CPU contention: the `--runs`
repetitions are submitted to a ProcessPoolExecutor whose worker count is capped
at the number of *physical* cores, so each run owns a core for its whole
lifetime and its wall-clock time is the real cost of one execution (the pool
schedules the runs in waves automatically). The cost columns (best/avg/dev) are
aggregated over all the runs.

The initial solution for the improvement methods (local search, VND, GVNS) is
always nearest-neighbour, exactly as in mcvrp.solve().

Examples:
    # tab:1 (constructive comparison)
    python run_experiments.py --methods random,nearest --instances sample --runs 48

    # tab:2 (local searches separately)
    python run_experiments.py --instances sample --runs 48 \
        --methods intra_swap,intra_insertion,two_opt,inter_swap,inter_insertion

    # tab:5/tab:6 (GVNS max_shake sweep + VND)
    python run_experiments.py --instances sample --runs 48 \
        --methods vnd,gvns:10,gvns:20,gvns:30,gvns:40,gvns:50

    # tab:7/tab:8 (full instance set)
    python run_experiments.py --instances all --runs 48 --methods all_ls,vnd,gvns:50
"""

import argparse
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import mcvrp


# 8-instance sample used in tab:1-6. Chosen to cover the size range, both
# constraint regimes (capacity only vs +route-time/drop), uniform vs clustered
# layouts and both compartment-balance variants (a/b). Numeric order.
SAMPLE = [
    'Abdulkader/vrpnc1a.txt',
    'Abdulkader/vrpnc3b.txt',
    'Abdulkader/vrpnc5b.txt',
    'Abdulkader/vrpnc7a.txt',
    'Abdulkader/vrpnc9a.txt',
    'Abdulkader/vrpnc10a.txt',
    'Abdulkader/vrpnc11a.txt',
    'Abdulkader/vrpnc14b.txt',
]

# All 28 reference instances (tab:7-8).
ALL = [
    'Abdulkader/vrpnc1a.txt', 'Abdulkader/vrpnc1b.txt',
    'Abdulkader/vrpnc2a.txt', 'Abdulkader/vrpnc2b.txt',
    'Abdulkader/vrpnc3a.txt', 'Abdulkader/vrpnc3b.txt',
    'Abdulkader/vrpnc4a.txt', 'Abdulkader/vrpnc4b.txt',
    'Abdulkader/vrpnc5a.txt', 'Abdulkader/vrpnc5b.txt',
    'Abdulkader/vrpnc6a.txt', 'Abdulkader/vrpnc6b.txt',
    'Abdulkader/vrpnc7a.txt', 'Abdulkader/vrpnc7b.txt',
    'Abdulkader/vrpnc8a.txt', 'Abdulkader/vrpnc8b.txt',
    'Abdulkader/vrpnc9a.txt', 'Abdulkader/vrpnc9b.txt',
    'Abdulkader/vrpnc10a.txt', 'Abdulkader/vrpnc10b.txt',
    'Abdulkader/vrpnc11a.txt', 'Abdulkader/vrpnc11b.txt',
    'Abdulkader/vrpnc12a.txt', 'Abdulkader/vrpnc12b.txt',
    'Abdulkader/vrpnc13a.txt', 'Abdulkader/vrpnc13b.txt',
    'Abdulkader/vrpnc14a.txt', 'Abdulkader/vrpnc14b.txt',
]


def physical_cores():
    """Number of physical cores, so parallel runs never share a core.

    Reads /proc/cpuinfo and counts distinct (physical id, core id) pairs;
    falls back to os.cpu_count() when that information is unavailable.
    """
    try:
        cores = set()
        current = {}
        with open('/proc/cpuinfo') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if 'physical id' in current and 'core id' in current:
                        cores.add((current['physical id'], current['core id']))
                    current = {}
                    continue
                key, _, value = line.partition(':')
                key = key.strip()
                if key in ('physical id', 'core id'):
                    current[key] = value.strip()
        if cores:
            return len(cores)
    except OSError:
        pass
    return os.cpu_count() or 1


# --- Initial solution -------------------------------------------------------

def build_initial(G):
    """Initial solution for the improvement methods, as in mcvrp.solve()."""
    sol, _ = mcvrp.nearest_neighbour_alg(G)
    return sol


# --- Methods ----------------------------------------------------------------
#
# Constructive methods build a solution from scratch: signature (G) -> (sol, cost).
# Improvement methods refine an initial solution: signature (G, sol) -> (sol, cost).
# Every method is a top-level function so the worker can be pickled.

def c_random(G):
    return mcvrp.random_alg(G)


def c_nearest(G):
    return mcvrp.nearest_neighbour_alg(G)


def i_intra_swap(G, sol):
    return mcvrp.intraroute_swap(G, sol)


def i_intra_insertion(G, sol):
    return mcvrp.intraroute_insertion(G, sol)


def i_two_opt(G, sol):
    return mcvrp.two_opt_swap(G, sol)


def i_inter_swap(G, sol):
    return mcvrp.interroute_swap(G, sol)


def i_inter_insertion(G, sol):
    return mcvrp.interroute_insertion(G, sol)


def i_all_ls(G, sol):
    return mcvrp.all_local_searches(G, sol)


def i_vnd(G, sol):
    return mcvrp.vnd(G, sol)


def i_vnd_exhaustive_first(G, sol):
    return mcvrp.vnd(G, sol, list(reversed(mcvrp.DEFAULT_VND_ORDER)))


# kind: 'construct' or 'improve'
CONSTRUCTIVE = {
    'random': c_random,
    'nearest': c_nearest,
}

IMPROVEMENT = {
    'intra_swap': i_intra_swap,
    'intra_insertion': i_intra_insertion,
    'two_opt': i_two_opt,
    'inter_swap': i_inter_swap,
    'inter_insertion': i_inter_insertion,
    'all_ls': i_all_ls,
    'vnd': i_vnd,
    'vnd_exhaustive_first': i_vnd_exhaustive_first,
}

# Canonical order of every method, used when --methods is omitted.
ALL_METHODS = (
    list(CONSTRUCTIVE)
    + list(IMPROVEMENT)
    + ['inter_swap@5', 'inter_insertion@5']
    + ['gvns:10', 'gvns:20', 'gvns:30', 'gvns:40', 'gvns:50']
)


def parse_method(method_name):
    """Split an optional '@SIZE' neighbourhood suffix off a method name.

    Returns (base_name, size) where size is None when no '@SIZE' is given. The
    suffix sets the local-search neighbourhood size for that run, e.g.
    'inter_swap@30' or 'vnd@50'.
    """
    size = None
    if '@' in method_name:
        method_name, size_str = method_name.split('@', 1)
        size = int(size_str)
    return method_name, size


def resolve(method_name):
    """Return (kind, fn, kwargs) for a method name. 'gvns:N' sets max_shake=N."""
    if method_name.startswith('gvns:'):
        max_shake = int(method_name.split(':', 1)[1])
        return 'improve', mcvrp.gvns, {'max_shake': max_shake}
    if method_name in CONSTRUCTIVE:
        return 'construct', CONSTRUCTIVE[method_name], {}
    if method_name in IMPROVEMENT:
        return 'improve', IMPROVEMENT[method_name], {}
    raise ValueError(f"Unknown method '{method_name}'")


# --- Worker (one repetition) ------------------------------------------------

def run_one(task):
    """One repetition of one method on one instance, run in a worker process.

    Builds its own graph (and initial solution for improvement methods) outside
    the timed region, then times only the method itself.
    """
    file_name, method_name, default_size = task
    # Workers are forked and inherit the parent's RNG state, which would make
    # the repetitions correlated. Reseed from OS entropy so every run is an
    # independent sample (needed for a meaningful best-of-R and deviation).
    random.seed()
    base_name, size = parse_method(method_name)
    if size is None:
        size = default_size  # global --neighbourhood, or None to keep the default

    G = mcvrp.init(file_name)
    if size is not None:
        mcvrp.set_neighbourhood_size(G, size)
    kind, fn, kwargs = resolve(base_name)

    if kind == 'construct':
        start = time.perf_counter()
        sol, cost = fn(G)
        elapsed = time.perf_counter() - start
    else:
        sol = build_initial(G)
        start = time.perf_counter()
        sol, cost = fn(G, sol, **kwargs)
        elapsed = time.perf_counter() - start

    return cost, elapsed, sol


# --- Validation & reporting -------------------------------------------------

def is_solution_valid(G, sol):
    expected = set(G.nodes) - {0}
    served = {n for path in sol for n in path if n != 0}
    return served == expected and all(mcvrp.is_path_feasible(G, p) for p in sol)


def format_time(t):
    """Time with 3 decimals, in seconds or milliseconds as needed.

    The short local searches run in fractions of a millisecond, so reporting
    everything in seconds would collapse them to a single figure (e.g. 0.000).
    Switching to milliseconds keeps a few significant figures for those cases.
    """
    if t >= 1:
        return f"{t:.3f} s"
    return f"{t * 1000:.3f} ms"


def format_line(label, width, best, avg, dev, avg_time):
    return (
        f"  {label.ljust(width)} "
        f"[best {best:9.1f} | avg {avg:9.1f} | "
        f"dev {dev:4.1f}% | time {format_time(avg_time):>10}]"
    )


def build_report(instances, methods, runs, workers, neighbourhood):
    width = max(len(m) for m in methods)
    default_neighbourhood = neighbourhood if neighbourhood else mcvrp.NEIGHBOURHOOD_SIZE
    lines = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for file_name in instances:
            print(f"Running instance: {file_name}")
            G = mcvrp.init(file_name)
            num_customers = G.number_of_nodes() - 1
            lines.append(f"Instance: {file_name}")
            lines.append(f"  customers: {num_customers}")
            lines.append(f"  runs: {runs} | workers: {workers} | "
                         f"neighbourhood: {default_neighbourhood}")

            for method_name in methods:
                futures = [executor.submit(run_one, (file_name, method_name, neighbourhood))
                           for _ in range(runs)]
                results = [f.result() for f in futures]

                if not all(is_solution_valid(G, sol) for _, _, sol in results):
                    lines.append(f"  {method_name.ljust(width)} [INVALID SOLUTION]")
                    print(f"  WARNING: {method_name} produced an invalid solution "
                          f"on {file_name}")
                    continue

                costs = [r[0] for r in results]
                times = [r[1] for r in results]
                best = min(costs)
                avg = sum(costs) / len(costs)
                # max(0.0, ...) avoids a spurious "-0.0" when every run has the
                # same cost: avg can fall a hair below best by float rounding.
                dev = max(0.0, (avg - best) / best * 100) if best > 0 else 0.0
                avg_time = sum(times) / len(times)
                lines.append(format_line(method_name, width, best, avg, dev, avg_time))

            lines.append("")

    return "\n".join(lines)


def check_parameters():
    parser = argparse.ArgumentParser(
        description="Run MCVRP experiments and report raw data blocks per method.")
    parser.add_argument(
        "--methods", type=str, default=None,
        help="Comma-separated method names (default: every method). Constructive: "
             "random, nearest. Local searches: intra_swap, intra_insertion, "
             "two_opt, inter_swap, inter_insertion. Composite: all_ls, vnd, "
             "vnd_exhaustive_first. "
             "Metaheuristic: gvns:K (K = max_shake, e.g. gvns:50). "
             "Append '@SIZE' to any method to set its neighbourhood size for that "
             "run (e.g. inter_swap@30), overriding --neighbourhood.")
    parser.add_argument(
        "--instances", type=str, default="sample",
        help="'sample' (8 instances), 'all' (28), or a comma-separated list of "
             "instance paths relative to instances/.")
    parser.add_argument(
        "--runs", type=int, default=48,
        help="Repetitions per method (same for every table).")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel workers (default: detected physical cores, so per-run "
             "times are measured without CPU contention).")
    parser.add_argument(
        "--neighbourhood", type=int, default=None,
        help="Default neighbourhood size (nearest customers considered by the "
             "local-search operators) for methods without a '@SIZE' suffix "
             f"(default: mcvrp's built-in, {mcvrp.NEIGHBOURHOOD_SIZE}).")
    parser.add_argument(
        "--output-file", type=str,
        help="Write the report to this file instead of stdout.")
    return parser.parse_args()


def resolve_instances(spec):
    if spec == "sample":
        return SAMPLE
    if spec == "all":
        return ALL
    return [s.strip() for s in spec.split(",") if s.strip()]


if __name__ == '__main__':
    args = check_parameters()

    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        methods = list(ALL_METHODS)
    for m in methods:
        resolve(parse_method(m)[0])  # validate names (and @SIZE) early

    instances = resolve_instances(args.instances)
    workers = args.workers if args.workers else physical_cores()

    report = build_report(instances, methods, args.runs, workers, args.neighbourhood)

    if args.output_file:
        with open(args.output_file, "w") as out_file:
            out_file.write(report)
    else:
        print(report)
