# Multi-Compartment Vehicle Routing Problem (MCVRP) Solver

This repository contains Python scripts for solving the Multi-Compartment Vehicle Routing Problem (MCVRP) using a solution shaking technique to escape local optima. The main execution script is `mcvrp.py`, which runs an instance of the MCVRP problem and prints the results.

### Files

- `mcvrp.py`: The core script that implements the MCVRP solver using General Variable Neighborhood Search (GVNS) and other optimization techniques.

- `run_instances.py`: Script to execute multiple MCVRP instances.

- `instances/`: Directory containing the MCVRP instances to be solved.

### Requirements

- Python >= 3.11

- NetworkX library (`pip install networkx`)

- NumPy library (`pip install numpy`)

## Usage

#### Running a Single Instance

To run the solver on a single instance, you can use the `mcvrp.py` script directly. The script requires the following command-line arguments:

- `--instance`: Path to the instance file (from the instaces directory).

- `--max-neighborhood`: Maximum neighborhood size for shaking the solution (default is 20, max is 50).

- `--runs`: Number of times to run the solver for the instance (default is 1).

Example command:
```bash
python mcvrp.py --instance Abdulkader/vrpnc1a.txt --max-neighborhood 20 --runs 12
```

#### Running Multiple Instances

To run the solver on multiple instances, use the `run_instances.py` script. This script will execute the solver on a predefined list of instances and can optionally save the results to an output file.

- `--output-file`: Path to the output file where results will be saved (optional).

- `--max-neighborhood`: Maximum neighborhood size for shaking the solution (default is 20, max is 50).

- `--runs`: Number of times to run the solver for each instance (default is 1).

Example command:
```bash
python run_instances.py --output-file results.txt --max-neighborhood 20 --runs 12
```

If no output file is specified, the results will be printed to the console.

## Instance Files

The instance files are located in the `instances/` directory. Each file represents a different instance of the MCVRP problem. The solver will run on all instances listed in the file_names array within `run_instances.py`. There are some example instances in the `instances/Abdulkader/` directory, which have been used to benchmark this algorithm.

### Output

The output includes the following information for each instance:

- Best solution: The best solution found.
- Best cost: The cost of the best solution.
- Best execution time: The time taken to find the best solution.
- Average cost: The average cost across all runs.
- Average time: The average execution time across all runs.
- Deviation: The percentage deviation of the average cost from the best cost.

### Custom Instances

You have information about the format of the instances files in the [Readme file](https://github.com/csolsona/mcvrp/blob/7b1af1de2492536f802ba284b1a5f8bba469dc5d/instances/Abdulkader/Readme.txt), which you can use to build your own custom instances.

To run the solver on custom instances, place your instance files in the `instances/` directory and update the `file_names` list in `run_instances.py` to include your files.


## Acknowledgments

This codebase incorporates ideas and instances presented in the following papers, among others:
- [C. L. Ramos Póvoa, M. Costa Roboredo, A. Soares Velasco, A. Alves Pessoa, F. Galaxe Paes. A hybrid GRASP and tabu-search heuristic and an exact method for a variant of the multi-compartment vehicle routing problem, 2024](https://doi.org/10.1016/j.eswa.2024.125319)

- [M. M. S. Abdulkader, Y. Gajpal, T. Y. ElMekkawy. Hybridized ant colony algorithm for the Multi Compartment Vehicle Routing Problem, 2015](https://doi.org/10.1016/j.asoc.2015.08.020)