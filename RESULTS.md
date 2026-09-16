# Results

Results of the GVNS on the 28 MCVRP instances proposed by Abdulkader et al. (2015) (found in `instances/Abdulkader/`), compared with the best known solutions (BKS).

## Instances

The 28 instances (vrpnc1–14, each in an _a_ and _b_ variant) are adapted from Christofides' classic VRP instances, with each customer's demand split between two products, each going in its own vehicle compartment. They have between 50 and 199 customers and fall into four groups:

| Instance group | Customer distribution | Main constraint |
| -------------- | --------------------- | --------------- |
| vrpnc1–5       | Uniform               | Capacity        |
| vrpnc6–10      | Uniform               | Route time      |
| vrpnc11–12     | Clustered             | Capacity        |
| vrpnc13–14     | Clustered             | Route time      |

vrpnc6–10 and vrpnc13–14 use the same customers as vrpnc1–5 and vrpnc11–12, adding a maximum route time and a drop time per customer. The _a_ and _b_ variants differ in how the demand is split between the two compartments.

## Methodology

- The GVNS starts from the nearest neighbour solution and uses $`\rho_{\max} = 0.5`$ (`--max-shake 50`). The local search operators only consider the $`k = 50`$ nearest customers of each customer as move candidates.
- Each instance is solved 1024 times. _Best_ and _Avg_ are the best and average cost over those runs, _Dev_ is the deviation of the average from the best, and _Time_ is the average time of a single run.
- _Gap_ is the distance from the best cost to the BKS. The BKS values are the best results published for these instances, as collected by [Póvoa et al. (2025)](https://doi.org/10.1016/j.eswa.2024.125319). **Bold** marks the instances where the GVNS matches the BKS.

### Setup

Run on an Intel Core i7-12700F (12 physical cores) with 32 GB RAM, using Python 3.14. Runs are executed in parallel, but never with more processes than physical cores, so each run has a core of its own.

## GVNS vs. BKS

| Instance |   n |    BKS |      Best |    Avg | Dev (%) | Gap (%) | Time (s) |
| -------- | --: | -----: | --------: | -----: | ------: | ------: | -------: |
| vrpnc1a  |  50 |  550.7 | **550.7** |  564.4 |     2.5 |     0.0 |     0.37 |
| vrpnc1b  |  50 |  551.9 | **551.9** |  572.5 |     3.7 |     0.0 |     0.45 |
| vrpnc2a  |  75 |  868.6 |     873.9 |  902.1 |     3.2 |     0.6 |     0.84 |
| vrpnc2b  |  75 |  878.7 |     887.0 |  919.6 |     3.7 |     0.9 |     0.75 |
| vrpnc3a  | 100 |  860.4 |     874.3 |  911.8 |     4.3 |     1.6 |     1.58 |
| vrpnc3b  | 100 |  859.4 |     876.3 |  920.9 |     5.1 |     2.0 |     1.60 |
| vrpnc4a  | 150 | 1075.7 |    1109.0 | 1165.0 |     5.0 |     3.1 |     2.82 |
| vrpnc4b  | 150 | 1092.1 |    1141.1 | 1218.4 |     6.8 |     4.5 |     3.26 |
| vrpnc5a  | 199 | 1381.6 |    1423.9 | 1494.5 |     5.0 |     3.1 |     4.92 |
| vrpnc5b  | 199 | 1373.3 |    1462.7 | 1601.8 |     9.5 |     6.5 |     5.14 |
| vrpnc6a  |  50 |  557.5 |     561.1 |  585.6 |     4.4 |     0.6 |     0.45 |
| vrpnc6b  |  50 |  557.5 |     559.4 |  584.4 |     4.5 |     0.3 |     0.48 |
| vrpnc7a  |  75 |  920.5 |     927.9 |  960.0 |     3.5 |     0.8 |     0.95 |
| vrpnc7b  |  75 |  930.7 |     937.0 |  958.0 |     2.2 |     0.7 |     0.80 |
| vrpnc8a  | 100 |  876.7 |     889.2 |  933.8 |     5.0 |     1.4 |     1.61 |
| vrpnc8b  | 100 |  875.3 |     884.8 |  931.4 |     5.3 |     1.1 |     1.49 |
| vrpnc9a  | 150 | 1177.2 |    1218.3 | 1259.1 |     3.4 |     3.5 |     3.24 |
| vrpnc9b  | 150 | 1175.7 |    1217.6 | 1260.1 |     3.5 |     3.6 |     3.32 |
| vrpnc10a | 199 | 1469.7 |    1489.4 | 1571.1 |     5.5 |     1.3 |     5.11 |
| vrpnc10b | 199 | 1470.8 |    1506.8 | 1577.1 |     4.7 |     2.4 |     5.76 |
| vrpnc11a | 120 | 1103.5 |    1229.1 | 1323.8 |     7.7 |    11.4 |     1.47 |
| vrpnc11b | 120 | 1199.0 |    1234.5 | 1314.1 |     6.4 |     3.0 |     1.48 |
| vrpnc12a | 100 |  899.6 |     923.8 |  982.7 |     6.4 |     2.7 |     0.74 |
| vrpnc12b | 100 |  950.8 |     957.6 | 1035.2 |     8.1 |     0.7 |     1.43 |
| vrpnc13a | 120 | 1556.5 |    1559.2 | 1656.0 |     6.2 |     0.2 |     1.86 |
| vrpnc13b | 120 | 1550.1 |    1561.8 | 1653.7 |     5.9 |     0.8 |     1.84 |
| vrpnc14a | 100 |  911.0 |     912.1 |  940.2 |     3.1 |     0.1 |     1.25 |
| vrpnc14b | 100 |  964.7 |     967.6 | 1012.2 |     4.6 |     0.3 |     1.58 |

Average gap to the BKS by instance group:

| Instance group | Instances | Average gap to BKS (%) |
| -------------- | --------: | ---------------------: |
| vrpnc1–5       |        10 |                    2.2 |
| vrpnc6–10      |        10 |                    1.6 |
| vrpnc11–12     |         4 |                    4.4 |
| vrpnc13–14     |         4 |                    0.3 |

## Reproducing the results

```bash
python run_experiments.py --methods gvns:50 --instances all --runs 1024
```

Costs will vary slightly between campaigns because the GVNS is randomised, and times of course depend on the machine.

As 1024 runs per instance take a while (around 2 hours on the machine above), you may want to leave out `--runs` and use the default of 48 runs.

## Other experiments

`run_experiments.py` can also compare the other components of the algorithm:

- The constructive methods (`random`, `nearest`)
- Each local search operator on its own (`intra_swap`, `intra_insertion`, `two_opt`, `inter_swap`, `inter_insertion`)
- A fixed sequence of all of them (`all_ls`)
- The VND with its default or reversed operator order (`vnd`, `vnd_exhaustive_first`)
- The GVNS with other values of $\rho_{\max}$ (`gvns:N` uses $\rho_{\max} = N/100$)

Appending `@K` to any method sets its neighbourhood size $k$ (e.g. `vnd@20`)

Run `python run_experiments.py --help` for more details.
