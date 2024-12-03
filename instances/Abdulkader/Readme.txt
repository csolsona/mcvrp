This file contains the data set of the Multi-compartment VRP used in the following paper.

  "Hybridized Ant colony algorithm for the multi compartment vehicle routing problem", M.M.S. Abdulkader, Yuvraj Gajpal, Tarek ElMekkawy, Applied Soft Computing Volume 37, December 2015, Pages 196–203

This paper used the following 28 data sets

Problem1
Problem1b
Problem2
Problem2b
...
...
Problem14
Problem14b

The format of data files is as follows:
 
The first line contains the follwing information:
0 	- Customer number of the depot
X[0]	- X-coordinate of customer 0 (depot)
Y[0]	- Y-coordinate of customer 0 (depot)
Q1	- Vehicle capacity for product 1
Q2	- Vehicle capacity for product 2
n	- Number of customers
Rt	- Maximum route time
Dt	- Drop time

The next n lines contain the follwing information for each customer:
i	- Customer number
X[i]	- X-coordinate of customer i 
Y[i]	- Y-coordinate of customer i 
D1[i]	- Demand of customer i for product 1
D2[i]	- Demand of customer i for product 2

