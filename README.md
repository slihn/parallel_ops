parallel_ops
============

A python script that demonstrates the response time of a cloud-based computation

# Introduction

Many finance analytics have two phases of calculation - 
The first phase is to read the raw positions from the database and construct a monthly/daily view.
The second phase is to aggregate the monthly/daily view to a historical range spanning multiple years.
How much parallelism you build into these two phases will determine the real-time response of your analytics -
whether you have a system of O(N), O(log N) or O(1).

# How to Run It

* Clone the project
* Modify the two latency variables to reflect relative cost of these two opreations.
* Run the script with Python 2.7
