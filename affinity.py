#!/usr/bin/env python3

import argparse

import numpy as np
import scipy as sp
"""Solve the linear sum assignment problem.
The linear sum assignment problem is also known as minimum weight matching
in bipartite graphs. A problem instance is described by a matrix C, where
each C[i,j] is the cost of matching vertex i of the first partite set
(a "worker") and vertex j of the second set (a "job"). The goal is to find
a complete assignment of workers to jobs of minimal cost.
Formally, let X be a boolean matrix where :math:`X[i,j] = 1` iff row i is
assigned to column j. Then the optimal assignment has cost
.. math::
    \min \sum_i \sum_j C_{i,j} X_{i,j}
s.t. each row is assignment to at most one column, and each column to at
most one row.
This function can also solve a generalization of the classic assignment
problem where the cost matrix is rectangular. If it has more rows than
columns, then not every row needs to be assigned to a column, and vice
versa.
The method used is the Hungarian algorithm, also known as the Munkres or
Kuhn-Munkres algorithm.
Parameters
----------
cost_matrix : array
    The cost matrix of the bipartite graph.
Returns
-------
row_ind, col_ind : array
    An array of row indices and one of corresponding column indices giving
    the optimal assignment. The cost of the assignment can be computed
    as ``cost_matrix[row_ind, col_ind].sum()``. The row indices will be
    sorted; in the case of a square cost matrix they will be equal to
    ``numpy.arange(cost_matrix.shape[0])``.
Notes
-----
.. versionadded:: 0.17.0
Examples
--------
>>> cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
>>> from scipy.optimize import linear_sum_assignment
>>> row_ind, col_ind = linear_sum_assignment(cost)
>>> col_ind
array([1, 0, 2])
>>> cost[row_ind, col_ind].sum()
5
References
----------
1. http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html
2. Harold W. Kuhn. The Hungarian Method for the assignment problem.
   *Naval Research Logistics Quarterly*, 2:83-97, 1955.
3. Harold W. Kuhn. Variants of the Hungarian method for assignment
   problems. *Naval Research Logistics Quarterly*, 3: 253-258, 1956.
4. Munkres, J. Algorithms for the Assignment and Transportation Problems.
   *J. SIAM*, 5(1):32-38, March, 1957.
5. https://en.wikipedia.org/wiki/Hungarian_algorithm
"""

def lin_cost(det_bbox, pred_bbox):
    position_diff = np.asarray(((a.cx - b.cx), (a.cy - b.cy)))
    position_cost = np.sqrt(np.sum(position_diff**2));

    shape_diff = np.asarray(((a.width - b.width), (a.height - b.height)))
    shape_cost = np.sqrt(np.sum(shape_diff**2));

    return position_cost * shape_cost

def exp_cost(det_bbox, pred_bbox):
    position_weight = 0.5
    shape_weight = 1.5

    position_diff = np.asarray((((a.cx - b.cx) / a.width),
                                ((a.cy - b.cy) / a.height)))
    position_cost = np.exp(-position_weight * np.sum(position_diff**2));

    shape_diff = np.asarray((abs(a.width - b.width) / (a.width + b.width),
                             abs(a.height - b.height) / (a.height + b.height)))
    shape_cost = np.exp(-shape_weight * np.sum(shape_diff));

    return position_cost * shape_cost

mapping_id = 1
def mapping(det_bboxes, pred_bboxes, affinity=lin_cost):
    global mapping_id
    id_map = dict()
    for bbox in det_bboxes.itertuples():
        name = bbox.name
        prob = bbox.prob
        # TODO: fix
        # id_map[f"{name}{prob:>1.6f}"] = np.random.randint(1, 100)
        id_map[f"{name}{prob:>1.6f}"] = mapping_id
        mapping_id += 1

    return id_map

def parse_opt():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def main():
    args = parse_opt()
    cost_matrix = np.random.randint(1, 100, size=(12, 12))
    sp.optimize.linear_sum_assignment(cost_matrix)

if __name__ == "__main__":
    main()
