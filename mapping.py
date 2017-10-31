#!/usr/bin/env python3

import argparse

import numpy as np
import scipy as sp
from scipy import optimize

import pandas as pd

def _convert_bbox(bbox):
    left = bbox.left
    top = bbox.top
    width = bbox.right - bbox.left
    height = bbox.bot - bbox.top

    return pd.Series({
        "left": left, "top": top, "width": width, "height": height
    })

def lin_cost(next_bbox, prev_bbox):
    a = _convert_bbox(next_bbox)
    b = _convert_bbox(prev_bbox)

    position_diff = np.asarray(((a.left - b.left), (a.top - b.top)))
    position_cost = np.sqrt(np.sum(position_diff**2))
    position_cost += 1

    shape_diff = np.asarray(((a.width - b.width), (a.height - b.height)))
    shape_cost = np.sqrt(np.sum(shape_diff**2))
    shape_cost += 1

    return position_cost * shape_cost

def exp_cost(next_bbox, prev_bbox):
    a = _convert_bbox(next_bbox)
    b = _convert_bbox(prev_bbox)

    position_weight = 0.5
    shape_weight = 1.5

    position_diff = np.asarray((((a.left - b.left) / a.width),
                                ((a.top - b.top) / a.height)))
    position_cost = np.exp(-position_weight * np.sum(position_diff**2));

    shape_diff = np.asarray((abs(a.width - b.width) / (a.width + b.width),
                             abs(a.height - b.height) / (a.height + b.height)))
    shape_cost = np.exp(-shape_weight * np.sum(shape_diff));

    return position_cost * shape_cost

def iou_cost(next_bbox, prev_bbox):
    cap_left = max(next_bbox.left, prev_bbox.left)
    cap_top = max(next_bbox.top, prev_bbox.top)
    cap_right = min(next_bbox.right, prev_bbox.right)
    cap_bot = min(next_bbox.bot, prev_bbox.bot)

    if cap_left <= cap_right and cap_top <= cap_bot:
        area_cap = (cap_right - cap_left + 1) * (cap_bot - cap_top + 1)
    else:
        area_cap = 0

    area_next = (next_bbox.right - next_bbox.left + 1) \
              * (next_bbox.bot - next_bbox.top + 1)
    area_prev = (prev_bbox.right - prev_bbox.left + 1) \
              * (prev_bbox.bot - prev_bbox.top + 1)

    area_cup = (area_next + area_prev - area_cap)

    iou = area_cap / area_cup

    return 1.0 - iou

def calc_cost(src_bboxes, dst_bboxes, affinity=lin_cost):
    cost_matrix = np.zeros((src_bboxes.shape[0], dst_bboxes.shape[0]))

    for src_bbox in src_bboxes.itertuples():
        for dst_bbox in dst_bboxes.itertuples():
            cost_matrix[src_bbox.Index, dst_bbox.Index] = \
                affinity(dst_bbox, src_bbox)

    return cost_matrix

class Mapper:
    def __init__(self):
        pass

    def set(self, next_bboxes, prev_bboxes):
        pass

    def get(self, bbox):
        pass

class SimpleMapper(Mapper):
    def __init__(self, affinity=iou_cost, cost_thresh=1.0):
        self.affinity = affinity
        self.cost_thresh = cost_thresh
        self.id_count = 1
        self.ids = dict()

    def _assign_id(self):
        new_id = self.id_count
        self.id_count += 1
        return new_id

    def set(self, next_bboxes, prev_bboxes):
        cost = calc_cost(prev_bboxes, next_bboxes, self.affinity)
        row_idx, col_idx = sp.optimize.linear_sum_assignment(cost)

        id_map = dict()
        for bbox in next_bboxes.itertuples():
            # id_map[bbox.Index] = self._assign_id()
            if bbox.Index in col_idx:
                arg_idx = np.where(col_idx == bbox.Index)[0][0]
                trans_cost = cost[row_idx, col_idx][arg_idx]
                if trans_cost <= self.cost_thresh:
                    id_map[bbox.Index] = self.ids[row_idx[arg_idx]]
                else:
                    id_map[bbox.Index] = self._assign_id()
            else:
                id_map[bbox.Index] = self._assign_id()

        # self.ids = id_map
        self.ids.update(id_map)

    def get(self, bboxes):
        for bbox in bboxes.itertuples():
            yield self.ids[bbox.Index], bbox

def parse_opt():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def main():
    args = parse_opt()
    det_bbox = pd.DataFrame({
        "name": ("det0", "det1"),
        "prob": (0.6, 0.4),
        "left": (100, 50), "top": (100, 150),
        "right": (100+200, 50+50), "bot": (100+100, 150+100)
    })
    pred_bbox = pd.DataFrame({
        "name": ("pred0", "pred1", "pred2"),
        "prob": (0.3, 0.7, 0.1),
        "left": (200, 300, 20), "top": (200, 100, 30),
        "right": (200+200, 300+100, 20+120), "bot": (200+200, 150+50, 30+130)
    })
    print(lin_cost(det_bbox.loc[0], pred_bbox.loc[0]))
    print(exp_cost(det_bbox.loc[0], pred_bbox.loc[0]))
    calc_cost(det_bbox, pred_bbox)
    cost = np.random.randint(1, 100, size=(12, 6))
    row_idx, col_idx = sp.optimize.linear_sum_assignment(cost)
    print(row_idx, col_idx)
    cost = np.random.randint(1, 100, size=(6, 6)).astype(np.float32)
    row_idx, col_idx = sp.optimize.linear_sum_assignment(cost)
    print(row_idx, col_idx)
    cost = np.random.randint(1, 100, size=(6, 12))
    row_idx, col_idx = sp.optimize.linear_sum_assignment(cost)
    print(row_idx, col_idx)

if __name__ == "__main__":
    main()

