#!/usr/bin/env python3

import os
import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import trange

from flow import get_flow, draw_flow
from annotate import pick_bbox, draw_bboxes
from draw import draw_none
from vis import open_video
from interp import draw_i_frame, draw_p_frame, find_inner, calc_flow_mean
from interp import interp_linear_unit, interp_divide_unit

def calc_center(bbox):
    center_y = np.mean((bbox.bot, bbox.top))
    center_x = np.mean((bbox.right, bbox.left))
    center = np.asarray((center_x, center_y))

    return center

def convert(bbox):
    left = bbox.left
    top = bbox.top
    width = bbox.right - left
    height = bbox.bot - top
    aspect = width / height
    return np.asarray((left, top, aspect, height), dtype=np.float32)

def invert(bbox):
    left = bbox[0]
    top = bbox[1]
    aspect = bbox[2]
    height = bbox[3]
    bot = top + height
    width = aspect * height
    right = left + width
    return pd.Series({"left": left, "top": top, "right": right, "bot": bot})

import filterpy.kalman
# delegated class
class KalmanInterpolator:
    def __init__(self, dp=2, mp=2, cp=2, processNoise=1e-0, measurementNoise=1e-1):
        # self.kalman = cv2.KalmanFilter(dp, mp, cp)
        # self.kalman.transitionMatrix = np.float32(1.0 * np.eye(dp, dp))
        # self.kalman.controlMatrix = np.float32(1.0 * np.eye(dp, cp))
        # self.kalman.measurementMatrix = np.float32(1.0 * np.eye(mp, dp))
        # self.kalman.processNoiseCov = np.float32(processNoise * np.eye(dp, dp))
        # self.kalman.measurementNoiseCov = np.float32(measurementNoise * np.eye(mp, mp))

        self.kalman = filterpy.kalman.KalmanFilter(dim_x=dp, dim_z=mp, dim_u=cp)
        self.kalman.F = np.float32(1.0 * np.eye(dp, dp))
        self.kalman.B = np.float32(1.0 * np.eye(dp, cp))
        self.kalman.H = np.float32(1.0 * np.eye(mp, dp))
        self.kalman.Q = np.float32(processNoise * np.eye(dp, dp))
        self.kalman.R = np.float32(measurementNoise * np.eye(mp, mp))

        self.total = 0
        self.count = 0
        self.stateList = []
        self.errorCovList = []

        self.dp, self.mp, self.cp = dp, mp, cp

    def init(self, dp, mp, cp):
        self.kalman.init(dp, mp, cp)

    def reset(self, bboxes):
        self.total = 0
        self.count = 0
        self.stateList = []
        self.errorCovList = []

        for bbox in bboxes.itertuples():
            self.total += 1
            self.stateList.append(
                calc_center(bbox).astype(np.float32))

        for i in range(self.total):
            self.errorCovList.append(
                    (1.0 * np.eye(self.dp, self.dp)).astype(np.float32))

    def predict(self, control):
        # self.kalman.statePost = self.stateList[self.count]
        # self.kalman.errorCovPost = self.errorCovList[self.count]
        self.kalman.x = self.stateList[self.count]
        self.kalman.P = self.errorCovList[self.count]
        self.kalman.predict(control)
        return self.kalman.x.reshape(self.mp, 1)

    def update(self, measurement):
        # result = self.kalman.correct(measurement)
        # self.stateList[self.count] = self.kalman.statePost
        # self.errorCovList[self.count] = self.kalman.errorCovPost
        self.kalman.update(measurement)
        self.stateList[self.count] = self.kalman.x
        self.errorCovList[self.count] = self.kalman.P

    def filter(self, center, flow_mean):
        flow_mean = flow_mean.astype(np.float32)
        state = self.predict(flow_mean)
        noise = np.random.randn(self.mp, 1)

        # new_center = np.dot(self.kalman.measurementMatrix, state) \
        #            + np.dot(self.kalman.measurementNoiseCov, noise)
        new_center = np.dot(self.kalman.H, state) \
                   + np.dot(self.kalman.R, noise)
        new_center = new_center.flatten().astype(np.float32)

        self.update(new_center)

        if self.count == self.total-1:
            self.count = 0
        else:
            self.count += 1

        return state[0:self.mp].flatten()

def divergence(field):
    grad_x = np.gradient(field[:, :, 0], axis=1)
    grad_y = np.gradient(field[:, :, 1], axis=0)
    div = grad_x + grad_y
    return div

def calc_flow_mean_kalman(inner_flow, kalman):
    if inner_flow.shape[0] > 1 and inner_flow.shape[1] > 1:
        # grad_x = np.abs(np.gradient(inner_flow[:, :, 0], axis=1))
        # if np.sum(grad_x) == 0:
        #     grad_x = np.ones_like(grad_x)
        # grad_y = np.abs(np.gradient(inner_flow[:, :, 1], axis=0))
        # if np.sum(grad_y) == 0:
        #     grad_y = np.ones_like(grad_y)
        # div_flow = np.stack((grad_x, grad_y), axis=-1)
        div_flow = np.abs(divergence(inner_flow))
        if np.sum(div_flow) == 0:
            div_flow = np.ones_like(div_flow)
        div_flow = np.stack((div_flow, div_flow), axis=-1)
        flow_mean = np.average(inner_flow, axis=(0, 1), weights=div_flow)
    else:
        flow_mean = np.mean(inner_flow, axis=(0, 1))

    # kalman.update(flow_mean)
    # kalman.predict()
    # new_flow_mean = kalman.x
    new_flow_mean = flow_mean

    return np.nan_to_num(new_flow_mean)

def sigmoid(x, alpha=20.0, scale=1.0, offset=(0.2, 1.0)):
# def sigmoid(x, alpha=10.0, scale=2.0, offset=(0.5, 1.0)):
    return scale / (1 + np.exp(-alpha*(x-offset[0]))) + offset[1]

def interp_kalman_unit(bbox, flow_mean, frame, kalman):
    # size_rate = ((bbox.right - bbox.left) * (bbox.bot-bbox.top)) \
    #           / (frame.shape[0] * frame.shape[1])
    # size_rate = np.sqrt(size_rate)
    # flow_mean *= sigmoid(size_rate)
    # flow_mean *= 1 + np.nan_to_num(size_rate)

    center = calc_center(bbox)
    new_center = kalman.filter(center, flow_mean)
    frame_mean = new_center - center

    left  = bbox.left + frame_mean[0]
    top   = bbox.top + frame_mean[1]
    right = bbox.right + frame_mean[0]
    bot   = bbox.bot + frame_mean[1]

    height = frame.shape[0]
    width = frame.shape[1]

    left  = np.clip(left, 0, width-1).astype(np.int)
    top   = np.clip(top, 0, height-1).astype(np.int)
    right = np.clip(right, 0, width-1).astype(np.int)
    bot   = np.clip(bot, 0, height-1).astype(np.int)

    return pd.Series({"name": bbox.name, "prob": bbox.prob,
        "left": left, "top": top, "right": right, "bot": bot
        ,"velo": f"{flow_mean}"
        })

def interp_kalman(bboxes, flow, frame, kalman):
    frame_rows = frame.shape[0]
    frame_cols = frame.shape[1]
    assert(frame.shape[2] == 3)

    rows = flow.shape[0]
    cols = flow.shape[1]
    assert(flow.shape[2] == 2)

    flow_index = np.asarray(tuple(np.ndindex((rows, cols))))
    index_rate = np.asarray((frame_rows // rows, frame_cols // cols))
    frame_index = flow_index * index_rate + (index_rate // 2)

    for bbox in bboxes.itertuples():
        inner_flow = find_inner(flow, bbox, flow_index, frame_index)

        # flow_mean = calc_flow_mean(inner_flow)
        # bboxes.loc[bbox.Index] = interp_kalman_unit(bbox, flow_mean, frame,
        #                                             kalman)

        # flow_mean = calc_flow_mean_kalman(inner_flow, bbox.kalman)
        flow_mean = calc_flow_mean_kalman(inner_flow, None)
        bboxes.loc[bbox.Index] = interp_kalman_unit(bbox, flow_mean, frame,
                                                    kalman)
        # bboxes.loc[bbox.Index] = interp_linear_unit(bbox, flow_mean, frame)

        # bboxes.loc[bbox.Index] = interp_divide_unit(bbox, inner_flow, frame)

    return bboxes

def vis_kalman(movie, header, flow, bboxes, base=False, worst=False):
    if base and worst:
        raise "rendering mode could not be duplicated"

    if base:
        cap, out = open_video(movie, postfix="base")
    elif worst:
        cap, out = open_video(movie, postfix="worst")
    else:
        cap, out = open_video(movie)

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    kalman = KalmanInterpolator()
    interp_kalman_clos = lambda bboxes, flow, frame: \
            interp_kalman(bboxes, flow, frame, kalman)

    pos = 0
    for i in trange(count):
        ret, frame = cap.read()
        if ret is False or i > bboxes.size:
            break

        if base:
            pos = i
            frame = draw_i_frame(frame, flow[i], bboxes[pos])
        elif header["pict_type"][i] == "I":
            pos = i
            frame = draw_i_frame(frame, flow[i], bboxes[pos])
            kalman.reset(bboxes[pos])
        elif worst:
            frame = draw_i_frame(frame, flow[i], bboxes[pos])
        else:
            # bboxes[pos] is updated by reference
            frame = draw_p_frame(frame, flow[i], bboxes[pos],
                                 interp=interp_kalman_clos)

        cv2.rectangle(frame, (width-220, 20), (width-20, 60), (0, 0, 0), -1)
        cv2.putText(frame,
                    f"pict_type: {header['pict_type'][i]}", (width-210, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        out.write(frame)

    cap.release()
    out.release()

def vis_composed(movie, header, flow, bboxes, base=False, worst=False):
    if base and worst:
        raise "rendering mode could not be duplicated"

    if base:
        cap, out = open_video(movie, postfix="base")
    elif worst:
        cap, out = open_video(movie, postfix="worst")
    else:
        cap, out = open_video(movie)

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    kalman = KalmanInterpolator()
    interp_kalman_clos = lambda bboxes, flow, frame: \
            interp_kalman(bboxes, flow, frame, kalman)

    color = (255, 0, 255)

    pos = 0
    for i in trange(count):
        ret, frame = cap.read()
        if ret is False or i > bboxes.size:
            break

        # linear
        if base:
            pos = i
            bboxes_0 = bboxes[pos].copy()
            bboxes_1 = bboxes[pos].copy()

            frame = draw_i_frame(frame, flow[i], bboxes_0)
            frame = draw_i_frame(frame, flow[i], bboxes_1, color=color)
        elif header["pict_type"][i] == "I":
            pos = i
            bboxes_0 = bboxes[pos].copy()
            bboxes_1 = bboxes[pos].copy()

            frame = draw_i_frame(frame, flow[i], bboxes_0)
            frame = draw_i_frame(frame, flow[i], bboxes_1, color=color)
            kalman.reset(bboxes[pos])
        elif worst:
            frame = draw_i_frame(frame, flow[i], bboxes_0)
            frame = draw_i_frame(frame, flow[i], bboxes_1, color=color)
        else:
            # bboxes_0 is updated by reference
            frame = draw_p_frame(frame, flow[i], bboxes_0)
            frame = draw_p_frame(frame, flow[i], bboxes_1, color=color,
                                 interp=interp_kalman_clos)

        cv2.rectangle(frame, (width-220, 20), (width-20, 60), (0, 0, 0), -1)
        cv2.putText(frame,
                    f"pict_type: {header['pict_type'][i]}", (width-210, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        out.write(frame)

    cap.release()
    out.release()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("movie")
    parser.add_argument("--base",
                        action="store_true", default=False,
                        help="render P-frame by true bbox")
    parser.add_argument("--worst",
                        action="store_true", default=False,
                        help="render P-frame by base I-frame's bbox")
    return parser.parse_args()

def main():
    args = parse_opt()

    flow, header = get_flow(args.movie)
    bboxes = pick_bbox(os.path.join(args.movie, "bbox_dump"))
    # vis_kalman(args.movie, header, flow, bboxes,
    #            base=args.base, worst=args.worst)
    vis_composed(args.movie, header, flow, bboxes,
                 base=args.base, worst=args.worst)

if __name__ == "__main__":
    main()
