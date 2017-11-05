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

# delegated class
class KalmanInterpolator:
    # def __init__(self, dp=8, mp=4, cp=2, processNoise=1e-0, measurementNoise=1e-1):
    def __init__(self, dp=2, mp=2, cp=2, processNoise=1e-0, measurementNoise=1e-1):
        self.kalman = cv2.KalmanFilter(dp, mp, cp)
        self.kalman.transitionMatrix = np.float32(1.0 * np.eye(dp, dp))
        for i in range(dp-mp):
            self.kalman.transitionMatrix[i, mp + i] = np.float32(1.0)
        self.kalman.controlMatrix = np.float32(1.0 * np.eye(dp, cp))
        self.kalman.measurementMatrix = np.float32(1.0 * np.eye(mp, dp))
        self.kalman.processNoiseCov = np.float32(processNoise * np.eye(dp, dp))
        self.kalman.measurementNoiseCov = np.float32(measurementNoise * np.eye(mp, mp))

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
                # np.append(convert(bbox), (1., 1., 1., 1.)).astype(np.float32))

        for i in range(self.total):
            self.errorCovList.append(
                    (1.0 * np.eye(self.dp, self.dp)).astype(np.float32))

    def predict(self, control):
        self.kalman.statePost = self.stateList[self.count]
        self.kalman.errorCovPost = self.errorCovList[self.count]
        result = self.kalman.predict(control)
        return result

    def correct(self, measurement):
        result = self.kalman.correct(measurement)
        self.stateList[self.count] = self.kalman.statePost
        self.errorCovList[self.count] = self.kalman.errorCovPost
        return result

    def filter(self, center, flow_mean):
    # def filter(self, bbox, flow_mean):
        flow_mean = flow_mean.astype(np.float32)
        state = self.predict(flow_mean)
        noise = np.random.randn(self.mp, 1)

        # center = center.reshape(self.mp, 1)
        new_center = np.dot(self.kalman.measurementMatrix, state) \
                   + np.dot(self.kalman.measurementNoiseCov, noise)
        new_center = new_center.astype(np.float32)
        # new_center = center.astype(np.float32)

        self.correct(new_center)
        # self.correct(convert(bbox))

        if self.count == self.total-1:
            self.count = 0
        else:
            self.count += 1

        return state[0:self.mp].flatten()

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

    # new_bbox = kalman.filter(bbox, flow_mean)
    # new_bbox = invert(new_bbox)
    #
    # left  = new_bbox.left
    # top   = new_bbox.top
    # right = new_bbox.right
    # bot   = new_bbox.bot

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

        flow_mean = calc_flow_mean(inner_flow)
        bboxes.loc[bbox.Index] = interp_kalman_unit(bbox, flow_mean, frame,
                                                    kalman)

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

    kalman = KalmanInterpolator(processNoise=1e-0, measurementNoise=1e-1)
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
    vis_kalman(args.movie, header, flow, bboxes,
               base=args.base, worst=args.worst)
    # vis_composed(args.movie, header, flow, bboxes,
    #              base=args.base, worst=args.worst)

if __name__ == "__main__":
    main()
