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
from interp import draw_i_frame, draw_p_frame, map_flow

def calc_center(bbox):
    center_y = np.mean((bbox.bot, bbox.top))
    center_x = np.mean((bbox.right, bbox.left))
    center = np.asarray((center_x, center_y))

    return center

# delegated class
class KalmanInterpolator:
    def __init__(self, alpha=512, processNoise=1e-3, measurementNoise=1e-3):
        self.kalman = cv2.KalmanFilter(2, 2, 2)
        self.kalman.transitionMatrix = np.float32(1.0 * np.eye(2))
        self.kalman.controlMatrix = np.float32(alpha * np.eye(2))
        self.kalman.measurementMatrix = np.float32(1.0 * np.eye(2))
        self.kalman.processNoiseCov = np.float32(processNoise * np.eye(2))
        self.kalman.measurementNoiseCov = np.float32(measurementNoise * np.eye(2))

        self.total = 0
        self.count = 0
        self.stateList = []
        self.errorCovList = []

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
                    calc_center(bbox).reshape(2, 1).astype(np.float32))

        for i in range(self.total):
            self.errorCovList.append(
                    (1.0 * np.eye(2)).astype(np.float32))
                    # (0.0 * np.eye(2)).astype(np.float32))

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

    def filter(self, flow_mean):
        # print(f"total: {self.total}, count: {self.count}")
        state = self.predict(flow_mean)
        noise = np.random.randn(2, 1)

        new_center = np.dot(self.kalman.measurementMatrix, state) \
                   + np.dot(self.kalman.measurementNoiseCov, noise)
        new_center = new_center.astype(np.float32)

        self.correct(new_center)

        if self.count == self.total-1:
            self.count = 0
        else:
            self.count += 1

        return state.flatten()

def interp_kalman_unit(bbox, flow_map, index_rate, kalman):
    flow_mean = np.mean(flow_map[bbox.top:bbox.bot, bbox.left:bbox.right,
                                 :], axis=(0, 1))
    flow_mean = np.nan_to_num(flow_mean)

    alpha = 2.0 * index_rate ** 2
    center = calc_center(bbox)

    new_center = kalman.filter(flow_mean)

    frame_mean = new_center - center

    left  = bbox.left + frame_mean[0]
    top   = bbox.top + frame_mean[1]
    right = bbox.right + frame_mean[0]
    bot   = bbox.bot + frame_mean[1]

    height = flow_map.shape[0]
    width = flow_map.shape[1]

    left  = np.clip(left, 0, width-1).astype(np.int)
    top   = np.clip(top, 0, height-1).astype(np.int)
    right = np.clip(right, 0, width-1).astype(np.int)
    bot   = np.clip(bot, 0, height-1).astype(np.int)

    return pd.Series({"name": bbox.name, "prob": bbox.prob,
        "left": left, "top": top, "right": right, "bot": bot}), frame_mean

def interp_kalman(bboxes, flow, frame, kalman):
    flow_map, index_rate = map_flow(flow, frame)

    frame_means = []
    for bbox in bboxes.itertuples():
        idx = bbox.Index
        bboxes.loc[idx], frame_mean = \
            interp_kalman_unit(bbox, flow_map, index_rate, kalman)
        frame_means.append(frame_mean)

    return bboxes, frame_means

def vis_kalman(movie, header, flow, bboxes, full=False, base=False):
    if full and base:
        raise "rendering mode could not be duplicated"

    if full:
        cap, out = open_video(movie, postfix="full")
    elif base:
        cap, out = open_video(movie, postfix="base")
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

        if header["pict_type"][i] == "I":
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[i])
            pos = i
            kalman.reset(bboxes[pos])
        elif full:
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[i])
        elif base:
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[pos])
        else:
            # bboxes[pos] is updated by reference
            frame_drawed = draw_p_frame(frame, flow[i], bboxes[pos],
                                        interp=interp_kalman_clos)

        cv2.rectangle(frame, (width-220, 20), (width-20, 60), (0, 0, 0), -1)
        cv2.putText(frame,
                    f"pict_type: {header['pict_type'][i]}", (width-210, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        out.write(frame_drawed)

    cap.release()
    out.release()

def vis_composed(movie, header, flow, bboxes, full=False, base=False):
    if full and base:
        raise "rendering mode could not be duplicated"

    if full:
        cap, out = open_video(movie, postfix="full")
    elif base:
        cap, out = open_video(movie, postfix="base")
    else:
        cap, out = open_video(movie)

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    kalman_0 = KalmanInterpolator(processNoise=1e0, measurementNoise=1e0)
    interp_kalman_clos_0 = lambda bboxes, flow, frame: \
            interp_kalman(bboxes, flow, frame, kalman_0)

    kalman_1 = KalmanInterpolator(processNoise=1e1, measurementNoise=1e1)
    interp_kalman_clos_1 = lambda bboxes, flow, frame: \
            interp_kalman(bboxes, flow, frame, kalman_1)

    pos = 0
    pos_0 = 0
    pos_1 = 0
    bboxes_0 = bboxes.copy()
    bboxes_1 = bboxes.copy()
    color_0 = (0, 255, 255)
    color_1 = (255, 255, 0)
    for i in trange(count):
        ret, frame = cap.read()
        if ret is False or i > bboxes.size:
            break

        # linear
        if header["pict_type"][i] == "I":
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[i])
            pos = i
        elif full:
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[i])
        elif base:
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[pos])
        else:
            # bboxes[pos] is updated by reference
            frame_drawed = draw_p_frame(frame, flow[i], bboxes[pos])

        # kalman_0
        if header["pict_type"][i] == "I":
            frame_drawed_0 = draw_i_frame(frame_drawed, flow[i], bboxes_0[i], color=color_0)
            pos_0 = i
            kalman_0.reset(bboxes[pos_0])
        elif full:
            frame_drawed_0 = draw_i_frame(frame_drawed, flow[i], bboxes_0[i], color=color_0)
        elif base:
            frame_drawed_0 = draw_i_frame(frame_drawed, flow[i], bboxes_0[pos_0], color=color_0)
        else:
            # bboxes[pos_0] is updated by reference
            frame_drawed_0 = draw_p_frame(frame_drawed, flow[i], bboxes_0[pos_0],
                                        interp=interp_kalman_clos_0, color=color_0)

        # kalman_1
        if header["pict_type"][i] == "I":
            frame_drawed_1 = draw_i_frame(frame_drawed_0, flow[i], bboxes_1[i], color=color_1)
            pos_1 = i
            kalman_1.reset(bboxes[pos_1])
        elif full:
            frame_drawed_1 = draw_i_frame(frame_drawed_0, flow[i], bboxes_1[i], color=color_1)
        elif base:
            frame_drawed_1 = draw_i_frame(frame_drawed_0, flow[i], bboxes_1[pos_1], color=color_1)
        else:
            # bboxes[pos_1] is updated by reference
            frame_drawed_1 = draw_p_frame(frame_drawed_0, flow[i], bboxes_1[pos_1],
                                        interp=interp_kalman_clos_1, color=color_1)

        cv2.rectangle(frame, (width-220, 20), (width-20, 60), (0, 0, 0), -1)
        cv2.putText(frame,
                    f"pict_type: {header['pict_type'][i]}", (width-210, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        out.write(frame_drawed_1)

    cap.release()
    out.release()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("movie")
    parser.add_argument("--full", "-f",
                        action="store_true", default=False,
                        help="render P-frame by true bbox")
    parser.add_argument("--base", "-b",
                        action="store_true", default=False,
                        help="render P-frame by base I-frame's bbox")
    return parser.parse_args()

def main():
    args = parse_opt()

    flow, header = get_flow(args.movie)
    bboxes = pick_bbox(os.path.join(args.movie, "bbox_dump"))
    # vis_kalman(args.movie, header, flow, bboxes,
    #            full=args.full, base=args.base)
    vis_composed(args.movie, header, flow, bboxes,
               full=args.full, base=args.base)

if __name__ == "__main__":
    main()
