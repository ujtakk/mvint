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
from interp import draw_i_frame, draw_p_frame, find_inner, interp_linear_unit

def calc_center(bbox):
    center_y = np.mean((bbox.bot, bbox.top))
    center_x = np.mean((bbox.right, bbox.left))
    center = np.asarray((center_x, center_y))

    return center

# delegated class
class KalmanInterpolator:
    def __init__(self, alpha=2.0, processNoise=1e-3, measurementNoise=1e-3):
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

def calc_flow_mean_kalman(inner_flow, bbox, kalman):
    if inner_flow.shape[0] < 2:
        flow_mean = np.zeros(2, dtype=np.float32)
    else:
        flow_mean = np.mean(inner_flow, axis=0)

    center = calc_center(bbox)

    new_center = kalman.filter(flow_mean)

    frame_mean = new_center - center

    return frame_mean

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
        flow_mean = calc_flow_mean_kalman(inner_flow, bbox, kalman)
        bboxes.loc[bbox.Index] = interp_linear_unit(bbox, flow_mean, frame)

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
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[pos])
        elif header["pict_type"][i] == "I":
            pos = i
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[pos])
            kalman.reset(bboxes[pos])
        elif worst:
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
        if base:
            pos = i
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[pos])
        elif header["pict_type"][i] == "I":
            pos = i
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[pos])
        elif worst:
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[pos])
        else:
            # bboxes[pos] is updated by reference
            frame_drawed = draw_p_frame(frame, flow[i], bboxes[pos])

        # kalman_0
        if base:
            pos_0 = i
            frame_drawed_0 = draw_i_frame(frame_drawed, flow[i], bboxes_0[pos_0], color=color_0)
        elif header["pict_type"][i] == "I":
            pos_0 = i
            frame_drawed_0 = draw_i_frame(frame_drawed, flow[i], bboxes_0[pos_0], color=color_0)
            kalman_0.reset(bboxes[pos_0])
        elif worst:
            frame_drawed_0 = draw_i_frame(frame_drawed, flow[i], bboxes_0[pos_0], color=color_0)
        else:
            # bboxes[pos_0] is updated by reference
            frame_drawed_0 = draw_p_frame(frame_drawed, flow[i], bboxes_0[pos_0],
                                          interp=interp_kalman_clos_0, color=color_0)

        # kalman_1
        if base:
            pos_1 = i
            frame_drawed_1 = draw_i_frame(frame_drawed_0, flow[i], bboxes_1[pos_1], color=color_1)
        elif header["pict_type"][i] == "I":
            pos_1 = i
            frame_drawed_1 = draw_i_frame(frame_drawed_0, flow[i], bboxes_1[pos_1], color=color_1)
            kalman_1.reset(bboxes[pos_1])
        elif worst:
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
