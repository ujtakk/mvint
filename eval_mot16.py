#!/usr/bin/env python3

"""Evaluation Procedure for MOT Dataset.

Requisite Inputs:
    - Frame Number
    - Object ID
    - Bounding Box

Frame Number is given as Advance Input.
Bounding box is predicted using arbitrary algorithm.

Key point is the method to determine Object ID.
The naive method is to use Hungarian algorithm
with appropriate affinity metric.
"""

import os
from os.path import join, exists
import argparse
from collections import defaultdict
import time

import cv2
import numpy as np
import pandas as pd
from tqdm import trange

from mot16 import detinfo
from flow import get_flow, draw_flow
from annotate import pick_bbox, draw_bboxes
from interp import interp_linear
from interp import draw_i_frame, draw_p_frame
from kalman import KalmanInterpolator, interp_kalman
from vis import open_video
from mapping import SimpleMapper
from bbox_ssd import predict, setup_model

class MOT16:
    def __init__(self, src_id, src_dir="MOT16/train", dst_dir="result"):
        if not exists(dst_dir):
            os.makedirs(dst_dir)

        self.source = detinfo(join(src_dir, src_id), poi=False)
        # self.source = detinfo(join(src_dir, src_id), poi=True)
        self.target = open(join(dst_dir, f"{src_id}.txt"), "w")

        self.frame_count = 1

        self.prev_bboxes = pd.DataFrame()
        self.mapper = SimpleMapper()

    def pick_bboxes(self):
        det_frames = self.source["frame"].unique()
        bboxes = [pd.DataFrame() for _ in np.arange(np.max(det_frames))]

        for frame in det_frames:
            det_entry = self.source.query(f"frame == {frame}").reset_index()
            left = (det_entry["left"]).astype(np.int)
            top = (det_entry["top"]).astype(np.int)
            right = (det_entry["left"] + det_entry["width"]).astype(np.int)
            bot = (det_entry["top"] + det_entry["height"]).astype(np.int)
            bboxes[frame-1] = pd.DataFrame({
                "name": "", "prob": det_entry["score"],
                "left": left, "top": top, "right": right, "bot": bot
            })

        return pd.Series(bboxes)

    def eval_frame(self, bboxes):
        self.mapper.set(bboxes)

        for bbox_id, bbox_body in self.mapper.get(bboxes):
            left = bbox_body.left
            top = bbox_body.top
            width = bbox_body.right - bbox_body.left
            height = bbox_body.bot - bbox_body.top

            print(f"{self.frame_count},{bbox_id},{left},{top},{width},{height}",
                  file=self.target)

        self.frame_count += 1

def eval_mot16(src_id, prefix="MOT16/train", MOT16=MOT16,
               thresh=0.0, baseline=False, worst=False, display=False,
               gop=12):
    mot = MOT16(src_id)
    bboxes = mot.pick_bboxes()

    movie = join(prefix, src_id)
    flow, header = get_flow(movie, prefix=".", gop=gop)

    if display:
        cap, out = open_video(movie, display=True)
    else:
        cap = open_video(movie, display=False)

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    kalman = KalmanInterpolator()
    interp_kalman_clos = lambda bboxes, flow, frame: \
            interp_kalman(bboxes, flow, frame, kalman)

    for index, bbox in enumerate(bboxes):
        if not bbox.empty:
            bboxes[index] = bbox.query(f"prob >= {thresh}")

    start = time.perf_counter()
    for i in range(count):
        ret, frame = cap.read()
        if ret is False or i > bboxes.size:
            break

        if baseline:
            bbox = bboxes[i].copy()
            if display:
                frame = draw_i_frame(frame, flow[i], bbox)
            mot.eval_frame(bbox)
            kalman.reset(bbox)
        elif header["pict_type"][i] == "I":
            bbox = bboxes[i].copy()
            if display:
                frame = draw_i_frame(frame, flow[i], bbox)
            mot.eval_frame(bbox)
            kalman.reset(bbox)
        elif worst:
            if display:
                frame = draw_i_frame(frame, flow[i], bbox)
            mot.eval_frame(bbox)
        else:
            # bbox is updated by reference
            if display:
                frame = draw_p_frame(frame, flow[i], bbox,
                                     interp=interp_kalman_clos)
            else:
                interp_kalman_clos(bbox, flow[i], frame)
            mot.eval_frame(bbox)

        if display:
            cv2.rectangle(frame, (width-220, 20), (width-20, 60), (0, 0, 0), -1)
            cv2.putText(frame,
                        f"pict_type: {header['pict_type'][i]}", (width-210, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

            out.write(frame)
    end = time.perf_counter()
    print(f"{src_id}: {count/(end-start):3.1f} FPS")
    # print(f"{end-start:5.2f}")

    cap.release()
    if display:
        out.release()

def eval_mot16_pred(src_id, model, prefix="MOT16/train", MOT16=MOT16,
                    thresh=0.0, baseline=False, worst=False, display=False):
    mot = MOT16(src_id)

    movie = join(prefix, args.src_id)
    flow, header = get_flow(movie, prefix=".")

    if display:
        cap, out = open_video(movie, display=True)
    else:
        cap = open_video(movie, display=False)

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start = time.perf_counter()
    for i in trange(count):
        ret, frame = cap.read()
        if ret is False:
            break

        if baseline:
            bbox = predict(model, frame, thresh=thresh)
            if display:
                frame = draw_i_frame(frame, flow[i], bbox)
            mot.eval_frame(bbox)
        elif header["pict_type"][i] == "I":
            bbox = predict(model, frame, thresh=thresh)
            if display:
                frame = draw_i_frame(frame, flow[i], bbox)
            mot.eval_frame(bbox)
        elif worst:
            if display:
                frame = draw_i_frame(frame, flow[i], bbox)
            mot.eval_frame(bbox)
        else:
            # bbox is updated by reference
            if display:
                frame = draw_p_frame(frame, flow[i], bbox)
            else:
                interp_linear(bbox, flow[i], frame)
            mot.eval_frame(bbox)

        if display:
            cv2.rectangle(frame, (width-220, 20), (width-20, 60), (0, 0, 0), -1)
            cv2.putText(frame,
                        f"pict_type: {header['pict_type'][i]}", (width-210, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

            out.write(frame)
    end = time.perf_counter()
    print(f"{src_id}: {count/(end-start):3.1f} FPS")

    cap.release()
    if display:
        out.release()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_id")
    parser.add_argument("--thresh", type=float, default=0.1)
    parser.add_argument("--baseline",
                        action="store_true", default=False)
    parser.add_argument("--worst",
                        action="store_true", default=False)
    parser.add_argument("--param",
                        default="/home/work/takau/6.image/mot/mot16_ssd512.h5")
    parser.add_argument("--display",
                        action="store_true", default=False)
    parser.add_argument("--pred",
                        action="store_true", default=False)
    parser.add_argument("--model",
                        choices=("ssd300", "ssd512"), default="ssd512")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gop", type=int, default=12)
    return parser.parse_args()

def main():
    args = parse_opt()
    if args.pred:
        # DEPRECATED
        eval_mot16_pred(args.src_id,
                        setup_model(args.param, args.model, args.gpu),
                        thresh=args.thresh,
                        baseline=args.baseline,
                        worst=args.worst,
                        display=args.display)
    else:
        eval_mot16(args.src_id,
                   thresh=args.thresh,
                   baseline=args.baseline,
                   worst=args.worst,
                   display=args.display,
                   gop=args.gop)

if __name__ == "__main__":
    main()
