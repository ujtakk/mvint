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

import cv2
import numpy as np
import pandas as pd
from tqdm import trange

from mot16 import pick_mot16_bboxes
from flow import get_flow, draw_flow
from annotate import pick_bbox, draw_bboxes
from interp import interp_linear
from interp import draw_i_frame, draw_p_frame, map_flow
from vis import open_video
from mapping import SimpleMapper
from bbox_ssd import predict, setup_model

class MOT16:
    def __init__(self, src_id, dst_dir="result", mapper=None):
        if not exists(dst_dir):
            os.makedirs(dst_dir)

        self.dst_fd = open(join(dst_dir, f"{src_id}.txt"), "w")

        self.prev_bboxes = pd.DataFrame()
        if mapper is None:
            # self.mapper = SimpleMapper(cost_thresh=cost_thresh, log_id=src_id)
            self.mapper = SimpleMapper(log_id=src_id)
        else:
            self.mapper = mapper

    # def __del__(self):
    #     self.dst_fd.close()

    def eval_frame(self, fnum, bboxes, do_mapping=False):
        if do_mapping:
            self.mapper.set(bboxes, self.prev_bboxes)

        names = []
        for bbox in bboxes.itertuples():
            obj_id = self.mapper.get(bbox)
            names.append(str(obj_id))

            left = bbox.left
            top = bbox.top
            width = bbox.right - bbox.left
            height = bbox.bot - bbox.top

            print(f"{fnum},{obj_id},{left},{top},{width},{height},-1,-1,-1,-1",
                  file=self.dst_fd)
        bboxes["name"] = names

        self.prev_bboxes = bboxes

def eval_mot16(src_id, prefix="MOT16/train",
               thresh=0.0, baseline=False, worst=False, cost_thresh=40000):
    # mot = MOT16(src_id, mapper=SimpleMapper(cost_thresh=cost_thresh, log_id=src_id))
    mot = MOT16(src_id, mapper=SimpleMapper(log_id=src_id))

    movie = join(prefix, src_id)
    bboxes = pick_mot16_bboxes(movie)
    flow, header = get_flow(movie, prefix=".")

    cap, out = open_video(movie)

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for index, bbox in enumerate(bboxes):
        if not bbox.empty:
            bboxes[index] = bbox.query(f"prob > {thresh}")

    pos = 0
    for i in trange(count):
        ret, frame = cap.read()
        if ret is False or i > bboxes.size:
            break

        if baseline:
            pos = i
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[pos])
            mot.eval_frame(i, bboxes[pos], do_mapping=True)
        elif header["pict_type"][i] == "I":
            pos = i
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[pos])
            mot.eval_frame(i, bboxes[pos], do_mapping=True)
        elif worst:
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[pos])
            mot.eval_frame(i, bboxes[pos], do_mapping=False)
            # mot.eval_frame(i, bboxes[pos], do_mapping=True)
        else:
            # bboxes[pos] is updated by reference
            frame_drawed = draw_p_frame(frame, flow[i], bboxes[pos])
            mot.eval_frame(i, bboxes[pos], do_mapping=False)

        cv2.rectangle(frame, (width-220, 20), (width-20, 60), (0, 0, 0), -1)
        cv2.putText(frame,
                    f"pict_type: {header['pict_type'][i]}", (width-210, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        out.write(frame_drawed)

    cap.release()
    out.release()

def eval_mot16_pred(args, prefix="MOT16/train",
                    thresh=0.0, baseline=False, worst=False, cost_thresh=40000):
    mot = MOT16(args.src_id, cost_thresh=cost_thresh)

    movie = join(prefix, args.src_id)
    model = setup_model(args)
    flow, header = get_flow(movie, prefix=".")

    cap, out = open_video(movie)

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bboxes = pd.DataFrame()
    for i in trange(count):
        ret, frame = cap.read()
        if ret is False:
            break

        if baseline:
            bboxes = predict(model, frame, thresh=thresh)
            frame_drawed = draw_i_frame(frame, flow[i], bboxes)
            mot.eval_frame(i, bboxes, do_mapping=True)
        elif header["pict_type"][i] == "I":
            bboxes = predict(model, frame, thresh=thresh)
            frame_drawed = draw_i_frame(frame, flow[i], bboxes)
            mot.eval_frame(i, bboxes, do_mapping=True)
        elif worst:
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[pos])
            mot.eval_frame(i, bboxes[pos], do_mapping=False)
            # mot.eval_frame(i, bboxes[pos], do_mapping=True)
        else:
            # bboxes[pos] is updated by reference
            frame_drawed = draw_p_frame(frame, flow[i], bboxes)
            mot.eval_frame(i, bboxes, do_mapping=False)

        cv2.rectangle(frame, (width-220, 20), (width-20, 60), (0, 0, 0), -1)
        cv2.putText(frame,
                    f"pict_type: {header['pict_type'][i]}", (width-210, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        out.write(frame_drawed)

    cap.release()
    out.release()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_id")
    parser.add_argument("--baseline",
                        action="store_true", default=False)
    parser.add_argument("--worst",
                        action="store_true", default=False)
    parser.add_argument("--thresh", type=float, default=0.0)
    parser.add_argument("--cost", type=float, default=40000)
    parser.add_argument("--model",
                        choices=("ssd300", "ssd512"), default="ssd512")
    parser.add_argument("--param",
                        default="/home/work/takau/6.image/mot/mot16_ssd512.h5")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_opt()
    eval_mot16(args.src_id,
               thresh=args.thresh,
               baseline=args.baseline,
               worst=args.worst,
               # cost_thresh=np.inf)
               cost_thresh=args.cost)
    # eval_mot16_pred(args,
    #                 thresh=args.thresh,
    #                 baseline=args.baseline,
    #                 worst=args.worst,
    #                 cost_thresh=args.cost)

if __name__ == "__main__":
    main()
