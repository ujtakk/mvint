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
import pandas as pd
from tqdm import trange

import mot16
from flow import get_flow, draw_flow
from annotate import pick_bbox, draw_bboxes
from interp import interp_linear
from interp import draw_i_frame, draw_p_frame, map_flow
from vis import open_video
from affinity import mapping

class MOT16:
    def __init__(self, src_id, dst_dir="result"):
        if not exists(dst_dir):
            os.makedirs(dst_dir)

        self.dst_fd = open(join(dst_dir, f"{src_id}.txt"), "w")
        self.ids = dict()
        self.prev_bboxes = pd.DataFrame()

    # def __del__(self):
    #     self.dst_fd.close()

    def bbox_id(self, bbox):
        return self.ids[f"{bbox.name}{bbox.prob:>1.6f}"]

    def eval_frame(self, fnum, bboxes, do_mapping=False):
        # if do_mapping:
        self.ids = mapping(bboxes, self.prev_bboxes)

        for bbox in bboxes.itertuples():
            obj_id = self.bbox_id(bbox)

            left = bbox.left
            top = bbox.top
            width = bbox.right - bbox.left
            height = bbox.bot - bbox.top

            print(f"{fnum},{obj_id},{left},{top},{width},{height},-1,-1,-1,-1",
                  file=self.dst_fd)

        prev_bboxes = bboxes

def eval_mot16(src_id, prefix="MOT16/train"):
    mot = MOT16(src_id)

    movie = join(prefix, src_id)
    bboxes = mot16.pick_mot16_bboxes(movie)
    flow, header = get_flow(movie, prefix=".")

    cap, out = open_video(movie)

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pos = 0
    for i in trange(count):
        ret, frame = cap.read()
        if ret is False or i > bboxes.size:
            break

        if header["pict_type"][i] == "I":
            pos = i
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[pos])
            mot.eval_frame(i, bboxes[pos], do_mapping=True)
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

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_id")
    return parser.parse_args()

def main():
    args = parse_opt()
    eval_mot16(args.src_id)

if __name__ == "__main__":
    main()
