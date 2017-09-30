#!/usr/bin/env python3

import os
from os.path import join, exists, basename
import re
import glob
import argparse

import cv2
import numpy as np
import pandas as pd
import tqdm

from flow import dump_flow, pick_flow, draw_flow

def frame2int(frame):
    return int(re.match(r"frame(\d+).dat", frame).group(1))

def pick_bbox(dir_path):
    A = []
    for frame in sorted(os.listdir(dir_path), key=frame2int):
        frame_path = os.path.join(dir_path, frame)
        try:
            B = pd.read_csv(frame_path)
        except:
            B = pd.DataFrame()
        A.append(B)

    return pd.Series(A)

def draw_bbox(frame, bbox, frame_mean=None):
    cv2.rectangle(frame, (bbox.left, bbox.top), (bbox.right, bbox.bot),
                  (0, 255, 0), 2)
    cv2.putText(frame, f"{bbox.name}: {bbox.prob}",
                (bbox.left, bbox.top-10),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    if frame_mean is not None:
        cv2.putText(frame, f"{frame_mean}",
                    (bbox.left, bbox.bot+20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    return frame

def draw_bboxes(frame, bboxes, frame_means=None):
    if bboxes.size == 0:
        return frame

    for bbox in bboxes.itertuples():
        if frame_means is None:
            frame = draw_bbox(frame, bbox)
        else:
            frame = draw_bbox(frame, bbox, frame_means[bbox.Index])

    return frame

def draw_annotate(frame, index, pos, flow, bboxes):
    frame = draw_flow(frame, flow[index])
    if index < bboxes.size:
        return draw_bboxes(frame, bboxes[pos])
    else:
        return draw_none(frame, index)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("movie")
    return parser.parse_args()

def main():
    args = parse_opt()
    dir_path = os.path.join(args.movie, "bbox_dump")
    bboxes = pick_bbox(dir_path)

    dump_flow(args.movie)
    flow, header = pick_flow(args.movie)

    draw_annotate_func = lambda frame, index, pos: \
        draw_annotate(frame, index, pos, flow, bboxes)
    vis_index_pos(args.movie, header, draw=draw_annotate_func)

if __name__ == "__main__":
    main()
