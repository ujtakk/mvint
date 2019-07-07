#!/usr/bin/env python3

"""Annotate extracted motion vectors and bounding boxes to the movie.
Input:
    movie: str
        Directory name that contains the mp4 movie (encoded one)
        (Name of the movie have to be same as the directory name)
"""

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
from vis import open_video

def pick_bbox(dir_path):
    def frame2int(frame):
        return int(re.match(r"frame(\d+).dat", frame).group(1))

    A = []
    for frame in sorted(os.listdir(dir_path), key=frame2int):
        frame_path = os.path.join(dir_path, frame)
        try:
            B = pd.read_csv(frame_path)
        except:
            B = pd.DataFrame()
        A.append(B)

    return pd.Series(A)

def draw_bbox(frame, bbox, frame_mean=None, color=(0, 255, 0), caption=True):
    cv2.rectangle(frame, (bbox.left, bbox.top), (bbox.right, bbox.bot),
                  color, 4)

    caption=False
    if caption:
        cv2.putText(frame, f"{bbox.name}: {bbox.prob}",
                    (bbox.left, bbox.top-10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        if frame_mean is not None:
            cv2.putText(frame, f"{bbox.velo}",
                        (bbox.left, bbox.bot+20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    return frame

def draw_bboxes(frame, bboxes, frame_means=None, color=(0, 255, 0)):
    if bboxes.size == 0:
        return frame

    for bbox in bboxes.itertuples():
        if frame_means is None:
            frame = draw_bbox(frame, bbox, color=color)
        else:
            frame = draw_bbox(frame, bbox,
                              frame_mean=frame_means[bbox.Index], color=color)

    return frame

def draw_annotate(frame, index, pos, flow, bboxes):
    frame = draw_flow(frame, flow[index])
    if index < bboxes.size:
        return draw_bboxes(frame, bboxes[pos])
    else:
        return draw_none(frame, index)

def vis_annotate(movie, header, draw):
    cap, out = open_video(movie, postfix='annotate')
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pos = 0
    for i in tqdm.trange(count):
        ret, frame = cap.read()
        if ret is False:
            break

        cv2.putText(frame,
                    f"pict_type: {header['pict_type'][i]}", (10, height-10),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
        if header["pict_type"][i] == "I":
            pos = i
        frame = draw(frame, i, pos)
        out.write(frame)

    cap.release()
    out.release()

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
    vis_annotate(args.movie, header, draw=draw_annotate_func)

if __name__ == "__main__":
    main()
