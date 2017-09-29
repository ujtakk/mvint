#!/usr/bin/env python3

import os
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import cv2
import tqdm

from flow import get_flow, draw_flow
from annotate import pick_bbox, draw_bboxes
from draw import draw_none
from vis import open_video

def map_flow(flow, frame):
    frame_rows = frame.shape[0]
    frame_cols = frame.shape[1]
    assert(frame.shape[2] == 3)

    rows = flow.shape[0]
    cols = flow.shape[1]
    assert(flow.shape[2] == 2)

    flow_map = np.zeros((frame_rows, frame_cols,  2))

    base_index = np.asarray(tuple(np.ndindex((rows, cols))))
    index_rate = np.asarray((frame_rows // rows, frame_cols // cols))
    flow_index = base_index * index_rate + (index_rate // 2)
    for i in range(flow.size//2):
        flow_map[flow_index[i][0], flow_index[i][1], :] = \
            flow[base_index[i][0], base_index[i][1], :]

    return flow_map

def interp_linear(bboxes, flow, frame):
    flow_map = map_flow(flow, frame)
    height = frame.shape[0]
    width = frame.shape[1]
    index_rate = np.asarray(frame.shape[0:2]) // np.asarray(flow.shape[0:2])
    def interp_bbox(bbox):
        flow_mean = np.mean(flow_map[bbox.top:bbox.bot,
                                     bbox.left:bbox.right,
                                     :], axis=(0, 1))
        flow_mean = 4 * index_rate * flow_mean

        left  = int(bbox.left + flow_mean[1])
        if left < 0:
            left = 0

        top   = int(bbox.top + flow_mean[0])
        if top < 0:
            top = 0

        right = int(bbox.right + flow_mean[1])
        if right > width-1:
            right = width-1

        bot   = int(bbox.bot + flow_mean[0])
        if bot > height-1:
            bot = height-1

        return pd.Series({"name": bbox.name, "prob": bbox.prob,
            "left": left, "top": top, "right": right, "bot": bot})

    for bbox in bboxes.itertuples():
        idx = bbox.Index
        bboxes.loc[idx] = interp_bbox(bbox)

    return bboxes

def interp_none(bboxes, flow, frame):
    return bboxes

def draw_i_frame(frame, flow, bboxes):
    frame = draw_flow(frame, flow)
    frame = draw_bboxes(frame, bboxes)
    return frame

def draw_p_frame(frame, flow, base_bboxes, interp=interp_linear):
    frame = draw_flow(frame, flow)
    interp_bboxes = interp(base_bboxes, flow, frame)
    frame = draw_bboxes(frame, interp_bboxes)
    return frame

def vis_interp(movie, header, flow, bboxes, draw_main, draw_sub):
    cap, out = open_video(movie)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pos = 0
    for i in tqdm.trange(count):
        ret, frame = cap.read()
        if ret is False or i > bboxes.size:
            break

        cv2.putText(frame,
                    f"pict_type: {header['pict_type'][i]}", (10, height-10),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
        if header["pict_type"][i] == "I":
            frame_drawed = draw_main(frame, flow[i], bboxes[i])
            pos = i
        else:
            # bboxes[pos] is updated by reference
            frame_drawed = draw_sub(frame, flow[i], bboxes[pos])
        out.write(frame_drawed)

    cap.release()
    out.release()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("movie")
    return parser.parse_args()

def main():
    args = parse_opt()
    flow, header = get_flow(args.movie)
    bboxes = pick_bbox(os.path.join(args.movie, "bbox_dump"))
    vis_interp(args.movie, header, flow, bboxes, draw_i_frame, draw_p_frame)

if __name__ == "__main__":
    main()
