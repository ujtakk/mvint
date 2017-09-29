#!/usr/bin/env python3

import os
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import cv2
from tqdm import trange

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

    return flow_map, index_rate

def interp_bbox(bbox, flow_map, index_rate):
    height = flow_map.shape[0]
    width = flow_map.shape[1]

    flow_mean = np.mean(flow_map[bbox.top:bbox.bot, bbox.left:bbox.right,
                                 :], axis=(0, 1))
    flow_mean = np.nan_to_num(flow_mean)
    frame_mean = index_rate * flow_mean
    frame_mean *= index_rate

    left  = bbox.left + frame_mean[1]
    top   = bbox.top + frame_mean[0]
    right = bbox.right + frame_mean[1]
    bot   = bbox.bot + frame_mean[0]

    left  = np.clip(left, 0, width-1).astype(np.int)
    top   = np.clip(top, 0, height-1).astype(np.int)
    right = np.clip(right, 0, width-1).astype(np.int)
    bot   = np.clip(bot, 0, height-1).astype(np.int)

    return pd.Series({"name": bbox.name, "prob": bbox.prob,
        "left": left, "top": top, "right": right, "bot": bot})

def interp_linear(bboxes, flow, frame):
    flow_map, index_rate = map_flow(flow, frame)

    for bbox in bboxes.itertuples():
        idx = bbox.Index
        bboxes.loc[idx] = interp_bbox(bbox, flow_map, index_rate)

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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pos = 0
    for i in trange(count):
        ret, frame = cap.read()
        if ret is False or i > bboxes.size:
            break

        if header["pict_type"][i] == "I":
            frame_drawed = draw_main(frame, flow[i], bboxes[i])
            pos = i
        else:
            # bboxes[pos] is updated by reference
            frame_drawed = draw_sub(frame, flow[i], bboxes[pos])
        cv2.putText(frame,
                    f"pict_type: {header['pict_type'][i]}", (10, height-10),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (200, 200, 200), 2)
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
