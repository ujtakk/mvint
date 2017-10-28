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

    flow_map = np.zeros((frame_rows, frame_cols,  2), dtype=np.float32)

    base_index = np.asarray(tuple(np.ndindex((rows, cols))))
    index_rate = np.asarray((frame_rows // rows, frame_cols // cols))
    flow_index = base_index * index_rate + (index_rate // 2)
    for i in range(flow.size//2):
        flow_map[flow_index[i][0], flow_index[i][1], :] = \
            flow[base_index[i][0], base_index[i][1], :]

    return flow_map, index_rate

# def interp_linear_unit(bbox, flow_map, index_rate, filling_rate=0.5):
def interp_linear_unit(bbox, flow_map, index_rate, filling_rate=0.85):
    inner_flow = flow_map[bbox.top:bbox.bot, bbox.left:bbox.right, :]
    if inner_flow.size == 0:
        flow_mean = np.nan
    else:
        flow_mean = np.nanmean(inner_flow, axis=(0, 1))
    flow_mean = np.nan_to_num(flow_mean)

    # macroblock_mean / n_pixels -> macroblock_mean / n_macroblocks
    # (n_macroblocks ~ n_pixels / index_rate ** 2)
    frame_mean = index_rate**2 * flow_mean
    # TODO: divide each corner
    frame_mean *= 1.0 / filling_rate

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

def interp_linear(bboxes, flow, frame):
    flow_map, index_rate = map_flow(flow, frame)

    frame_means = []
    for bbox in bboxes.itertuples():
        idx = bbox.Index
        bboxes.loc[idx], frame_mean = \
            interp_linear_unit(bbox, flow_map, index_rate)
        frame_means.append(frame_mean)

    return bboxes, frame_means

def interp_none(bboxes, flow, frame):
    return bboxes

def draw_i_frame(frame, flow, bboxes, color=(0, 255, 0)):
    frame = draw_flow(frame, flow)
    frame = draw_bboxes(frame, bboxes, color=color)
    return frame

def draw_p_frame(frame, flow, base_bboxes, interp=interp_linear, color=(0, 255, 0)):
    frame = draw_flow(frame, flow)
    interp_bboxes, frame_means = interp(base_bboxes, flow, frame)
    frame = draw_bboxes(frame, interp_bboxes, frame_means, color=color)
    return frame

def vis_interp(movie, header, flow, bboxes, full=False, base=False):
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

    pos = 0
    for i in trange(count):
        ret, frame = cap.read()
        if ret is False or i > bboxes.size:
            break

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

        # cv2.rectangle(frame, (width-220, 20), (width-20, 60), (0, 0, 0), -1)
        # cv2.putText(frame,
        #             f"pict_type: {header['pict_type'][i]}", (width-210, 50),
        #             cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        out.write(frame_drawed)

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
    # for index, bbox in enumerate(bboxes):
    #     if not bbox.empty:
    #         bboxes[index] = bbox.query(f"prob >= 0.4")
    vis_interp(args.movie, header, flow, bboxes,
               full=args.full, base=args.base)

if __name__ == "__main__":
    main()
