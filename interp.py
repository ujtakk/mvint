#!/usr/bin/env python3

import os
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import cv2
from tqdm import trange
import sklearn.mixture

from flow import get_flow, draw_flow
from annotate import pick_bbox, draw_bboxes
from draw import draw_none
from vis import open_video

def find_inner(flow, bbox, flow_index, frame_index):
    mask_y = (bbox.top <= frame_index[:, 0]) \
           * (frame_index[:, 0] < bbox.bot)
    mask_x = (bbox.left <= frame_index[:, 1]) \
           * (frame_index[:, 1] < bbox.right)
    mask = mask_y * mask_x

    inner_flow = flow[flow_index[mask, 0], flow_index[mask, 1], :]

    return inner_flow.astype(np.float32)

def calc_flow_mean(inner_flow, filling_rate=0.85):
    if inner_flow.shape[0] < 2:
        flow_mean = np.zeros(2)
    else:
        # flow_mean = np.mean(inner_flow, axis=0)

        dist = sklearn.mixture.GaussianMixture(n_components=2)
        dist.fit(inner_flow)
        weights = dist.weights_
        means = dist.means_

        # flow_cand = (means.T / weights).T
        # flow_mean = flow_cand[np.argmax(np.linalg.norm(flow_cand, axis=1)), :]

        # filling_rate = 1.0
        # flow_mean = np.sum(means, axis=0)

        # index = np.argmin(np.linalg.norm(means, axis=1))
        # filling_rate = weights[index]
        # flow_mean = means[index, :]

        index = np.argmax(weights)
        filling_rate = weights[index]
        flow_mean = means[index, :]

        # TODO: divide each corner
        flow_mean *= 1.0 / filling_rate

    return flow_mean

# def interp_linear_unit(bbox, flow_map, index_rate, filling_rate=0.5):
def interp_linear_unit(bbox, flow_mean, frame):
    left  = bbox.left + flow_mean[0]
    top   = bbox.top + flow_mean[1]
    right = bbox.right + flow_mean[0]
    bot   = bbox.bot + flow_mean[1]

    height = frame.shape[0]
    width = frame.shape[1]

    left  = np.clip(left, 0, width-1).astype(np.int)
    top   = np.clip(top, 0, height-1).astype(np.int)
    right = np.clip(right, 0, width-1).astype(np.int)
    bot   = np.clip(bot, 0, height-1).astype(np.int)

    return pd.Series({"name": bbox.name, "prob": bbox.prob,
        "left": left, "top": top, "right": right, "bot": bot})

def interp_linear(bboxes, flow, frame):
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
        flow_mean = calc_flow_mean(inner_flow)
        bboxes.loc[bbox.Index] = interp_linear_unit(bbox, flow_mean, frame)

    return bboxes

def interp_none(bboxes, flow, frame):
    return bboxes

def draw_i_frame(frame, flow, bboxes, color=(0, 255, 0)):
    frame = draw_flow(frame, flow)
    frame = draw_bboxes(frame, bboxes, color=color)
    return frame

def draw_p_frame(frame, flow, base_bboxes, interp=interp_linear, color=(0, 255, 0)):
    frame = draw_flow(frame, flow)
    interp_bboxes = interp(base_bboxes, flow, frame)
    frame = draw_bboxes(frame, interp_bboxes, color=color)
    return frame

def vis_interp(movie, header, flow, bboxes, base=False, worst=False):
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
        elif worst:
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[pos])
        else:
            # bboxes[pos] is updated by reference
            frame_drawed = draw_p_frame(frame, flow[i], bboxes[pos])

        cv2.rectangle(frame, (width-220, 20), (width-20, 60), (0, 0, 0), -1)
        cv2.putText(frame,
                    f"pict_type: {header['pict_type'][i]}", (width-210, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        out.write(frame_drawed)

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
    # for index, bbox in enumerate(bboxes):
    #     if not bbox.empty:
    #         bboxes[index] = bbox.query(f"prob >= 0.4")
    vis_interp(args.movie, header, flow, bboxes,
               base=args.base, worst=args.worst)

if __name__ == "__main__":
    main()
