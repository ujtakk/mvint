#!/usr/bin/env python3

import os
import argparse
from multiprocessing import Pool

import numpy as np
import scipy as sp
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

    if np.sum(mask) == 0:
        return np.zeros((1, 1, 2)).astype(np.float32)

    inner_flow = flow[flow_index[mask, 0], flow_index[mask, 1], :]

    mask_shape = flow_index[mask, :][-1] - flow_index[mask, :][0] + 1
    inner_flow = inner_flow.reshape((mask_shape[0], mask_shape[1], -1))

    # if np.sum(mask) < 2:
    #     return inner_flow.astype(np.float32)
    #
    # mask_shape = flow_index[mask, :][-1] - flow_index[mask, :][0] + 1
    # weight = np.prod(mask_shape/2.75) \
    #        * sp.stats.norm.pdf(flow_index[mask, :],
    #                            loc=flow_index[mask, :][0] + mask_shape/2,
    #                            scale=mask_shape/8)
    #
    # inner_flow *= weight

    return inner_flow.astype(np.float32)

def calc_flow_mean(inner_flow, filling_rate=1.0):
    # if inner_flow.shape[0] < 2 or inner_flow.shape[1] < 2:
    #     return np.zeros(2)

    # ********************* Your MOT16 Results *********************
    # IDF1  IDP  IDR| Rcll  Prcn   FAR|   GT  MT   PT   ML|    FP    FN   IDs    FM|  MOTA  MOTP MOTAL
    # 54.6 64.4 47.4| 62.3  84.6  2.35|  517 110  300  107| 12484 41597   803  1502|  50.3  77.9  51.0
    flow_mean = np.mean(inner_flow, axis=(0, 1))
    # TODO: divide each corner
    flow_mean *= 1.0 / filling_rate

    return flow_mean

def median_mean(inner_flow):
    flow = inner_flow.reshape((-1, 2))
    flow_norm = np.linalg.norm(flow, axis=1)
    norm_index = np.argsort(flow_norm)
    lower_median = norm_index[norm_index.shape[0]//4*1]
    upper_median = norm_index[norm_index.shape[0]//4*3]
    flow_lut = inner_flow[inner_flow.shape[0]//2-1:inner_flow.shape[0]//2+2,
                          inner_flow.shape[1]//2-1:inner_flow.shape[1]//2+2,
                          :].reshape((-1, 2))
    judges = np.linalg.norm(flow_lut-flow[lower_median], axis=1) \
           < np.linalg.norm(flow_lut-flow[upper_median], axis=1)

    if np.sum(judges) < 5:
        frame_mean = flow[upper_median, :]
    else:
        frame_mean = flow[lower_median, :]

    return frame_mean.astype(np.float32)

def calc_flow_mean_adhoc(inner_flow):
    flow_mean = np.mean(inner_flow, axis=(0, 1))
    # flow_mean = median_mean(inner_flow)

    flow_mean *= 1 + np.nan_to_num(1.0 / np.nan_to_num(np.var(inner_flow, axis=(0, 1))))
    # flow_mean *= np.nan_to_num(np.log2(np.std(inner_flow, axis=(0, 1))))
    # flow_mean *= 1 + 1.0 / np.nanmean(np.std(inner_flow, axis=(0, 1)))

    return np.nan_to_num(flow_mean)

def calc_flow_mean_mixture(inner_flow):
    dist = sklearn.mixture.GaussianMixture(n_components=2)
    # dist = sklearn.mixture.BayesianGaussianMixture(n_components=2)
    dist.fit(inner_flow)
    weights = dist.weights_
    means = dist.means_
    index = np.argmax(weights)

    # ********************* Your MOT16 Results *********************
    # IDF1  IDP  IDR| Rcll  Prcn   FAR|   GT  MT   PT   ML|    FP    FN   IDs    FM|  MOTA  MOTP MOTAL
    # 54.9 64.8 47.6| 62.7  85.2  2.25|  517 118  291  108| 11977 41209   779  1425|  51.1  78.1  51.8
    flow_mean = means[index, :]

    # # ********************* Your MOT16 Results *********************
    # # IDF1  IDP  IDR| Rcll  Prcn   FAR|   GT  MT   PT   ML|    FP    FN   IDs    FM|  MOTA  MOTP MOTAL
    # # 53.0 62.5 46.0| 60.8  82.6  2.66|  517  85  315  117| 14159 43271   919  1773|  47.2  78.2  48.0
    # flow_mean = means[index, :] / weights[index]

    # # ********************* Your MOT16 Results *********************
    # # IDF1  IDP  IDR| Rcll  Prcn   FAR|   GT  MT   PT   ML|    FP    FN   IDs    FM|  MOTA  MOTP MOTAL
    # # 51.4 60.5 44.7| 59.8  81.1  2.90|  517  78  324  115| 15429 44398  1079  1917|  44.8  77.8  45.8
    # flow_mean = means[np.argmax(weights), :] / weights[np.argmax(weights)] \
    #           + means[np.argmin(weights), :] * weights[np.argmin(weights)]

    # # ********************* Your MOT16 Results *********************
    # # IDF1  IDP  IDR| Rcll  Prcn   FAR|   GT  MT   PT   ML|    FP    FN   IDs    FM|  MOTA  MOTP MOTAL
    # # 37.7 44.4 32.8| 51.9  70.2  4.59|  517  27  357  133| 24379 53099  1877  3734|  28.1  74.7  29.8
    # flow_mean = np.sum(means, axis=0)

    # NOTE: may bad
    # flow_cand = (means.T / weights).T
    # flow_mean = np.sum(flow_cand, axis=0)
    # flow_mean = flow_cand[np.argmax(np.linalg.norm(flow_cand, axis=1)), :]

    # NOTE: may bad
    # index = np.argmax(np.linalg.norm(means, axis=1))
    # flow_mean = means[index, :] / weights[index]

    return np.nan_to_num(flow_mean)

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

def interp_divide_unit(bbox, inner_flow, frame):
    if inner_flow.shape[0] < 2 or inner_flow.shape[1] < 2:
        left  = np.int(bbox.left)
        top   = np.int(bbox.top)
        right = np.int(bbox.right)
        bot   = np.int(bbox.bot)
        return pd.Series({"name": bbox.name, "prob": bbox.prob,
            "left": left, "top": top, "right": right, "bot": bot})

    center = np.asarray(inner_flow.shape) // 2
    upper_left = np.mean(inner_flow[:center[0], :center[1]], axis=(0, 1))
    upper_right = np.mean(inner_flow[:center[0], center[1]:], axis=(0, 1))
    lower_left = np.mean(inner_flow[center[0]:, :center[1]], axis=(0, 1))
    lower_right = np.mean(inner_flow[center[0]:, center[1]:], axis=(0, 1))

    a = 1.0

    left  = bbox.left + a* np.mean((upper_left, lower_left), axis=0)[1]
    top   = bbox.top + a* np.mean((upper_left, upper_right), axis=0)[0]
    right = bbox.right + a* np.mean((upper_right, lower_right), axis=0)[1]
    bot   = bbox.bot + a* np.mean((lower_left, lower_right), axis=0)[0]

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

        # bboxes.loc[bbox.Index] = interp_divide_unit(bbox, inner_flow, frame)

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
            frame = draw_i_frame(frame, flow[i], bboxes[pos])
        elif header["pict_type"][i] == "I":
            pos = i
            frame = draw_i_frame(frame, flow[i], bboxes[pos])
        elif worst:
            frame = draw_i_frame(frame, flow[i], bboxes[pos])
        else:
            # bboxes[pos] is updated by reference
            frame = draw_p_frame(frame, flow[i], bboxes[pos])

        cv2.rectangle(frame, (width-220, 20), (width-20, 60), (0, 0, 0), -1)
        cv2.putText(frame,
                    f"pict_type: {header['pict_type'][i]}", (width-210, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        out.write(frame)

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
