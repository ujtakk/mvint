#!/usr/bin/env python3

import os
import time
import pickle
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import sklearn.mixture
from tqdm import trange

from flow import get_flow, draw_flow
from annotate import pick_bbox, draw_bboxes
from draw import draw_none
from vis import open_video

def plot(flow, frame, flow_index, frame_index):
    def plot_bbox(graph, bbox):
        mask_y = (bbox.top <= frame_index[:, 0]) \
               * (frame_index[:, 0] < bbox.bot)
        mask_x = (bbox.left <= frame_index[:, 1]) \
               * (frame_index[:, 1] < bbox.right)
        mask = mask_y * mask_x
        if np.sum(mask) < 2:
            return

        inner_flow = flow[flow_index[mask, 0], flow_index[mask, 1], :]
        if (inner_flow == 0).all():
            inner_flow += 1e-8

        dist = sklearn.mixture.GaussianMixture(n_components=2)
        dist.fit(inner_flow[:, ::-1])

        weights = dist.weights_
        means = dist.means_
        covs = dist.covariances_
        caption_text = f"weights: {weights}\nmeans: {means}\ncovs: {covs}"

        sns.jointplot(inner_flow[:, 1], inner_flow[:, 0], kind="hex",
                      stat_func=None) \
           .set_axis_labels(xlabel=caption_text) \
           .savefig(graph)
        plt.close()

        igraph = cv2.imread(graph, cv2.IMREAD_COLOR)
        ibox = frame[bbox.top:bbox.bot, bbox.left:bbox.right, :]
        if igraph.shape[1] > ibox.shape[1]:
            fig_graph = igraph

            fig_bbox = 255 * np.ones((ibox.shape[0]+10, igraph.shape[1], 3))
            pad_left = (igraph.shape[1] - ibox.shape[1]) // 2
            pad_right = pad_left + ibox.shape[1]
            fig_bbox[:ibox.shape[0], pad_left:pad_right, :] = ibox
        else:
            fig_graph = 255 * np.ones((igraph.shape[0], ibox.shape[1], 3))
            pad_left = (ibox.shape[1] - igraph.shape[1]) // 2
            pad_right = pad_left + igraph.shape[1]
            fig_graph[:, pad_left:pad_right, :] = igraph

            fig_bbox = ibox

        fig_whole = np.concatenate((fig_graph, fig_bbox), axis=0)
        cv2.imwrite(graph, fig_whole)

    return lambda graph, bbox: plot_bbox(graph, bbox)

def draw_histogram(i, frame, flow, bboxes, prefix="graph"):
    frame_rows = frame.shape[0]
    frame_cols = frame.shape[1]
    assert(frame.shape[2] == 3)

    rows = flow.shape[0]
    cols = flow.shape[1]
    assert(flow.shape[2] == 2)

    flow_index = np.asarray(tuple(np.ndindex((rows, cols))))
    index_rate = np.asarray((frame_rows // rows, frame_cols // cols))
    frame_index = flow_index * index_rate + (index_rate // 2)

    frame = draw_flow(frame, flow)

    plot_bbox = plot(flow, frame, flow_index, frame_index)
    for bbox in bboxes.itertuples():
        graph = os.path.join("graph", prefix,
                             f"{prefix}{i}_{bbox.name}{bbox.Index}.jpg")
        plot_bbox(graph, bbox)

def vis_histogram(movie, header, flow, bboxes):
    graph_prefix = os.path.basename(movie)
    if not os.path.exists(os.path.join("graph", graph_prefix)):
        os.makedirs(os.path.join("graph", graph_prefix))

    cap = open_video(movie, use_out=False)

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for i in trange(count):
        ret, frame = cap.read()
        if ret is False or i > bboxes.size:
            break

        draw_histogram(i, frame, flow[i], bboxes[i], prefix=graph_prefix)

    cap.release()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("movie")
    parser.add_argument("--reset",
                        action="store_true", default=False)
    return parser.parse_args()

def main():
    args = parse_opt()

    if os.path.exists("flow.pkl") and not args.reset:
        with open("flow.pkl", "rb") as f:
            flow, header = pickle.load(f)
    else:
        flow, header = get_flow(args.movie)
        with open("flow.pkl", "wb") as f:
            pickle.dump((flow, header), f, pickle.HIGHEST_PROTOCOL)
    if os.path.exists("bboxes.pkl") and not args.reset:
        with open("bboxes.pkl", "rb") as f:
            bboxes = pickle.load(f)
    else:
        bboxes = pick_bbox(os.path.join(args.movie, "bbox_dump"))
        with open("bboxes.pkl", "wb") as f:
            pickle.dump(bboxes, f, pickle.HIGHEST_PROTOCOL)

    vis_histogram(args.movie, header, flow, bboxes)

if __name__ == "__main__":
    main()
