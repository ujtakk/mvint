#!/usr/bin/env python3

import os
from os.path import join, exists, basename
import re
import sys
import glob

import argparse
from subprocess import run

import cv2
import numpy as np
import pandas as pd
import tqdm

from vis import open_video

GREP_CMD = "/usr/bin/env grep"
FLOW_CMD = join("mpegflow", "mpegflow")
VIS_CMD = join("mpegflow", "vis")

# refer to gpac/src/media_tools/av_parsers.c for profile and level
# TODO: profile and level option for mpeg4 in ffmpeg won't be active.
# def dump_flow(movie, prefix=None, codec="h264", gop=12):
# def dump_flow(movie, prefix=None, codec="mpeg4", gop=12):
def dump_flow(movie, prefix=None, codec="mpeg2", gop=12):
    if prefix is None:
        prefix = movie

    movie_name = join(movie, basename(movie))
    if not exists(movie_name+".avi"):
        if not exists(movie_name+".mp4"):
            raise Exception("source movie doesn't exist.")

        option = {
            # "h264": f"-codec:v libx264 -sc_threshold 0 -g 12 -b_strategy 0 -bf 2",
            "h264": f"-codec:v libx264 -profile:v baseline -level 3.0 -sc_threshold 0 -g {gop}",
            # "mpeg4": f"-codec:v mpeg4 -profile:v 0 -level 8 -sc_threshold 0 -g {gop}",
            "mpeg4": f"-codec:v mpeg4 -profile:v 0 -level 1 -sc_threshold 0 -g {gop}",
            "mpeg2": f"-codec:v mpeg2video -profile:v 5 -level 8 -sc_threshold 0 -g {gop}",
        }

        if codec not in option:
            Exception("Specified movie type is not supported.")

        run(f"ffmpeg -y -i {movie_name+'.mp4'} {option[codec]} {movie_name+'.avi'}",
            shell=True)

    flow_dir = join(prefix, "mpegflow_dump")
    if not exists(flow_dir):
        os.makedirs(flow_dir)

    # extract motion vectors
    flow_base = join(flow_dir, basename(movie))
    if not exists(flow_base+".txt"):
        run(f"{FLOW_CMD} {movie_name+'.avi'} > {flow_base+'.txt'}", shell=True)
    if not exists(flow_base+"_header.txt"):
        run(f"{GREP_CMD} '^#' {flow_base+'.txt'} > {flow_base+'_header.txt'}",
            shell=True)

def pick_flow(movie, prefix=None):
    if prefix is None:
        prefix = movie

    flow_dir = join(prefix, "mpegflow_dump")
    flow_base = join(flow_dir, basename(movie))

    def convert(header_line):
        header_re = r"# pts=(\d+) frame_index=(-?\d+) pict_type=([IPB?]) " \
                  + r"output_type=(\w+) shape=(\w+) origin=(\w+)"

        string = header_line.strip()
        H = re.match(header_re, string)

        pts = int(H.group(1))
        frame_index = int(H.group(2))
        pict_type = H.group(3)
        output_type = H.group(4)
        shape = tuple(map(int, H.group(5).split('x')))
        origin = H.group(6)

        return pd.DataFrame({"pts": [pts],
                             "frame_index": [frame_index],
                             "pict_type": [pict_type],
                             "output_type": [output_type],
                             "shape_first": [shape[0]],
                             "shape_second": [shape[1]],
                             "origin": [origin]})

    headers = list(map(convert, open(flow_base+"_header.txt").readlines()))
    header = pd.concat(headers, ignore_index=True)

    flow = np.loadtxt(flow_base+".txt").reshape(
            (-1, 2, header["shape_first"][0]//2, header["shape_second"][1]))
    flow = np.moveaxis(flow, 1, 3)

    return flow, header

def get_flow(movie, vis=False, occupancy=False, prefix=None, gop=12):
    dump_flow(movie, prefix=prefix, gop=gop)

    movie_file = join(movie, basename(movie)) + ".avi"

    if vis:
        vis_dir = join(prefix, "vis_dump")
        if not exists(vis_dir):
            os.makedirs(vis_dir)

        # visualize motion vectors
        if not glob.glob(join(vis_dir, "*.png")):
            run(f"{FLOW_CMD} {movie_file} | {VIS_CMD} {movie_file} {vis_dir}",
                shell=True)

    # visualize motion vectors and occupancy info
    if occupancy:
        occu_dir = join(movie, "vis_dump_occupancy")
        if not exists(occu_dir):
            os.makedirs(occu_dir)

        run(f"{FLOW_CMD} --occupancy {movie_file} "
            + "| {VIS_CMD} --occupancy {movie_file} {occu_dir}",
             shell=True)

    flow, header = pick_flow(movie, prefix=prefix)

    return flow, header

def divergence(field):
    grad_x = np.gradient(field[:, :, 0], axis=1)
    grad_y = np.gradient(field[:, :, 1], axis=0)
    div = grad_x + grad_y
    return div

def draw_flow(frame, flow):
    def draw_arrow(frame, start, end, len=2.0, alpha=20.0,
                   line_color=(0, 0, 255), start_color=(0, 255, 0)):
        cv2.line(frame, start, end, line_color, 2)
        cv2.circle(frame, end, 1, (0, 255, 255), -1)
        cv2.circle(frame, start, 1, (255, 255, 0), -1)
        return frame

    def draw_grad(frame, point, val, alpha=0.3):
        start = (point[0] - 6, point[1] - 6)
        end = (point[0] + 6, point[1] + 6)
        color = int(np.clip(127 + 8*val, 0, 255))
        cv2.rectangle(frame, start, end, (color, 0, 0), -1)
        return frame

    frame_rows = frame.shape[0]
    frame_cols = frame.shape[1]
    assert(frame.shape[2] == 3)

    rows = flow.shape[0]
    cols = flow.shape[1]
    assert(flow.shape[2] == 2)

    div = divergence(flow)

    for i in range(rows):
        for j in range(cols):
            dx = flow[i, j, 0]
            dy = flow[i, j, 1]

            start = (j * frame_cols / cols + frame_cols / cols / 2,
                     i * frame_rows / rows + frame_rows / rows / 2)
            start = tuple(map(int, start))

            end = (start[0] + dx, start[1] + dy)
            end = tuple(map(int, end))

            frame = draw_grad(frame, start, div[i, j])
            frame = draw_arrow(frame, start, end)

    return frame

def vis_flow(movie, flow, draw=draw_flow):
    cap, out = open_video(movie)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm.trange(count):
        ret, frame = cap.read()
        if ret is False:
            break

        frame = draw(frame, flow[i])
        out.write(frame)

    cap.release()
    out.release()

def parseopt():
    parser = argparse.ArgumentParser(
        description="script for extracting motion vectors")
    parser.add_argument("movie",
                        default="mpi_sinel_final_alley_1",
                        help="source movie dir to extract motion vectors")
    parser.add_argument("--occupancy", "-o",
                        action="store_true", default=False,
                        help="dump occupancy enabled version")
    return parser.parse_args()

def main():
    args = parseopt()
    dump_flow(args.movie)
    flow, header = pick_flow(args.movie)
    vis_flow(args.movie, flow)

if __name__ == "__main__":
    main()
