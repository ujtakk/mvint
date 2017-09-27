#!/usr/bin/env python3

import argparse

import cv2

from flow import dump_flow, pick_flow, vis_flow
from detect import Detector

def draw_box_flow(frame, flow, detector):
    bbox, label, score = detector(frame)

    frame_rows = frame.shape[0]
    frame_cols = frame.shape[1]
    assert(frame.shape[2] == 3)

    rows = flow.shape[0]
    cols = flow.shape[1]
    assert(flow.shape[2] == 2)

    for i in range(rows):
        for j in range(cols):
            dx = flow[i, j, 0]
            dy = flow[i, j, 1]

            start = (j * frame_cols / cols + frame_cols / cols / 2,
                     i * frame_rows / rows + frame_rows / rows / 2)
            start = tuple(map(int, start))

            end = (start[0] + dx, start[1] + dy)
            end = tuple(map(int, end))

            cv2.line(frame, start, end, (0, 0, 255), 2)

    for box in bbox:
        cv2.rectangle(frame, tuple(box[0:2]), tuple(box[2:4]),
                      (0, 255, 0), 2)

    return frame

def parseopt():
    parser = argparse.ArgumentParser()
    parser.add_argument("movie",
                        default="mpi_sinel_final_alley_1",
                        help="source movie dir to extract motion vectors")
    parser.add_argument("--model",
                        choices=("ssd300", "ssd512"), default="ssd300")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--pretrained_model", default="voc0712")
    return parser.parse_args()

def main():
    args = parseopt()
    dump_flow(args.movie)
    flow = pick_flow(args.movie)
    detector = Detector()
    draw_box_flow_func = \
        lambda movie, flow: draw_box_flow(movie, flow, detector)
    vis_flow(args.movie, flow, draw=draw_box_flow_func)

if __name__ == "__main__":
    main()
