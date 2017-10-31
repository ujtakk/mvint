#!/usr/bin/env python3

import os
from os.path import join, exists, basename
import argparse
import subprocess

import cv2
import tqdm

from draw import draw_none

def open_video(movie, postfix="out", use_out=True):
    movie_name = join(movie, basename(movie))
    if not exists(movie_name+".mp4"):
        if exists(movie_name+".avi"):
            subprocess.run(
                f"ffmpeg -y -i {movie_name+'.avi'} {movie_name+'.mp4'}",
                shell=True)
        else:
            raise Exception("source movie doesn't exist.")
    movie_file = join(movie, basename(movie)) + ".mp4"

    cap = cv2.VideoCapture(movie_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if use_out:
        out_file = basename(movie) + f"_{postfix}.mp4"
        if exists(out_file):
            os.remove(out_file)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

        return cap, out
    else:
        return cap

def vis(movie, header, draw=draw_none):
    cap, out = open_video(movie)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm.trange(count):
        ret, frame = cap.read()
        if ret is False:
            break

        frame = draw(frame)
        out.write(frame)

    cap.release()
    out.release()

def vis_index(movie, header, draw):
    cap, out = open_video(movie)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm.trange(count):
        ret, frame = cap.read()
        if ret is False:
            break

        frame = draw(frame, i)
        out.write(frame)

    cap.release()
    out.release()

def parse_opt():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def main():
    args = parse_opt()

if __name__ == "__main__":
    main()
