#!/usr/bin/env python3

"""DEPRECATED
"""

import os
from os.path import join
from glob import glob
import argparse
import multiprocessing

import pandas as pd
import youtube_dl

def youtube_download(youtube_id, out_dir="youtube"):
    ydl_opts = {
        "format": "mp4",
        "outtmpl": os.path.join(out_dir, "%(id)s", "%(id)s.mp4"),
        "ignoreerrors": True,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(youtube_id)

def youtube_bb_csv_download(csv_path):
    ytb_src = pd.read_csv(csv_path, names=("youtube_id", "timestamp_ms",
                                           "class_id", "class_name",
                                           "object_id", "object_presence",
                                           "xmin", "xmax", "ymin", "ymax"))

    youtube_download(ytb_src["youtube_id"].unique())

def youtube_bb_csv(csv_path):
    ytb_src = pd.read_csv(csv_path, names=("youtube_id", "timestamp_ms",
                                           "class_id", "class_name",
                                           "object_id", "object_presence",
                                           "xmin", "xmax", "ymin", "ymax"))
    return ytb_src

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", "-i",
                        help="Input Youtube ID")
    parser.add_argument("--csv", "-c",
                        default="/home/work/vision/youtube-bb",
                        help="Input Youtube-BB Detection CSV")
    return parser.parse_args()

def main():
    args = parse_opt()
    # ids = [
    #     "9qSEfcIfYbw",
    #     "IGJ2jMZ-gaI",
    #     "_w9xVpsGt0c",
    # ]
    # youtube_download(ids)
    csv_path = os.path.join(args.csv, "yt_bb_detection_validation.csv")
    youtube_bb_csv_download(csv_path)

if __name__ == "__main__":
    main()
