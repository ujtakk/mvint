#!/usr/bin/env python3

import os
from os.path import join, exists, basename
import sys
import subprocess
import argparse

FLOW_CMD = join("mpegflow", "mpegflow")
VIS_CMD = join("mpegflow", "vis")

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

    flow_dir = join(args.movie, "mpegflow_dump")
    if not exists(flow_dir):
        os.makedirs(flow_dir)

    vis_dir = join(args.movie, "vis_dump")
    if not exists(vis_dir):
        os.makedirs(vis_dir)

    movie_name = join(args.movie, basename(args.movie))
    if not exists(movie_name+".avi"):
        if exists(movie_name+".mp4"):
            subprocess.run(f"ffmpeg -y -i {movie_name+'.mp4'} {movie_name+'.avi'}", shell=True)
        else:
            raise Exception("source movie doesn't exist.")

    movie_file = movie_name + ".avi"

    # extract motion vectors
    flow_base = join(flow_dir, basename(args.movie))
    subprocess.run(f"{FLOW_CMD} {movie_file} > {flow_base+'.txt'}",
                    shell=True)
    subprocess.run(f"{FLOW_CMD} --raw {movie_file} > {flow_base+'_raw.txt'}",
                    shell=True)

    #visualize motion vectors
    subprocess.run(f"{FLOW_CMD} {movie_file} | {VIS_CMD} {movie_file} {vis_dir}",
                    shell=True)

    if args.occupancy:
        occu_dir = join(args.movie, "vis_dump_occupancy")
        os.makedirs(occu_dir)

        subprocess.run(f"{FLOW_CMD} --occupancy {movie_file} | {VIS_CMD} --occupancy {movie_file} {occu_dir}",
                        shell=True)


if __name__ == "__main__":
    main()
