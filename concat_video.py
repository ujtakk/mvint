#!/usr/bin/env python3

"""Snippets used to concat movies (not maintained)
"""

import os
import cv2
import numpy as np

left = cv2.VideoCapture("corgi_out.mp4")
right = cv2.VideoCapture("corgi_base.mp4")

fps = left.get(cv2.CAP_PROP_FPS)
width = int(left.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(left.get(cv2.CAP_PROP_FRAME_HEIGHT))
count = int(left.get(cv2.CAP_PROP_FRAME_COUNT))

if os.path.exists("corgi_concat.mp4"):
    os.remove("corgi_concat.mp4")
fourcc = cv2.VideoWriter_fourcc(*"avc1")
out = cv2.VideoWriter("corgi_concat.mp4", fourcc, fps, (width, 2*height+10))

hspace = np.zeros((height, 10, 3))
vspace = np.zeros((10, width, 3))

for i in range(count):
    ret, frame_left = left.read()
    if ret is False:
        break

    ret, frame_right = right.read()
    if ret is False:
        break

    # frame_out = np.concatenate((frame_left, hspace, frame_right), axis=1) \
    #               .astype(np.uint8)
    frame_out = np.concatenate((frame_left, vspace, frame_right), axis=0) \
                  .astype(np.uint8)
    out.write(frame_out)

left.release()
right.release()
out.release()
