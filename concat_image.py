#!/usr/bin/env python3

import cv2
import numpy as np

# first = 515
# last = 965
# first = 1715
# last = first+7
first = 1717
last = first+3

# row0 = np.asarray([])
row0 = cv2.imread(f"corgi_full/{first-1}.png", cv2.IMREAD_COLOR)
hspace = 255 * np.ones((row0.shape[0], 10, 3))
for i in range(first, last):
    img = cv2.imread(f"corgi_full/{i}.png", cv2.IMREAD_COLOR)
    row0 = np.concatenate((row0, hspace, img), axis=1)

# row1 = np.asarray([])
row1 = cv2.imread(f"corgi_out/{first-1}.png", cv2.IMREAD_COLOR)
for i in range(first, last):
    img = cv2.imread(f"corgi_out/{i}.png", cv2.IMREAD_COLOR)
    row1 = np.concatenate((row1, hspace, img), axis=1)

# row2 = np.asarray([])
row2 = cv2.imread(f"corgi_base/{first-1}.png", cv2.IMREAD_COLOR)
for i in range(first, last):
    img = cv2.imread(f"corgi_base/{i}.png", cv2.IMREAD_COLOR)
    row2 = np.concatenate((row2, hspace, img), axis=1)
vspace = 255 * np.ones((10, row2.shape[1], 3))

imgs = np.concatenate((row0, vspace, row1, vspace, row2), axis=0)
cv2.imwrite("out.png", imgs)
