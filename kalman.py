#!/usr/bin/env python3

import argparse

import cv2
import numpy as np
import pandas as pd

def kalman_filter(kalman, center, flow_mean, alpha=1.0):
    kalman.transitionMatrix = (1.0 * np.eye(2)).astype(np.float32)
    kalman.controlMatrix = (alpha * np.eye(2)).astype(np.float32)
    kalman.measurementMatrix = (1.0 * np.eye(2)).astype(np.float32)
    kalman.processNoiseCov = (1e-5 * np.eye(2)).astype(np.float32)
    # kalman.processNoiseCov = (0.0 * np.eye(2)).astype(np.float32)
    kalman.measurementNoiseCov = (1e-1 * np.ones((2, 2))).astype(np.float32)
    # kalman.measurementNoiseCov = (0.0 * np.ones((2, 2))).astype(np.float32)
    kalman.errorCovPost = (1.0 * np.ones((2, 2))).astype(np.float32)
    # kalman.errorCovPost = (0.0 * np.ones((2, 2))).astype(np.float32)
    kalman.statePost = (center.reshape(2, 1)).astype(np.float32)

    state = kalman.predict(flow_mean)
    new_center = np.dot(kalman.measurementMatrix, state) \
               + np.dot(kalman.measurementNoiseCov, np.random.randn(2, 1))
    new_center = new_center.astype(np.float32)
    kalman.correct(new_center)

    return new_center.flatten()

def interp_kalman(bbox, flow_map, index_rate):
    flow_mean = np.mean(flow_map[bbox.top:bbox.bot, bbox.left:bbox.right,
                                 :], axis=(0, 1))
    flow_mean = np.nan_to_num(flow_mean)

    alpha = 2.0 * index_rate ** 2
    center = np.asarray(((bbox.bot+bbox.top)/2, (bbox.right+bbox.left)/2))

    kalman = cv2.KalmanFilter(2, 2, 2)

    new_center = kalman_filter(kalman, center, flow_mean, alpha)

    frame_mean = new_center - center

    left  = bbox.left + frame_mean[0]
    top   = bbox.top + frame_mean[1]
    right = bbox.right + frame_mean[0]
    bot   = bbox.bot + frame_mean[1]

    height = flow_map.shape[0]
    width = flow_map.shape[1]

    left  = np.clip(left, 0, width-1).astype(np.int)
    top   = np.clip(top, 0, height-1).astype(np.int)
    right = np.clip(right, 0, width-1).astype(np.int)
    bot   = np.clip(bot, 0, height-1).astype(np.int)

    return pd.Series({"name": bbox.name, "prob": bbox.prob,
        "left": left, "top": top, "right": right, "bot": bot}), frame_mean

def parse_opt():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def main():
    args = parse_opt()

if __name__ == "__main__":
    main()
