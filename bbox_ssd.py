#!/usr/bin/env python3

import argparse

import cv2
import numpy as np
import pandas as pd

import chainer
from chainer import serializers
from chainercv import utils
from chainercv.links import SSD300, SSD512

from mot16 import MOT16Dataset, MOT16Transform

def setup_model(args):
    if args.model == 'ssd300':
        model = SSD300(
            n_fg_class=len(MOT16Dataset.class_map),
            pretrained_model='imagenet')
    elif args.model == 'ssd512':
        model = SSD512(
            n_fg_class=len(MOT16Dataset.class_map),
            pretrained_model='imagenet')

    model.use_preset('evaluate')
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    serializers.load_hdf5(args.param, model)

    return model

def predict(model, img, thresh=0.5):
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.moveaxis(img, 2, 0)

    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    # bbox is (top, left, bot, right) format
    bbox = bbox[score > thresh, :].astype(np.int)
    label = label[score > thresh]
    score = score[score > thresh]

    return pd.DataFrame({
        "name": label,
        "prob": score,
        "left": bbox[:, 1], "top": bbox[:, 0],
        "right": bbox[:, 3], "bot": bbox[:, 2]
    })

def vis(img, bboxes, label_names=None, color=(0, 255, 0)):
    if img.shape[0] == 3:
        img = np.moveaxis(img, 0, 2)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for bbox in bboxes.itertuples():
        cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bot),
                      color, 2)

        if label_names is not None:
            cv2.putText(img, f"{label_names[bbox.name]}: {bbox.prob}",
                        (bbox.left, bbox.top-10),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 255), 1)

    return img

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--model",
                        choices=("ssd300", "ssd512"), default="ssd512")
    parser.add_argument("--param",
                        default="/home/work/takau/6.image/mot/mot16_ssd512.h5")
    parser.add_argument("--gpu", type=int, default=-1)
    return parser.parse_args()

def main():
    args = parse_opt()

    model = setup_model(args)

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    bboxes = predict(model, img)
    drawed_img = vis(img, bboxes)
    cv2.imwrite("weaver.jpg", drawed_img)

if __name__ == "__main__":
    main()
