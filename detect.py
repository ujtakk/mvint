#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt

import numpy as np
import chainer
from chainercv.datasets import voc_detection_label_names
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv import utils
from chainercv.visualizations import vis_bbox

class Detector:
    def __init__(self, model="ssd300", gpu=-1, pretrained_model="voc0712"):
        if model == "ssd300":
            self.model = SSD300(
                n_fg_class=len(voc_detection_label_names),
                pretrained_model=pretrained_model)
        elif model == "ssd512":
            self.model = SSD512(
                n_fg_class=len(voc_detection_label_names),
                pretrained_model=pretrained_model)

        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            self.model.to_gpu()

    def __call__(self, image):
        if isinstance(image, str):
            img = utils.read_image(image, color=True)
        else:
            img = np.moveaxis(image, [2, 0], [0, 2])
        bboxes, labels, scores = self.model.predict([img])
        bbox, label, score = bboxes[0], labels[0], scores[0]

        return bbox, label, score

def detect(image, model="ssd300", gpu=-1, pretrained_model="voc0712"):
    detector = Detector(model, gpu, pretrained_model)
    img = utils.read_image(image, color=True)

    bbox, label, score = detector(image)

    vis_bbox(
        img, bbox, label, score, label_names=voc_detection_label_names)
    plt.show()

def parseopt():
    parser = argparse.ArgumentParser(description="detection script by chainercv SSD300")
    parser.add_argument("image",
                        help="source image to be detected")
    parser.add_argument("--model",
                        choices=("ssd300", "ssd512"), default="ssd300")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--pretrained_model", default="voc0712")
    return parser.parse_args()

def main():
    args = parseopt()
    detect(args.image, args.model, args.gpu, args.pretrained_model)

if __name__ == "__main__":
    main()
