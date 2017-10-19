#!/usr/bin/env python3

from os.path import join, exists, split
import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import trange

from mot16 import pick_mot16_bboxes, detinfo
from flow import get_flow, draw_flow
from annotate import pick_bbox, draw_bboxes
from interp import interp_linear
from interp import draw_i_frame, draw_p_frame, map_flow
from vis import open_video
from mapping import Mapper, SimpleMapper
from bbox_ssd import predict, setup_model
from eval_mot16 import MOT16

from deep_sort.application_util import preprocessing
from deep_sort.application_util import visualization
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort_app import create_detections

class DeepSORTMapper(Mapper):
    SORT_PREFIX = \
        "deep_sort/deep_sort_data/resources/detections/MOT16_POI_train"

    def __init__(self, thresh=0.3):
        max_cosine_distance = 0.2
        nn_budget = 100

        self.thresh = thresh
        self.metric = nn_matching.NearestNeighborDistanceMetric(
                        "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(self.metric)
        self.ids = dict()

    def set(self, next_detections, prev_detections, use_prev=False):
        # if use_prev:
        #     self.tracker.predict()
        #     self.tracker.update(prev_detections)
        # else:
        self.tracker.predict()
        self.tracker.update(next_detections)

    def get(self, bbox):
        return self.tracker.tracks[bbox.Index].track_id

    def get_iter(self, bboxes):
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            yield track.to_tlwh(), track.track_id

class MOT16_SORT:
    def eval_frame(self, fnum, bboxes, do_mapping=False):
        if do_mapping:
            nms_max_overlap = 1.0
            min_confidence = 0.3
            detections = create_detections(self.source, fnum, self.min_height)
            detections = [d for d in detections
                          if d.confidence >= min_confidence]
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            self.indices = preprocessing.non_max_suppression(
                boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in self.indices]
            assert len(detections) == len(bboxes)
            self.mapper.set(detections, self.prev_detections)

        for bbox, obj_id in self.mapper.get_iter(bboxes):
            left = bbox[0]
            top = bbox[1]
            width = bbox[2]
            height = bbox[3]

            print(f"{fnum},{obj_id},{left},{top},{width},{height},-1,-1,-1,-1",
                  file=self.dst_fd)

    SORT_PREFIX = \
        "deep_sort/deep_sort_data/resources/detections/MOT16_POI_train"

    def __init__(self, src_id, src_dir=SORT_PREFIX, dst_dir="result"):
        if not exists(dst_dir):
            os.makedirs(dst_dir)

        self.src_path = join(src_dir, src_id)
        self.dst_fd = open(join(dst_dir, f"{src_id}.txt"), "w")

        detection_file = self.src_path + ".npy"
        self.source = np.load(detection_file)

        self.min_height = 0
        # self.prev_detections = \
        #     create_detections(self.source, 1, self.min_height)
        self.prev_detections = []
        self.mapper = DeepSORTMapper()

        # TEMP
        max_cosine_distance = 0.2
        nn_budget = 100
        self.metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(self.metric)

    def pick_bboxes(self):
        det_frames = np.unique(self.source[:, 0].astype(np.int))

        bboxes = [pd.DataFrame() for _ in np.arange(np.max(det_frames))]
        for frame in det_frames:
            detections = create_detections(self.source, frame, self.min_height)
            bbox = np.asarray([d.to_tlbr() for d in detections]).astype(np.int)
            score = np.asarray([d.confidence for d in detections])

            bboxes[frame-1] = pd.DataFrame({
                "name": "",
                "prob": score,
                "left": bbox[:, 0], "top": bbox[:, 1],
                "right": bbox[:, 2], "bot": bbox[:, 3]
            })

        return pd.Series(bboxes)

    def update_detections(self, bboxes):
        for bbox, det in zip(bboxes.itertuples(), self.prev_detections):
            left = bbox.left
            top = bbox.top
            width = bbox.right - bbox.left
            height = bbox.bot - bbox.top
            det.tlwh = np.asarray((left, top, width, height), dtype=np.float)

def pick_mot16_poi_bboxes(path, det_prefix=None, min_height=0):
    src_id = split(path)[-1]
    if det_prefix is None:
        det_prefix = \
            "deep_sort/deep_sort_data/resources/detections/MOT16_POI_train"
    detection_file = join(det_prefix, src_id+".npy")
    det_source = np.load(detection_file)
    det_frames = np.unique(det_source[:, 0].astype(np.int))

    bboxes = [pd.DataFrame() for _ in np.arange(np.max(det_frames))]
    for frame in det_frames:
        detections = create_detections(det_source, frame, min_height)
        bbox = np.asarray([d.to_tlbr() for d in detections]).astype(np.int)
        score = np.asarray([d.confidence for d in detections])

        bboxes[frame-1] = pd.DataFrame({
            "name": "",
            "prob": score,
            "left": bbox[:, 0], "top": bbox[:, 1],
            "right": bbox[:, 2], "bot": bbox[:, 3]
        })

    return pd.Series(bboxes)

def eval_mot16_sort(src_id, prefix="MOT16/train",
                    thresh=0.0, baseline=False, worst=False, cost_thresh=40000):
    # mot = MOT16(src_id)
    mot = MOT16_SORT(src_id)
    bboxes = mot.pick_bboxes()

    movie = join(prefix, src_id)
    # bboxes = pick_mot16_poi_bboxes(movie)
    flow, header = get_flow(movie, prefix=".")

    cap, out = open_video(movie)

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for index, bbox in enumerate(bboxes):
        if not bbox.empty:
            bboxes[index] = bbox.query(f"prob >= {thresh}")

    pos = 0
    for i in trange(count):
        ret, frame = cap.read()
        if ret is False or i > bboxes.size:
            break

        if baseline:
            pos = i
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[pos])
            mot.eval_frame(i+1, bboxes[pos], do_mapping=True)
        elif header["pict_type"][i] == "I":
            pos = i
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[pos])
            mot.eval_frame(i+1, bboxes[pos], do_mapping=True)
        elif worst:
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[pos])
            mot.eval_frame(i+1, bboxes[pos], do_mapping=False)
        else:
            # bboxes[pos] is updated by reference
            frame_drawed = draw_p_frame(frame, flow[i], bboxes[pos])
            mot.eval_frame(i+1, bboxes[pos], do_mapping=False)

        cv2.rectangle(frame, (width-220, 20), (width-20, 60), (0, 0, 0), -1)
        cv2.putText(frame,
                    f"pict_type: {header['pict_type'][i]}", (width-210, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        out.write(frame_drawed)

    cap.release()
    out.release()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_id")
    parser.add_argument("--baseline",
                        action="store_true", default=False)
    parser.add_argument("--worst",
                        action="store_true", default=False)
    parser.add_argument("--thresh", type=float, default=0.0)
    parser.add_argument("--cost", type=float, default=40000)
    parser.add_argument("--model",
                        choices=("ssd300", "ssd512"), default="ssd512")
    parser.add_argument("--param",
                        default="/home/work/takau/6.image/mot/mot16_ssd512.h5")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_opt()
    # path = join("MOT16/train", args.src)
    # bboxes = pick_mot16_poi_bboxes(path)
    # print(bboxes[42])
    eval_mot16_sort(args.src_id,
                    thresh=args.thresh,
                    baseline=args.baseline,
                    worst=args.worst,
                    cost_thresh=args.cost)

if __name__ == "__main__":
    main()
