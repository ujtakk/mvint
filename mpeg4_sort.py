#!/usr/bin/env python3

from os.path import join, split
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
    def __init__(self, src_id, det_prefix=None, thresh=0.3):
        if det_prefix is None:
            det_prefix = \
                "deep_sort/deep_sort_data/resources/detections/MOT16_POI_train"
        detection_file = join(det_prefix, src_id+".npy")
        self.detections = np.load(detection_file)
        self.thresh = thresh
        self.metric = nn_matching.NearestNeighborDistanceMetric(
                        "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

    def set(self, next_bboxes, prev_bboxes):
        self.tracker.update(prev_bboxes)
        self.tracker.predict()
        pass

    def get(self, bbox):
        pass

    def frame_callback(self, vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            self.detections, frame_idx, min_detection_height
        )
        detections = [d for d in detections if d.confidence >= self.thresh]

        # Run non-maxima suppression.
        boxes = np.asarray([d.tlwh for d in detections])
        scores = np.asarray([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes,
                                                    nms_max_overlap,
                                                    scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        self.tracker.predict()
        self.tracker.update(detections)

        # Store results.
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]
            ])

def pick_mot16_poi_bboxes(path, det_prefix=None, min_height=0):
    src_id = split(path)[-1]
    if det_prefix is None:
        det_prefix = \
            "deep_sort/deep_sort_data/resources/detections/MOT16_POI_train"
    detection_file = join(det_prefix, src_id+".npy")
    poi_det = np.load(detection_file)
    det_frames = np.unique(poi_det[:, 0].astype(np.int))

    bboxes = [pd.DataFrame() for _ in np.arange(np.max(det_frames))]
    for frame in det_frames:
        detections = create_detections(poi_det, frame, min_height)
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
    # mot = MOT16(src_id, mapper=DeepSORTMapper())
    mot = MOT16(src_id, mapper=SimpleMapper(log_id=src_id))

    movie = join(prefix, src_id)
    bboxes = pick_mot16_poi_bboxes(movie)
    flow, header = get_flow(movie, prefix=".")

    cap, out = open_video(movie)

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for index, bbox in enumerate(bboxes):
        if not bbox.empty:
            bboxes[index] = bbox.query(f"prob > {thresh}")

    pos = 0
    for i in trange(count):
        ret, frame = cap.read()
        if ret is False or i > bboxes.size:
            break

        if baseline:
            pos = i
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[pos])
            mot.eval_frame(i, bboxes[pos], do_mapping=True)
        elif header["pict_type"][i] == "I":
            pos = i
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[pos])
            mot.eval_frame(i, bboxes[pos], do_mapping=True)
        elif worst:
            frame_drawed = draw_i_frame(frame, flow[i], bboxes[pos])
            mot.eval_frame(i, bboxes[pos], do_mapping=False)
        else:
            # bboxes[pos] is updated by reference
            frame_drawed = draw_p_frame(frame, flow[i], bboxes[pos])
            mot.eval_frame(i, bboxes[pos], do_mapping=False)

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
