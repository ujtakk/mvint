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
from interp import draw_i_frame, draw_p_frame
from kalman import KalmanInterpolator, interp_kalman
from vis import open_video
from mapping import Mapper, SimpleMapper
from bbox_ssd import predict, setup_model
from eval_mot16 import eval_mot16

from deep_sort.application_util import preprocessing
from deep_sort.application_util import visualization
from deep_sort.deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.deep_sort.detection import Detection
# from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import kalman_filter
from deep_sort.deep_sort import linear_assignment
from deep_sort.deep_sort import iou_matching
from deep_sort.deep_sort.track import Track
from deep_sort.deep_sort_app import create_detections

# Custom Tracker
class DeepSORTMapper(Mapper):
    SORT_PREFIX = \
        "deep_sort/deep_sort_data/resources/detections/MOT16_POI_train"

    def __init__(self, max_iou_distance=0.7, max_age=30, n_init=3,
                 max_cosine_distance=0.2, nn_budget=100):
                 # max_cosine_distance=0.0, nn_budget=100):

        self.metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.ids = dict()

    def predict(self):
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        id_map = dict()
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
            id_map[self.tracks[track_idx].track_id] = detection_idx
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
            if self.tracks[track_idx].is_confirmed() \
            and self.tracks[track_idx].time_since_update < 1:
                id_map[self.tracks[track_idx].track_id] = \
                    self.ids[self.tracks[track_idx].track_id]
        for detection_idx in unmatched_detections:
            id_map[self._next_id] = detection_idx
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        self.ids = id_map

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1

    def set(self, next_detections, prev_detections, use_prev=False):
        if use_prev:
            self.predict()
            self.update(prev_detections)
        else:
            self.predict()
            self.update(next_detections)

    def get(self, bboxes):
        for track in self.tracks:
            if not track.is_confirmed() or track.time_since_update >= 1:
                continue

            bbox_idx = self.ids[track.track_id]
            yield track.track_id, bboxes.loc[bbox_idx]

class SORTMapper(Mapper):
    def __init__(self):
        pass

    def set(self, next_bboxes, prev_bboxes):
        pass

    def get(self, bbox):
        pass

class MOT16_SORT(MOT16):
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
        self.prev_bboxes = pd.DataFrame()
        self.mapper = SimpleMapper()
        # self.prev_detections = []
        # self.mapper = DeepSORTMapper()

    def eval_frame(self, fnum, bboxes, do_mapping=False):
        # if do_mapping:
        #     min_confidence = 0.3
        #     detections = create_detections(self.source, fnum, self.min_height)
        #     detections = [d for d in detections
        #                   if d.confidence >= min_confidence]
        #     # nms_max_overlap = 1.0
        #     # boxes = np.array([d.tlwh for d in detections])
        #     # scores = np.array([d.confidence for d in detections])
        #     # self.indices = preprocessing.non_max_suppression(
        #     #     boxes, nms_max_overlap, scores)
        #     # detections = [detections[i] for i in self.indices]
        #     assert len(detections) == len(bboxes)
        #     self.mapper.set(detections, self.prev_detections)
        #     self.prev_detections = detections
        # else:
        #     self.mapper.set(None, self.prev_detections, use_prev=True)

        if do_mapping:
            self.mapper.set(bboxes, self.prev_bboxes)

        for bbox_id, bbox_body in self.mapper.get(bboxes):
            left = bbox_body.left
            top = bbox_body.top
            width = bbox_body.right - bbox_body.left
            height = bbox_body.bot - bbox_body.top

            print(f"{fnum},{bbox_id},{left},{top},{width},{height},-1,-1,-1,-1",
                  file=self.dst_fd)

        # self._update_detections(bboxes)
        self.prev_bboxes = bboxes

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

    def _update_detections(self, bboxes):
        for bbox, det in zip(bboxes.itertuples(), self.prev_detections):
            left = bbox.left
            top = bbox.top
            width = bbox.right - bbox.left
            height = bbox.bot - bbox.top
            det.tlwh = np.asarray((left, top, width, height), dtype=np.float)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_id")
    parser.add_argument("--thresh", type=float, default=0.4)
    parser.add_argument("--baseline",
                        action="store_true", default=False)
    parser.add_argument("--worst",
                        action="store_true", default=False)
    return parser.parse_args()

def main():
    args = parse_opt()
    eval_mot16(args.src_id,
               MOT16=MOT16_SORT,
               thresh=args.thresh,
               baseline=args.baseline,
               worst=args.worst)

if __name__ == "__main__":
    main()
