#!/usr/bin/env python3

"""Evaluate MOT16 using SORT
Input:
    movie: str
        Directory name that contains the mp4 movie (encoded one)
        (Name of the movie have to be same as the directory name)
    --thresh: option[float]
    --baseline: option[bool]
    --worst: option[bool]
    --gop: option[int]
"""

from os.path import join, exists, split
import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import trange

from flow import get_flow
from kalman import vis_composed
from mapping import Mapper, SimpleMapper, KalmanMapper
from eval_mot16 import MOT16, eval_mot16

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

from sort.sort import associate_detections_to_trackers
from sort.sort import convert_bbox_to_z
from sort.sort import KalmanBoxTracker

# Custom Tracker
class DeepSORTMapper(Mapper):
# {{{
    def __init__(self, max_iou_distance=0.7, max_age=30, n_init=3,
                 max_cosine_distance=0.2, nn_budget=100):
    # def __init__(self, max_iou_distance=1.0, max_age=30, n_init=3,
    #              max_cosine_distance=-100., nn_budget=100):

        self.metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

        self.detections = []
        self.id_map = dict()

    def predict(self):
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, dets, matches, unmatched_tracks, unmatched_detections):
        id_map = dict()

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, dets[detection_idx])
            id_map[self.tracks[track_idx].track_id] = detection_idx

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
            if self.tracks[track_idx].is_confirmed() and \
               self.tracks[track_idx].time_since_update < 1:
                id_map[self.tracks[track_idx].track_id] = \
                    self.id_map[self.tracks[track_idx].track_id]

        for detection_idx in unmatched_detections:
            self._initiate_track(dets[detection_idx])
            id_map[self._next_id] = detection_idx

        self.id_map = id_map

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

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
        def gated_metric(tracks, detections, track_indices, detection_indices):
            features = np.array([detections[i].feature
                                 for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, detections, track_indices,
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

    def set_detections(self, source, fnum, min_confidence=0.4, min_height=0.0):
        detections = create_detections(source, fnum, min_height)
        detections = [d for d in detections
                      if d.confidence >= min_confidence]

        nms_max_overlap = 1.0
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        self.indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in self.indices]

        self.detections = detections

    def set(self, bboxes):
        self.update_detections(bboxes)

        self.predict()

        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(self.detections)

        self.update(self.detections,
                    matches, unmatched_tracks, unmatched_detections)

    def get(self, bboxes):
        for track in self.tracks:
            if not track.is_confirmed() or track.time_since_update >= 1:
                continue

            idx = self.id_map[track.track_id]
            s = track.to_tlbr()
            bbox = pd.Series({
                "name": bboxes["name"][idx], "prob": bboxes["prob"][idx],
                "left": s[0], "top": s[1], "right": s[2], "bot": s[3]
            })
            # bboxes.loc[idx] = bbox

            yield track.track_id, bbox
            # yield track.track_id, bboxes.loc[idx]

    def update_detections(self, bboxes):
        assert len(self.detections) == len(bboxes)

        for bbox, det in zip(bboxes.itertuples(), self.detections):
            left = bbox.left
            top = bbox.top
            width = bbox.right - bbox.left
            height = bbox.bot - bbox.top
            det.tlwh = np.asarray((left, top, width, height), dtype=np.float)
# }}}

class SORTMapper(Mapper):
# {{{
    def __init__(self, max_age=1, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.id_map = dict()

    def predict(self):
        # get predicted locations from existing trackers.
        tracks = np.zeros((len(self.trackers), 5))

        to_del = []
        for t, track in enumerate(tracks):
            bbox = self.trackers[t].predict()[0]
            track[:] = [bbox[0], bbox[1], bbox[2], bbox[3], 0]

            if np.any(np.isnan(bbox)):
                to_del.append(t)

        tracks = np.ma.compress_rows(np.ma.masked_invalid(tracks))

        for t in reversed(to_del):
            self.trackers.pop(t)

        return tracks

    def update(self, dets, matched, unmatched_detections, unmatched_tracks):
        id_map = dict()
        # update matched trackers with assigned detections
        for t, track in enumerate(self.trackers):
            if t not in unmatched_tracks:
                matched_mask = np.where(matched[:, 1] == t)
                d = matched[matched_mask[0], 0]
                track.update(dets[d, :][0])
                assert d.size == 1
                id_map[track.id] = d[0]

        # create and initialise new trackers for unmatched detections
        for i in unmatched_detections:
            track = KalmanBoxTracker(dets[i,:])
            self.trackers.append(track)
            id_map[track.id] = i

        self.id_map = id_map

    def convert(self, bboxes):
        detections = []
        for bbox in bboxes.itertuples():
            np_bbox = np.zeros((5,))

            np_bbox[0] = bbox.left
            np_bbox[1] = bbox.top
            np_bbox[2] = bbox.right
            np_bbox[3] = bbox.bot

            detections.append(np_bbox)

        return np.asarray(detections)

    def set(self, bboxes):
        self.frame_count += 1

        detections = self.convert(bboxes)
        tracks = self.predict()

        matched, unmatched_detections, unmatched_tracks = \
                associate_detections_to_trackers(detections, tracks)

        self.update(detections, matched, unmatched_detections, unmatched_tracks)

    # bboxes: mut pd.DataFrame
    def get(self, bboxes):
        i = len(self.trackers)
        for track in reversed(self.trackers):
            s = track.get_state()[0]
            s = s.astype(np.int)

            if track.time_since_update < 1 and \
              (track.hit_streak >= self.min_hits or \
               self.frame_count <= self.min_hits):
                idx = self.id_map[track.id]
                bbox = pd.Series({
                    "name": bboxes["name"][idx], "prob": bboxes["prob"][idx],
                    "left": s[0], "top": s[1], "right": s[2], "bot": s[3]
                })
                # bboxes.loc[idx] = bbox

                # +1 as MOT benchmark requires positive
                yield track.id+1, bbox
                # yield track.id+1, bboxes.loc[idx]

            i -= 1

            # remove dead tracklet
            if track.time_since_update > self.max_age:
                self.trackers.pop(i)
# }}}

class MOT16_SORT(MOT16):
# {{{
    SORT_PREFIX = \
        "deep_sort/deep_sort_data/resources/detections/MOT16_POI_train"

    def __init__(self, src_id, src_dir=SORT_PREFIX, dst_dir="result"):
        if not exists(dst_dir):
            os.makedirs(dst_dir)

        self.source = np.load(join(src_dir, src_id) + ".npy")
        self.target = open(join(dst_dir, f"{src_id}.txt"), "w")

        self.min_height = 0.0
        self.frame_count = 1

        self.mapper = SimpleMapper()
        # self.mapper = KalmanMapper()
        # self.mapper = SORTMapper()
        # self.mapper = DeepSORTMapper()

    def pick_bboxes(self):
        det_frames = np.unique(self.source[:, 0].astype(np.int))
        bboxes = [pd.DataFrame() for _ in np.arange(np.max(det_frames))]

        for frame in det_frames:
            detections = create_detections(self.source, frame, self.min_height)
            bbox = np.asarray([d.to_tlbr() for d in detections]).astype(np.int)
            score = np.asarray([d.confidence for d in detections])

            bboxes[frame-1] = pd.DataFrame({
                "name": "", "prob": score,
                "left": bbox[:, 0], "top": bbox[:, 1],
                "right": bbox[:, 2], "bot": bbox[:, 3]
            })

        return pd.Series(bboxes)

    def eval_frame(self, bboxes):
        if isinstance(self.mapper, DeepSORTMapper):
            self.mapper.set_detections(self.source, self.frame_count)

        self.mapper.set(bboxes)

        for bbox_id, bbox_body in self.mapper.get(bboxes):
            left = bbox_body.left
            top = bbox_body.top
            width = bbox_body.right - bbox_body.left
            height = bbox_body.bot - bbox_body.top

            print(f"{self.frame_count},{bbox_id},{left},{top},{width},{height}",
                  file=self.target)

        self.frame_count += 1
# }}}

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_id")
    parser.add_argument("--thresh", type=float, default=0.4,
                        help='')
    parser.add_argument("--baseline",
                        action="store_true", default=False,
                        help='')
    parser.add_argument("--worst",
                        action="store_true", default=False,
                        help='')
    parser.add_argument("--display",
                        action="store_true", default=False,
                        help='')
    parser.add_argument("--gop", type=int, default=12,
                        help='')
    return parser.parse_args()

def main():
    args = parse_opt()
    eval_mot16(args.src_id,
               MOT16=MOT16_SORT,
               thresh=args.thresh,
               baseline=args.baseline,
               worst=args.worst,
               display=args.display,
               gop=args.gop)

    # movie = join("MOT16", "train", args.src_id)
    # flow, header = get_flow(movie)
    # mot = MOT16_SORT(args.src_id)
    # bboxes = mot.pick_bboxes()
    # vis_composed(movie, header, flow, bboxes)

if __name__ == "__main__":
    main()
