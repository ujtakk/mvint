#!/usr/bin/env python3

import argparse

from mot16 import pick_mot16_bboxes
from flow import get_flow, draw_flow
from annotate import pick_bbox, draw_bboxes
from interp import interp_linear
from interp import draw_i_frame, draw_p_frame, map_flow
from vis import open_video
from mapping import Mapper
from bbox_ssd import predict, setup_model
from eval_mot16 import MOT16

from deep_sort.application_util import preprocessing
from deep_sort.application_util import visualization
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker

class DeepSORT():
    def __init__(self, src_id, det_prefix=None, thresh=0.3):
        if det_prefix is None:
            det_prefix = \
                "deep_sort/deep_sort_data/resources/detections/MOT16_POI_train"
        detection_file = join(det_prefix, src_id+".npy")
        self.detections = np.load(detection_file)
        self.thresh = thresh

    def frame_callback(self, vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            self.detections, frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= self.thresh]

        # Run non-maxima suppression.
        boxes = np.asarray([d.tlwh for d in detections])
        scores = np.asarray([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

def eval_mot16_sort():
    mot = MOT16(src_id, cost_thresh=cost_thresh)

    movie = join(prefix, src_id)
    bboxes = pick_mot16_bboxes(movie)
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
        else:
            # bboxes[pos] is updated by reference
            frame_drawed = draw_p_frame(frame, flow[i], bboxes[pos])
            # frame_drawed = draw_i_frame(frame, flow[i], bboxes[pos])
            mot.eval_frame(i, bboxes[pos], do_mapping=False)
            # mot.eval_frame(i, bboxes[pos], do_mapping=True)

        cv2.rectangle(frame, (width-220, 20), (width-20, 60), (0, 0, 0), -1)
        cv2.putText(frame,
                    f"pict_type: {header['pict_type'][i]}", (width-210, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        out.write(frame_drawed)

    cap.release()
    out.release()

def parse_opt():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def main():
    args = parse_opt()
    sort = DeepSORT()

if __name__ == "__main__":
    main()
