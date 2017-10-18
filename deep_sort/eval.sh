#!/bin/bash

MOT_TRAIN="
MOT16-02
MOT16-04
MOT16-05
MOT16-09
MOT16-10
MOT16-11
MOT16-13
"

mkdir -p result

# for i in $MOT_TRAIN; do
#   python deep_sort_app.py \
#     --sequence_dir=MOT16/train/$i \
#     --detection_file=deep_sort_data/resources/detections/MOT16_POI_train/$i.npy \
#     --output_file=result/$i.txt \
#     --min_confidence=0.3 \
#     --nn_budget=100
# done

python deep_sort_app.py \
  --sequence_dir=MOT16/train/MOT16-04 \
  --detection_file=deep_sort_data/resources/detections/MOT16_POI_train/MOT16-04.npy \
  --output_file=result/MOT16-04.txt \
  --min_confidence=0.3 \
  --nn_budget=100
