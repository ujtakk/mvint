#!/bin/sh

MOT_TRAIN="
MOT16-02
MOT16-04
MOT16-05
MOT16-09
MOT16-10
MOT16-11
MOT16-13
"

# echo $MOT_TRAIN | xargs -P 8 -n 1 -t python eval_mot16.py --thresh 0.1
# echo $MOT_TRAIN | xargs -P 8 -n 1 -t python mpeg4_sort.py --thresh 0.3 --baseline
# echo $MOT_TRAIN | xargs -P 8 -n 1 -t python mpeg4_sort.py --thresh 0.3 --worst
echo $MOT_TRAIN | xargs -P 8 -n 1 -t python mpeg4_sort.py --thresh 0.3
