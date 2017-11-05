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

# find MOT16/ -name "*.avi" | xargs rm
# rm -rf mpegflow_dump

# echo $MOT_TRAIN | xargs -P 8 -n 1 python eval_mot16.py --baseline
# echo $MOT_TRAIN | xargs -P 8 -n 1 python eval_mot16.py --worst
# echo $MOT_TRAIN | xargs -P 8 -n 1 python eval_mot16.py

# echo $MOT_TRAIN | xargs -P 8 -n 1 python sort.py --baseline
# echo $MOT_TRAIN | xargs -P 8 -n 1 python sort.py --worst
echo $MOT_TRAIN | xargs -P 8 -n 1 python sort.py --display

# for mot in $MOT_TRAIN; do
#   python sort.py $mot
# done

