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

# SOURCE=eval_mot16.py
SOURCE=mpeg4_sort.py

# echo $MOT_TRAIN | xargs -P 8 -n 1 -t python $SOURCE --thresh 0.1 --cost 3000
echo $MOT_TRAIN | xargs -P 8 -n 1 -t python $SOURCE --thresh 0.3 --worst
