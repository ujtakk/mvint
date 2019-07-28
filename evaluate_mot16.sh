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

MOT_TEST="
MOT16-01
MOT16-03
MOT16-06
MOT16-07
MOT16-08
MOT16-12
MOT16-14
"
find MOT16/ -name "*.avi" | xargs rm
rm -rf mpegflow_dump

source venv/bin/activate
# echo $MOT_TRAIN | xargs -P 8 -n 1 python sort.py --display

# echo $MOT_TRAIN | xargs -P 8 -n 1 python eval_mot16.py --baseline
# echo $MOT_TRAIN | xargs -P 8 -n 1 python eval_mot16.py --worst
# echo $MOT_TRAIN | xargs -P 8 -n 1 python eval_mot16.py

# echo $MOT_TRAIN | xargs -P 8 -n 1 python sort.py --baseline
# echo $MOT_TRAIN | xargs -P 8 -n 1 python sort.py --worst
# echo $MOT_TRAIN | xargs -P 8 -n 1 python sort.py

for mot in $MOT_TRAIN; do
  # python eval_mot16.py $mot --baseline
  # python eval_mot16.py $mot --worst
  # python eval_mot16.py $mot
  # python sort.py $mot --baseline
  # python sort.py $mot --worst
  python sort.py $mot
done

(cd motchallenge-devkit;
  DISPLAY= matlab -log -r "demo_evalMOT16; exit" > ../matlab.txt
  cat ../matlab.txt
)

