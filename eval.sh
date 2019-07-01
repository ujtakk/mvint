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

# find MOT16/ -name "*.avi" | xargs rm
# rm -rf mpegflow_dump

# echo $MOT_TRAIN | xargs -P 8 -n 1 python sort.py --display

# echo $MOT_TRAIN | xargs -P 8 -n 1 python eval_mot16.py --baseline
# echo $MOT_TRAIN | xargs -P 8 -n 1 python eval_mot16.py --worst
# echo $MOT_TRAIN | xargs -P 8 -n 1 python eval_mot16.py

# echo $MOT_TRAIN | xargs -P 8 -n 1 python sort.py --baseline
# echo $MOT_TRAIN | xargs -P 8 -n 1 python sort.py --worst
# echo $MOT_TRAIN | xargs -P 8 -n 1 python sort.py

# for mot in $MOT_TRAIN; do
#   # python eval_mot16.py $mot --baseline
#   # python eval_mot16.py $mot --worst
#   # python eval_mot16.py $mot
#   # python eval_mot16.py $mot --thresh 0.4
#   # python sort.py $mot --baseline
#   # python sort.py $mot --worst
#   python sort.py $mot
# done

# (cd motchallenge-devkit;
# matlab -log -r "demo_evalMOT16; exit" > ../matlab.txt
# cat ../matlab.txt
# )

# for mot in $MOT_TEST; do
#   python test.py $mot
# done
# echo $MOT_TEST | xargs -P 8 -n 1 python test.py

mkdir -p gop
for gop in `seq 1 20`; do
  echo "gop-$gop"
  find MOT16/ -name "*.avi" | xargs rm
  rm -rf mpegflow_dump

  echo $MOT_TRAIN | xargs -P 8 -n 1 python sort.py --baseline --gop $gop
  (cd motchallenge-devkit;
    matlab -log -r "demo_evalMOT16; exit" > ../gop/base_$gop.txt
  )

  echo $MOT_TRAIN | xargs -P 8 -n 1 python sort.py --worst --gop $gop
  (cd motchallenge-devkit;
    matlab -log -r "demo_evalMOT16; exit" > ../gop/worst_$gop.txt
  )

  # echo $MOT_TRAIN | xargs -P 8 -n 1 python sort.py --gop $gop
  # (cd motchallenge-devkit;
  #   matlab -log -r "demo_evalMOT16; exit" > ../gop/lin_$gop.txt
  #   # matlab -log -r "demo_evalMOT16; exit" > ../gop/link_$gop.txt
  #   # matlab -log -r "demo_evalMOT16; exit" > ../gop/divk_$gop.txt
  # )
done

