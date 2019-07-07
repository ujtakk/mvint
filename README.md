[WIP] MVint (for Tracking performance Evaluation)
============================================================

MVint is a tracking framework enhanced by motion vectors.
This project is written and used for tracking performance evaluation in our paper [1].



Requirements
--------------------------------------------------

### Building Darknet
As the utility for MVint, we use some modified version of Darknet [2]
(placed under `darknet/`).
Before using MVint scripts, we have to build it.
To build Darknet, execute `make` from the root directory as:
```
make -C darknet
```

If you want to perform detections using GPU/CUDNN/OpenCV, then command will be:
```
make -C darknet GPU=1 CUDNN=1 OPENCV=1
```

### Building MPEG-flow
We also use MPEG-flow [3] implementation to generate motion vectors
from the target movie encoded by some specific codec.
MPEG-flow contains two tools named `mpegflow` and `vis`.
`mpegflow` uses libavcodec and `vis` uses OpenCV3,
so we first have to prepare libavcodec / OpenCV3.
Then, we build `mpegflow` and `vis` from the root directory as:
```
make -C mpegflow mpegflow vis
```

### Installing MVint dependencies
MVint is developed and tested on Python 3.6.5.

Python libraries can be set up by executing the commands below:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Also for MOT16 [4] evaluation, we have to setup MOT16 dataset and MATLAB.



Usage
--------------------------------------------------

### Extract bounding boxes using `darknet`
To evaluate tracking performance, we first extract the bounding boxes offline.
(Note: Offline detection is only for the convenience of evaluation.
MVint is basically designed to be used in online tracking.)

To extract the bounding boxes of given movie, execute the command below:
(We used `movie/hoge/hoge.mp4` as an example movie.
You have to place your target movie `TARGET.mp4` to the directory `movie/TARGET`.)
```
./darknet/extract.sh movie/hoge
```
As the default extractor, we utilized Darknet implementation of YOLO9000 [2].
Extracted bounding boxes are dumped into `movie/hoge/bbox_dump`.
There is also the movie (`movie/hoge/out_hoge.mp4`) that bounding boxes are annotated.

### Extract motion vectors using `mpegflow`

We call MPEG-flow tools from python scripts to generate motion vectors.
You can visualize the motion vectors field of specific movie `movie/hoge/hoge.mp4`
by the `flow.py` script as below:
```
python flow.py movie/hoge
```
Then the result movie is saved as `movie/hoge/hoge_flow.mp4`.
Extracted motion vectors are dumped into `movie/hoge/mpegflow_dump`.

To combine bounding boxes extracted above for visualizing,
we can use the `annotate.py` script as below:
```
python annotate.py movie/hoge
```
The result movie is saved as `movie/hoge/hoge_annotate.mp4`.

### Qualititative Evaluation
In prior to quantitative evaluation,
we can qualititatively check results produced by MVint.

```
python interp.py movie/hoge
```

```
python kalman.py movie/hoge
```

### Quantitative Evaluation using MOT16
Finally we will perform quantitative evaluation using MOT16.

```
./eval.sh
```



The example MOT16 Result
--------------------------------------------------

With the evaluation using MOT16, we reproduced some example result as below:
```
```
Tracking performance may differ when using different codec settings,
interpolation settings, and the detector.



License
--------------------------------------------------

Sources are licensed under GPLv3 (inherited from `deep_sort`, `sort` sub-repository)

The sample movie (`movie/`, [URL](https://))
is redistributed under Creative Commons.



References
--------------------------------------------------

[1] T. Ujiie, M. Hiromoto, and T. Sato, "Interpolation-Based Object Detection Using Motion Vectors for Embedded Real-Time Tracking Systems", In Proceedings of CVPRW, 2018

[2] J. Redmon and A. Farhadi, "YOLO9000: Better, Faster, Stronger", In Proceedings of CVPR, 2017

[3] V. Kantorov and I. Laptev, "Efficient feature extraction, encoding and classification for action recognition", In Proceedings of CVPR, 2014

[4] A. Milan, L. L. Taixé, I. D. Reid, S. Roth and K. Schindler, "MOT16: A Benchmark for Multi-Object Tracking", arxiv:1603.00831

[5] A. Bewley, Z. Ge, L. Ott, F. Ramos, and B. Upcroft, "Simple Online and Realtime Tracking", In Proceedings of ICIP, 2016

[6] N. Wojke, A. Bewley and D. Paulus, "Simple Online and Realtime Tracking with a Deep Association Metric", In Proceedings of ICIP, 2017

When you used this project for writing the paper,
please cite our paper based on the bibtex below:

```
@inproceedings{mvint,
  author    = {Takayuki Ujiie and Masayuki Hiromoto and Takashi Sato},
  booktitle = {Proceedings of CVPR Workshop},
  title     = {Interpolation-Based Object Detection Using Motion Vectors for Embedded Real-Time Tracking Systems},
  year      = {2018},
}
```
