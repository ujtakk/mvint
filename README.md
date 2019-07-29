MVint (for Tracking performance Evaluation)
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
`mpegflow` uses FFmpeg (libavcodec) and `vis` uses OpenCV3,
so we first have to prepare FFmpeg (libavcodec) / OpenCV3.

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

Also for MOT16 [4] evaluation, we have to prepare MOT16 dataset and
MATLAB-based development kit.
All commands for the preparation are scripted to `prepare_mot16.sh`,
so execute the script as:
```
./prepare_mot16.sh
```
(NOTE: Development kit is distributed using Mercurial.
If you haven't installed Mercurial, please install it.)


Usage
--------------------------------------------------

### Extract bounding boxes using `darknet`
To evaluate tracking performance, we first extract the bounding boxes offline.
(Note: Offline detection is only for the convenience of evaluation.
MVint is basically designed to be used in online tracking.)

To extract the bounding boxes of given movie, execute the command below:  
(We used `movie/corgi/corgi.mp4` as an example movie.
You have to place your target movie `TARGET.mp4` to the directory `movie/TARGET`.)
```
./darknet/extract.sh movie/corgi
```
As the default extractor, we utilized Darknet implementation of YOLO9000 [2].

Extracted bounding boxes are dumped into `movie/corgi/bbox_dump`.
There is also the movie (`movie/corgi/out_corgi.mp4`) that bounding boxes are annotated.

### Extract motion vectors using `mpegflow`
We call MPEG-flow tools from python scripts to generate motion vectors.
You can visualize the motion vectors field of specific movie `movie/corgi/corgi.mp4`
by the `flow.py` script as below:
```
python flow.py movie/corgi
```
Then the result movie is saved as `movie/corgi/corgi_flow.mp4`.
Extracted motion vectors are dumped into `movie/corgi/mpegflow_dump`.

To combine bounding boxes extracted above for showing,
we can use the `annotate.py` script as below:
```
python annotate.py movie/corgi
```
The result movie is saved as `movie/corgi/corgi_annotate.mp4`.
If you specified `--iframes` option,
bounding boxes are visualized for only I-frames.

### Qualitative Evaluation
In prior to quantitative evaluation,
we could qualitatively check results produced by MVint.
Execute the command below to check:
```
python interp.py movie/corgi
```
Result movie is saved as `movie/corgi/corgi_interp.mp4`

There is some enhanced version of script with the Kalman filter:
```
python kalman.py movie/corgi
```
Result movie is saved as `movie/corgi/corgi_kalman.mp4`

### Quantitative Evaluation using MOT16
Finally we will perform quantitative evaluation using MOT16.

Evaluation procedures are scripted to `evaluate_mot16.sh`,
so execute the script as:
```
./evaluate_mot16.sh
```

With the evaluation using MOT16, we reproduced an example result as below:
```
 ********************* Your MOT16 Results *********************
 IDF1  IDP  IDR| Rcll  Prcn   FAR|   GT  MT   PT   ML|    FP    FN   IDs    FM|  MOTA  MOTP MOTAL
 49.5 59.5 42.4| 62.1  87.0  1.92|  517 114  282  121| 10199 41881   725  1308|  52.2  78.6  52.8
```

Tracking performance may differ when using different codec settings,
interpolation settings, and the detector.



License
--------------------------------------------------

Sources are licensed under GPLv3 (inherited from `deep_sort`, `sort` sub-repository).

The example movie (`movie/corgi`, [7])
is redistributed under Creative Commons (CC-BY).



References
--------------------------------------------------

[1] T. Ujiie, M. Hiromoto, and T. Sato, "Interpolation-Based Object Detection Using Motion Vectors for Embedded Real-Time Tracking Systems", In Proceedings of CVPRW, 2018  
[2] J. Redmon and A. Farhadi, "YOLO9000: Better, Faster, Stronger", In Proceedings of CVPR, 2017  
[3] V. Kantorov and I. Laptev, "Efficient feature extraction, encoding and classification for action recognition", In Proceedings of CVPR, 2014  
[4] A. Milan, L. L. Taixé, I. D. Reid, S. Roth and K. Schindler, "MOT16: A Benchmark for Multi-Object Tracking", arxiv:1603.00831  
[5] A. Bewley, Z. Ge, L. Ott, F. Ramos, and B. Upcroft, "Simple Online and Realtime Tracking", In Proceedings of ICIP, 2016  
[6] N. Wojke, A. Bewley and D. Paulus, "Simple Online and Realtime Tracking with a Deep Association Metric", In Proceedings of ICIP, 2017  
[7] iPhotolife101, Corgi of BC Halloween Costume Parade, [URL](https://www.youtube.com/watch?v=blqjlztBYew)  

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
