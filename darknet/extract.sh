#!/bin/bash
# wget http://pjreddie.com/media/files/yolo9000.weights

TOPLEVEL=`git rev-parse --show-toplevel`
MOVIE=$TOPLEVEL/$1
if [[ $MOVIE = "" ]]; then
  MOVIE=$TOPLEVEL/movie/bird
fi

DARKNET=./darknet
DATA=cfg/combine9k.data
NETWORK=cfg/yolo9000.cfg
WEIGHTS=yolo9000.weights
SOURCE=`basename $MOVIE`.mp4
TARGET=out_`basename $MOVIE`.mp4

(cd $TOPLEVEL/darknet;
  rm -rf bbox_dump
  mkdir -p bbox_dump

  $DARKNET detector demo $DATA $NETWORK $WEIGHTS $MOVIE/$SOURCE

  ffmpeg -y -i out.avi $MOVIE/$TARGET
  rm -f out.avi

  rm -rf $MOVIE/bbox_dump
  mkdir -p $MOVIE/bbox_dump
  mv -f bbox_dump $MOVIE
)
