#!/bin/bash
# wget http://pjreddie.com/media/files/yolo9000.weights

MOVIE=$1
if [[ $MOVIE = "" ]]; then
  MOVIE=/home/work/takau/6.image/flow_detect/movie/bird
fi

DARKNET=./darknet
DATA=cfg/combine9k.data
NETWORK=cfg/yolo9000.cfg
WEIGHTS=yolo9000.weights
SOURCE=`basename $MOVIE`.mp4
TARGET=out_`basename $MOVIE`.mp4

rm -rf bbox_dump
mkdir -p bbox_dump

$DARKNET detector demo $DATA $NETWORK $WEIGHTS $MOVIE/$SOURCE

ffmpeg -y -i out.avi $MOVIE/$TARGET
rm -f out.avi

mkdir -p $MOVIE
rm -rf $MOVIE/bbox_dump
mv -f bbox_dump $MOVIE
