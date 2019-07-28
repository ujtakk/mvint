#!/bin/bash

DATASET_URL='https://motchallenge.net/data/MOT16.zip'
DEVKIT_URL='https://bitbucket.org/amilan/motchallenge-devkit'

MOT16_TRAIN_NAME=(
"MOT16-02"
"MOT16-04"
"MOT16-05"
"MOT16-09"
"MOT16-10"
"MOT16-11"
"MOT16-13"
)

MOT16_TRAIN_DET=(
"https://drive.google.com/uc?id=1Va__9NWU2ZCmaxIq4oIabi05NYWEOk1K"
"https://drive.google.com/uc?id=1EH7orgDPp7kqRY5OA0hEctcEtQnYq0Ea"
"https://drive.google.com/uc?id=1RCfHJx5ZoUecapbZCsgp0tCEiItvLsd8"
"https://drive.google.com/uc?id=1VLOvn-mbpY0Q1rsMONQZhaEQIGEmyLQL"
"https://drive.google.com/uc?id=1SbMhOgYPvZ84xE8lRtXc7CLXJF86lwf4"
"https://drive.google.com/uc?id=1a4w-HopWJHLFVi4e5wM_CEpv_ZgAVSys"
"https://drive.google.com/uc?id=1EOOPm2-09roynRlIxUCRSxBhChY8PA9D"
)

MOT16_TEST_NAME=(
"MOT16-01"
"MOT16-03"
"MOT16-06"
"MOT16-07"
"MOT16-08"
"MOT16-12"
"MOT16-14"
)

MOT16_TEST_DET=(
"https://drive.google.com/uc?id=1aEzvFHPK-N6hqLXMqhh3i9JJzn7WFUA3"
"https://drive.google.com/uc?id=1h_ktJDBIEXaSBAA-RxKNYnL9e4fp2HPd"
"https://drive.google.com/uc?id=1ilOElwfYZLwQKH57HoYdXfuYhpPibfqF"
"https://drive.google.com/uc?id=1TajzH3GbumKmtYvKBvOtGERFGD0tStwG"
"https://drive.google.com/uc?id=1WB9Mi4RLVPHV4_20sVq7FdoeG5JYQ_J1"
"https://drive.google.com/uc?id=1mksH9GWNT7zmcuq6rlRev8pevZz8Rfsm"
"https://drive.google.com/uc?id=1FVVhn_IpxQ_jkYhc0CUQHSQMm1SMTEBj"
)

# Prepare MOT16 Dataset
curl -O $DATASET_URL
unzip -d MOT16 MOT16.zip

# Prepare MOT16 Devkit
hg clone $DEVKIT_URL

# Compile hex modules
(cd motchallenge-devkit;
  DISPLAY= matlab -r "compile; exit"
)

# Convert MOT16 image sequence to video
source venv/bin/activate
python mot16.py

# Rewrite variables for evaluation
sed -i \
  -e 's|res/MOT16/data/|../result/|g' \
  -e 's|../data/2DMOT16/train/|../MOT16/train/|g' \
  motchallenge-devkit/demo_evalMOT16.m

# Download pre-generated MOT16 detections
# (Check https://github.com/nwojke/deep_sort if dets are not available)
MOT16_TRAIN_PATH=deep_sort/deep_sort_data/resources/detections/MOT16_POI_train
mkdir -p $MOT16_TRAIN_PATH
for i in `seq 0 $((${#MOT16_TRAIN_NAME[*]}-1))`; do
  FILEPATH=${MOT16_TRAIN_PATH}/${MOT16_TRAIN_NAME[$i]}.npy
  curl -o $FILEPATH "${MOT16_TRAIN_DET[$i]}"
done

MOT16_TEST_PATH=deep_sort/deep_sort_data/resources/detections/MOT16_POI_test
mkdir -p $MOT16_TEST_PATH
for i in `seq 0 $((${#MOT16_TEST_NAME[*]}-1))`; do
  FILEPATH=${MOT16_TEST_PATH}/${MOT16_TEST_NAME[$i]}.npy
  curl -o $FILEPATH "${MOT16_TEST_DET[$i]}"
done

