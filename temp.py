#!/usr/bin/env python3

"""DEPRECATED
"""

def annotate_bb(db):

def youtube_bb_csv(csv_path):
    ytb_src = pd.read_csv(csv_path, names=("youtube_id", "timestamp_ms",
                                           "class_id", "class_name",
                                           "object_id", "object_presence",
                                           "xmin", "xmax", "ymin", "ymax"))
    return ytb_src

def main():
    csv_path = "/home/work/vision/youtube-bb/yt_bb_detection_validation.csv"
    db = youtube_bb_csv(csv_path)

if __name__ == "__main__":
    main()
