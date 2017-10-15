#!/usr/bin/env python3

import pandas as pd

pd.options.display.width = 999
pd.options.display.max_rows = 640
pd.options.display.max_columns = 20

df = pd.read_csv("MOT16_results.csv").drop("Private Detector", axis=1).query("Sequence == 'MOT16'")

print(df.sort_values("MOTA", ascending=False), file=open("result.txt", "w"))
