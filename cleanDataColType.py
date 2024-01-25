import pandas as pd
import numpy as np

df = pd.read_excel('./data/data3.xlsx', header=None)

df = df.T

pressure_values = df.iloc[:, 0].values
time_series_values = df.iloc[:, 1:].values

segments = []

for i in range(time_series_values.shape[1]):
    time_series = time_series_values[:, i]

    for j in range(len(time_series) - 10):
        segment = np.vstack((time_series[j:j+11], pressure_values[j:j+11]))
        segments.append(segment)

segments_array = np.array(segments)

np.save('segments_ori_3.npy', segments_array)