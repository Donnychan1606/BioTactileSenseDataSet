import random
import pandas as pd
import numpy as np

df = pd.read_excel('./data/data1.xlsx')

sequences = np.empty((0, 11))

for index, row in df.iterrows():
    time_series = row.values
    if len(time_series) < 11:
        continue
    for i in range(len(time_series) - 10):
        segment = time_series[i:i+11]
        sequences = np.vstack([sequences, segment])


new_column = np.full((sequences.shape[0], 1), round(random.uniform(4, 6), 2))
#
sequences = np.hstack((sequences, new_column))
#
np.save('sequences3.npy', sequences)