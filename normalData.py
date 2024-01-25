import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = np.load('new_dataset.npy')

# 变形为（n*11，3）的形式
reshaped_data = data.reshape(-1, 3)

# 对数组按列进行归一化
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(reshaped_data)

# 重新变形为（n，11，3）的形式
new_data = normalized_data.reshape(data.shape)

np.save('final_dataset.npy', new_data)