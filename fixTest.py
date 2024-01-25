import numpy as np

# 读取npy文件
data = np.load('y_train.npy')

print('Array dimensions:', data.shape)

# 遍历第0个维度，保留每个子数组的第一列
new_array = data[:, 0]

