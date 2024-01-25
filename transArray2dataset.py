import numpy as np

result_array = np.load('result.npy')

n, _, _ = result_array.shape

# 创建一个空的三维数组
new_array = np.empty((n, 11, 3))

# 遍历数组的第一维度的所有子数组
for i in range(n):
    # 将子数组转置
    transposed_array = np.transpose(result_array[i])
    # 将转置后的子数组赋值给新的三维数组
    new_array[i] = transposed_array

np.save('dataset.npy', new_array)