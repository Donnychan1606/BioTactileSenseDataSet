import numpy as np

data = np.load('dataset.npy')

new_data = []

# 遍历第0个维度的子数组
for sub_array in data:
    # 判断第一列是否有大于58000的数字
    if np.any(sub_array[:, 0] > 58000):
        continue  # 跳过包含大于58000的子数组

    # 判断第二列是否有大于100的数字
    if np.any(sub_array[:, 1] > 100):
        sub_array[:, [1, 2]] = sub_array[:, [2, 1]]

    # 判断子数组的任意一行的三列是否全是同一个数字
    if np.any(np.all(sub_array[:, 0] == sub_array[:, 1]) and np.all(sub_array[:, 1] == sub_array[:, 2])):
        print("子数组中有一行的三列全是同一个数字:", sub_array)
        continue  # 跳过处理该子数组

    # 将符合条件的子数组保存到新数组中
    new_data.append(sub_array)

new_data = np.array(new_data)

np.save('new_dataset.npy', new_data)