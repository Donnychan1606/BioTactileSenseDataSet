import os
import numpy as np

dir_path = './sortedData'

# 获取该文件夹内的所有.npy文件
npy_files = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith('.npy')]
total_files = len(npy_files)

all_arrays = []

# 处理每个.npy文件
for i, npy_file in enumerate(npy_files, start=1):
    print(f"Processing file {i} of {total_files}: {npy_file}")
    array = np.load(npy_file)
    for sub_array in array:
        if not np.isnan(sub_array).any():
            all_arrays.append(sub_array)

# 将列表转换为numpy数组
result = np.array(all_arrays)
assert result.shape[1:] == (3, 11), "Array shape mismatch!"

np.save('result.npy', result)

print("All files processed.")