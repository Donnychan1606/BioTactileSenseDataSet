import os
import numpy as np

dir_path = './sortedData'

def process_npy_file(npy_file):
    # 读取numpy数组
    array = np.load(npy_file)

    # 检查数组的维度
    if array.ndim == 3:
        # 如果数组已经是三维的，那么直接跳过
        print(f"Array in file {npy_file} is already 3D. Skipping.")
        return None
    elif array.ndim == 2 and array.shape[1] == 12:
        # 如果数组是二维的，并且有12列，那么将其重塑为（n，1，11）的形状
        array_reshaped = array[:, :11].reshape(array.shape[0], 1, 11)
        # 创建一个新的数组，该数组的每个元素都是对应的第12列的值
        new_row = np.repeat(array[:, 11:], 11, axis=1).reshape(array.shape[0], 1, 11)
        # 在第二维上添加新的一行，形成新的（n，2，11）的数组
        array_final = np.concatenate([array_reshaped, new_row], axis=1)
        # 保存新的数组
        np.save(npy_file, array_final)
        print(f"Processed file {npy_file}.")
    else:
        print(f"Array in file {npy_file} is not 2D with 12 columns. Skipping.")
        return None

npy_files = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith('.npy')]
total_files = len(npy_files)

# 处理每个.npy文件
for i, npy_file in enumerate(npy_files, start=1):
    print(f"Processing file {i} of {total_files}: {npy_file}")
    process_npy_file(npy_file)

print("All files processed.")