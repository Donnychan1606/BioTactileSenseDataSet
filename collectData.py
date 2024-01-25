import os
import numpy as np
from tqdm import tqdm

dir_path = './sortedData'

def process_subdir(path):
    combined_array = None
    npy_files = [os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files if file.endswith('.npy')]
    for npy_file in tqdm(npy_files, desc='Processing files', unit='files'):
        array = np.load(npy_file)
        if len(array.shape) == 2 and array.shape[1] == 11:
            array = array.reshape((array.shape[0], 1, array.shape[1]))
            new_row = np.zeros((array.shape[0], 1, array.shape[2]))
            array = np.concatenate((array, new_row), axis=1)
        if combined_array is None:
            combined_array = array
        else:
            combined_array = np.concatenate((combined_array, array), axis=0)
    return combined_array

# 遍历所有子文件夹
for subdir in os.listdir(dir_path):
    subdir_path = os.path.join(dir_path, subdir)
    if os.path.isdir(subdir_path):
        combined_array = process_subdir(subdir_path)
        np.save(f'combined_array_{subdir}.npy', combined_array)
