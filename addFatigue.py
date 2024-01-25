# import os
# import numpy as np
# import re
#
# dir_path = './sortedData'
#
# def process_npy_file(npy_file):
#     x = float(re.search(r'combined_array_(\d+)', npy_file).group(1))
#
#     array = np.load(npy_file)
#     if array.ndim == 3 and array.shape[1] == 2:
#         new_row = np.full((array.shape[0], 1, array.shape[2]), x)
#         array_final = np.concatenate([array, new_row], axis=1)
#         np.save(npy_file.replace('.npy', '_addFatigue.npy'), array_final)
#         print(f"Processed file {npy_file}.")
#     else:
#         print(f"Array in file {npy_file} is not 3D with second dimension 2. Skipping.")
#         return None
#
# npy_files = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.startswith('combined_array_') and file.endswith('.npy')]
# total_files = len(npy_files)
#
# for i, npy_file in enumerate(npy_files, start=1):
#     print(f"Processing file {i} of {total_files}: {npy_file}")
#     process_npy_file(npy_file)
#
# print("All files processed.")

import os
import numpy as np

dir_path = './sortedData'

def process_npy_file(npy_file):
    array = np.load(npy_file)
    print(f"Array in file {npy_file} has shape: {array.shape}")
    return array

npy_files = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith('.npy')]
total_files = len(npy_files)

for i, npy_file in enumerate(npy_files, start=1):
    print(f"Processing file {i} of {total_files}: {npy_file}")
    array = process_npy_file(npy_file)
    array_var_name = 'array_' + os.path.basename(npy_file).replace('.npy', '')
    globals()[array_var_name] = array

print("All files processed.")


