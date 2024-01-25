import matplotlib
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time


matlab_data = scipy.io.loadmat('new_dataset_adjusted2.mat')
matrix = matlab_data['new_dataset_adjusted2']


data = [[34764.5, 34226.1, 34178.56, 34383.91, 35261.7, 34577.5, 34205.06, 33923.69, 33824.95, 34244.9, 34363.69, 34863.74,
  35593.38, 35491.15, 34966.76, 34611.44, 34328.13, 34099.02, 33934.49, 34283.76, 34434.56, 34710.22, 34785.5, 35423.15],
 [38193.43, 37223.88, 37002.28, 37178.62, 38669.95, 36702.68, 37795.44, 36323.07, 35881.07, 36823.28, 37125.35, 38042.04,
  38337.66, 37691.65, 36954.74, 37403.77, 37326.25, 36954.36, 37138.16, 37100.51, 36512.84, 36981.88, 36816.15, 37848.61],
 [39951.0, 39700.62, 38979.59, 39141.41, 40560.03, 38572.84, 40456.43, 38128.69, 36888.02, 38542.9, 39020.47, 40017.79,
  40406.02, 39255.48, 38858.17, 39952.62, 40268.48, 39634.91, 39671.66, 38777.93, 37977.75, 39219.57, 38345.87, 39135.68],
 [41324.23, 41753.48, 41004.74, 41058.13, 42141.42, 40698.16, 42041.03, 39965.78, 38134.95, 40274.89, 40564.36, 41559.75,
  42176.53, 40710.57, 40718.76, 41694.67, 41995.31, 41908.37, 41777.71, 40378.31, 39957.11, 41352.19, 40188.5, 40591.1],
 [42552.8, 43150.31, 42778.06, 42649.85, 43515.72, 42775.62, 43338.39, 41356.59, 39557.61, 41865.02, 41801.9, 42788.03,
  43670.51, 42179.51, 42226.4, 43006.19, 43537.3, 43611.79, 43421.92, 41736.78, 41626.11, 42958.23, 41878.09, 42071.76]]

fatigue200 = [
    [34283.36, 33631.23, 33545.14, 33583.81, 33689.19, 33580.13, 33433.61, 33359.92, 33381.89, 33856.31, 33852.05, 34131.82, 34559.46, 33992.41, 33572.36, 33443.1, 33425.46, 33472.86, 33475.05, 33487.17, 33499.05, 33514.87, 33638.14, 34379.5],
    [37157.04, 37341.38, 37343.14, 36603.4, 37337.91, 36875.97, 37314.84, 36995.24, 35699.33, 37303, 39119.58, 37665.16, 38276.35, 36631.09, 35891.89, 37163.9, 37030.87, 38295.32, 36682.76, 37900.1, 37314.48, 37168.52, 37110.12, 37819.78],
    [39670.05, 39516.65, 39840.19, 38108.87, 38893.44, 39019.02, 39879.5, 38431.32, 37766.98, 39613.5, 40416.12, 39653.7, 41166.94, 38149.69, 37158.65, 39656.66, 40443.86, 40186.39, 39058.57, 39884.65, 38898.1, 39314.7, 39550.7, 38919.01],
    [41517.3, 41395.52, 42203.54, 40028.06, 40143.59, 41096.69, 42022.7, 40065.4, 39809.64, 41820.67, 41647.54, 41434.68, 43310.3, 39966.07, 39191.59, 41381.99, 42823.31, 42033.87, 41244.71, 41752.26, 40812.24, 41446.49, 41879.64, 40154.89],
    [42742.47, 42772.85, 43873.14, 41571.09, 41278.86, 43019.34, 43372.31, 41425.04, 41339.27, 43418.37, 42782.03, 42794.63, 44724.96, 41760.93, 40779.91, 42768.61, 44230.67, 43459.91, 43377.25, 43467.74, 42371.32, 43089.72, 43401.96, 41556.82]
]

data = np.array(data)
fatigue200 = np.array(fatigue200)

interpolated = np.empty_like(data)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        interpolated[i, j] = np.interp(0.5, [0, 1], [data[i, j], fatigue200[i, j]])


np.savez('interpolated.npz', *interpolated)

pressure_values = np.array([0, 10, 20, 30, 40])

mapping_functions = []

for i in range(24):
    f = interp1d(interpolated[:, i], pressure_values, kind='cubic')
    mapping_functions.append(f)

mapping_functions_startTime = []

# 记录开始时间
start_time = time.perf_counter()

for i in range(24):
    f = interp1d(data[:, i], pressure_values, kind='cubic')
    mapping_functions_startTime.append(f)


# sensor_reading = 41058
# kpa_value = mapping_functions[0](sensor_reading)
# print(kpa_value)

mapped_matrix = np.empty_like(matrix)

for i in range(24):
    f = mapping_functions[i]
    mapped_matrix[:, i] = f(matrix[:, i])

mapped_matrix_no_cal = np.empty_like(matrix)

for i in range(24):
    f = mapping_functions_startTime[i]
    mapped_matrix_no_cal[:, i] = f(matrix[:, i])

# 记录结束时间
end_time = time.perf_counter()
# 计算并打印运行时间
elapsed_time = end_time - start_time
print(f"The algorithm ran for {elapsed_time} seconds")

######################################## 计算误差值
# 对每个数组的每一列减去该列的最小值
mapped_matrix -= np.min(mapped_matrix, axis=0)
mapped_matrix_no_cal -= np.min(mapped_matrix_no_cal, axis=0)

# 计算两个数组的差值
difference_matrix = mapped_matrix - mapped_matrix_no_cal

print(difference_matrix)

np.savez('stacked_biases_interp.npz', *difference_matrix)
np.savez('mapped_matrix.npz', *mapped_matrix)

for i in range(24):
    plt.plot(mapped_matrix[:, i], label=f'Channel {i+1}')

plt.legend(loc='upper right')

plt.title('Difference Matrix by Channel')
plt.xlabel('Index')
plt.ylabel('Difference')

plt.show()

for i in range(24):
    plt.plot(mapped_matrix_no_cal[:, i], label=f'Channel {i+1}')

plt.legend(loc='upper right')

plt.title('Difference Matrix by Channel')
plt.xlabel('Index')
plt.ylabel('Difference')

plt.show()

for i in range(24):
    plt.plot(difference_matrix[:, i], label=f'Channel {i+1}')

plt.legend(loc='upper right')

plt.title('Difference Matrix by Channel')
plt.xlabel('Index')
plt.ylabel('Difference')

plt.show()

