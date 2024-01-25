# 加载数据
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from sklearn.gaussian_process import GaussianProcessRegressor

print("Loading data...")
stacked_biases = np.load('stacked_biases.npz')
keys = sorted(stacked_biases.keys())
stacked_biases = [stacked_biases[key] for key in keys]
# 将偏差列表转换为 NumPy 数组
biases_array = np.array(stacked_biases)
# 重新整形为（472，24）的数组
stacked_biases = biases_array.reshape(472, 24)
stacked_biases = stacked_biases[:-1, :]

data = np.load('mapped_matrix.npz')
mapped_matrix = [data[f'arr_{i}'] for i in range(len(data.files))]
mapped_matrix = np.array(mapped_matrix)
mapped_matrix = mapped_matrix[10:-1,:]

data.close()

raw = [[34764.5, 34226.1, 34178.56, 34383.91, 35261.7, 34577.5, 34205.06, 33923.69, 33824.95, 34244.9, 34363.69, 34863.74,
  35593.38, 35491.15, 34966.76, 34611.44, 34328.13, 34099.02, 33934.49, 34283.76, 34434.56, 34710.22, 34785.5, 35423.15],
 [38193.43, 37223.88, 37002.28, 37178.62, 38669.95, 36702.68, 37795.44, 36323.07, 35881.07, 36823.28, 37125.35, 38042.04,
  38337.66, 37691.65, 36954.74, 37403.77, 37326.25, 36954.36, 37138.16, 37100.51, 36512.84, 36981.88, 36816.15, 37848.61],
 [39951.0, 39700.62, 38979.59, 39141.41, 40560.03, 38572.84, 40456.43, 38128.69, 36888.02, 38542.9, 39020.47, 40017.79,
  40406.02, 39255.48, 38858.17, 39952.62, 40268.48, 39634.91, 39671.66, 38777.93, 37977.75, 39219.57, 38345.87, 39135.68],
 [41324.23, 41753.48, 41004.74, 41058.13, 42141.42, 40698.16, 42041.03, 39965.78, 38134.95, 40274.89, 40564.36, 41559.75,
  42176.53, 40710.57, 40718.76, 41694.67, 41995.31, 41908.37, 41777.71, 40378.31, 39957.11, 41352.19, 40188.5, 40591.1],
 [42552.8, 43150.31, 42778.06, 42649.85, 43515.72, 42775.62, 43338.39, 41356.59, 39557.61, 41865.02, 41801.9, 42788.03,
  43670.51, 42179.51, 42226.4, 43006.19, 43537.3, 43611.79, 43421.92, 41736.78, 41626.11, 42958.23, 41878.09, 42071.76]]


data = np.array(raw)

# 压力值，这是一个 5*1 的向量，单位为 kPa
pressure_values = np.array([0, 10, 20, 30, 40])


mapping_functions_startTime = []

# 检查数据是否包含NaN或Inf
if np.any(np.isnan(data)) or np.any(np.isinf(data)):
    raise ValueError("Data contains NaN or Inf values.")

for i in range(24):
    # 注意：这里的x是压力值，y是传感器的输出值
    f = interp1d(pressure_values, data[:, i], kind='cubic', fill_value="extrapolate")
    mapping_functions_startTime.append(f)

# 创建一个新的映射函数列表
new_mapping_functions = []

# 用于存储原始和新映射函数的预测值，用于可视化
y_pred_old = []
y_pred_new = []

# 设置原始数据点和新数据点的权重
weight_old = 1000
weight_new = 1

for i in range(24):
    # 获取这个通道的偏差值
    biases = stacked_biases[:, i]

    # 创建一个原始映射函数的逆函数来找到对应的x值
    # f_inv = interp1d(mapping_functions_startTime[i].y, mapping_functions_startTime[i].x, kind='cubic', fill_value="extrapolate")

    # 使用逆函数将mapped_matrix的值映射回传感器读数值
    sensor_outputs = mapping_functions_startTime[i](mapped_matrix[:, i])

    # 将偏差值添加到新的传感器读数值中
    new_sensor_outputs = sensor_outputs + biases

    # 新的压力值应该是mapped_matrix的列值
    new_pressure_values = mapped_matrix[:, i]

    # 将新的数据点添加到原始数据点中
    combined_pressure_values = np.concatenate([pressure_values, new_pressure_values])
    combined_sensor_outputs = np.concatenate([data[:, i], new_sensor_outputs])

    # 设置权重
    weights = np.concatenate([np.repeat(weight_old, len(pressure_values)), np.repeat(weight_new, len(new_pressure_values))])

    # 检查权重是否都是正数
    if np.any(weights <= 0):
        raise ValueError("Weights must be positive.")

    # 对数据进行排序，并同时调整权重数组
    sort_index = np.argsort(combined_pressure_values)
    combined_pressure_values = combined_pressure_values[sort_index]
    combined_sensor_outputs = combined_sensor_outputs[sort_index]
    weights = weights[sort_index]


    df = pd.DataFrame({
        'x': combined_pressure_values,
        'y': combined_sensor_outputs,
        'w': weights
    })

    # 删除对于每个x值的重复行，只保留第一个
    df_unique = df.drop_duplicates(subset='x', keep='first')


    combined_pressure_values = df_unique['x'].values
    combined_sensor_outputs = df_unique['y'].values
    weights = df_unique['w'].values

    # 创建一个新的映射函数
    # f_new = UnivariateSpline(combined_pressure_values, combined_sensor_outputs, w=weights, s=1)
    # f_new = interp1d(combined_pressure_values, combined_sensor_outputs, kind='cubic', fill_value="extrapolate")
    # Apply Savitzky-Golay filter
    smoothed_sensor_outputs = savgol_filter(combined_sensor_outputs, window_length=5, polyorder=3)
    # f_new = UnivariateSpline(combined_pressure_values, smoothed_sensor_outputs, w=weights, s=1)

    # 计算权重
    weights = np.sqrt(weights)

    # 使用权重进行多项式拟合
    coef = np.polyfit(combined_pressure_values, combined_sensor_outputs, deg=3, w=weights)
    f_new = np.poly1d(coef)

    new_mapping_functions.append(f_new)

    # 使用映射函数生成预测值，用于可视化
    x = np.linspace(0, 40, 1000)
    y_pred_old.append(mapping_functions_startTime[i](x))
    y_pred_new.append(f_new(x))

# 可视化原始和新映射函数
for i in range(24):
    plt.figure(figsize=(10, 8))

    # 绘制原始映射函数
    plt.plot(x, y_pred_old[i], '--', label='Original mapping function')

    # 绘制新映射函数
    plt.plot(x, y_pred_new[i], label='New mapping function')

    plt.xlabel('Pressure (kPa)')
    plt.ylabel('Sensor output')
    plt.title(f'Channel {i+1}')
    plt.legend()

    plt.tight_layout()
    plt.show()