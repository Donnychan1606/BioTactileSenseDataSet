import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time


class SensorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SensorNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))
        fc_out = self.fc(lstm_out[:, -1, :])  # 只使用最后一个时间步的输出
        return fc_out


hidden_size = 20
num_layers = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = SensorNetwork(input_size=3, hidden_size=hidden_size, output_size=1, num_layers=num_layers)
model.load_state_dict(torch.load(f'best_model_{hidden_size}_{num_layers}.pt'))
model.to(device)
model.eval()

# # 加载测试数据
# x_test = np.load('X_test.npy')
# x_test_tensor = torch.from_numpy(x_test).float().to(device)
#
# # 进行预测
# with torch.no_grad():
#     prediction = model(x_test_tensor[:1000])
#
# # 将预测结果转换为numpy数组，并保存
# prediction_np = prediction.cpu().numpy()
# # np.save('predictions_test.npy', prediction_np)
#
# # 生成预测结果的折线图
# plt.plot(prediction_np)
# plt.title('Predictions for Test Data')
# plt.show()

# # 加载数据
# 加载 .npz 文件
data = np.load('result_matrices.npz')
pulse_data = np.load('matrix.npz')

# 获取所有的键
keys = sorted(data.keys())
pulse_keys = sorted(pulse_data.keys())

result_matrices = [data[key] for key in keys]
pulse_matrices = [pulse_data[pulse_keys] for pulse_keys in pulse_keys]
pulse_matrices = np.array(pulse_matrices)
pulse_matrices = pulse_matrices[10:,:]

data.close()

scaler_for_second_column = []

new_result_matrices = []

biases = []

for arr in result_matrices:
    # 创建一个新的归一化器
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 变形为（n*11，3）的形式
    reshaped_data = arr.reshape(-1, 3)

    # 对数组按列进行归一化
    normalized_data = scaler.fit_transform(reshaped_data)

    # 保存第二列的归一化器
    scaler_for_second_column.append(MinMaxScaler().fit(reshaped_data[:, [1]]))

    # 重新变形为（n，11，3）的形式
    new_data = normalized_data.reshape(arr.shape)

    # 将处理后的数组添加到新的列表中
    new_result_matrices.append(new_data)

predictions = []

# 记录开始时间
start_time = time.perf_counter()

# 对于每个通道，进行预测
for i, channel_matrix in enumerate(new_result_matrices):
    channel_matrix = torch.from_numpy(channel_matrix).float().to(device)

    # 进行预测
    with torch.no_grad():
        prediction = model(channel_matrix)

    # 计算偏差，为了将预测结果和原始数据对齐，我们需要将预测结果反归一化
    pred_np = prediction.cpu().numpy().reshape(-1, 3)
    pred_np = scaler_for_second_column[i].inverse_transform(pred_np)[:, 1]
    actual_np = channel_matrix.cpu().numpy().reshape(-1, 3)[:, 1]
    bias = pred_np[:-1] - actual_np[1:]
    biases.append(bias)

    # 将预测结果保存
    predictions.append(prediction.cpu().numpy())

# 记录结束时间
end_time = time.perf_counter()
# 计算并打印运行时间
elapsed_time = end_time - start_time
print(f"The algorithm ran for {elapsed_time} seconds")

# 将预测结果合并为一维数组
stacked_predictions = np.hstack(predictions)

# 将偏差结果合并为一维数组
stacked_biases = np.hstack(biases)
np.savez('stacked_biases.npz', *stacked_biases)



inv_predictions = [scaler_for_second_column[i].inverse_transform(predictions[i]) for i in range(len(predictions))]


######################################################################
# 看差值
stacked_predictions = np.hstack(inv_predictions)
# 使用 reshape 方法将结果数组的形状变为 (n, 24)
final_array = stacked_predictions.reshape(-1, 24)
sru_diff = pulse_matrices - final_array[:-1,:]
sru_diff = scaler_for_second_column[i].fit_transform(sru_diff)

######################################################################


np.savez('predictions.npz', *predictions)

for i, prediction in enumerate(inv_predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(prediction, label=f'Channel {i + 1}')
    plt.legend()
    plt.show()

###################################################  和 Ground Truth 进行对比
data = np.load('mapped_matrix.npz')
mapped_matrix = [data[f'arr_{i}'] for i in range(len(data.files))]
mapped_matrix = np.array(mapped_matrix)

mapped_matrix = mapped_matrix[10:-1,:]

stacked_array = np.hstack(inv_predictions)
final_array = stacked_array.reshape(-1, 24)

final_array -= np.min(final_array, axis=0)
final_array = final_array[:-2, :]

# 获取 mapped_matrix 的最小值和最大值
min_val = np.min(mapped_matrix)
max_val = np.max(mapped_matrix)

# 创建一个空的数组，用于存储缩放后的结果
scaled_final_array = np.empty(final_array.shape)

# 遍历 final_array 的每一列
for i in range(final_array.shape[1]):
    # 获取当前列
    column = final_array[:, i]

    # 执行最小-最大缩放
    column_std = (column - column.min()) / (column.max() - column.min())
    scaled_column = column_std * (max_val - min_val) + min_val

    # 将缩放后的列保存到结果数组中
    scaled_final_array[:, i] = scaled_column

# 计算两个数组的差值
diff_matrix = np.zeros_like(scaled_final_array)

# 迭代 scaled_final_array 的每个元素
for i in range(scaled_final_array.shape[0]):
    for j in range(scaled_final_array.shape[1]):
        start = max(0, i - 200)
        end = min(mapped_matrix.shape[0], i + 200)

        min_diff_index = np.argmin(np.abs(mapped_matrix[start:end, j] - scaled_final_array[i, j]))

        # 计算这两个元素之间的差值
        diff = mapped_matrix[start + min_diff_index, j] - scaled_final_array[i, j]

        diff_matrix[i, j] = diff

print(diff_matrix)

# diff_matrix = mapped_matrix - scaled_final_array

for i in range(24):
    plt.plot(sru_diff[:, i], label=f'Channel {i+1}')

plt.legend(loc='upper right')

plt.title('Difference Matrix by Channel')
plt.xlabel('Index')
plt.ylabel('Difference')

plt.show()
