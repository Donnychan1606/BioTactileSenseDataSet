import numpy as np
import joblib
from matplotlib import pyplot as plt
from pykalman import KalmanFilter
import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0
        self.paused = False

    def start(self):
        if not self.paused:
            self.start_time = time.perf_counter()
        else:
            self.paused = False
            self.start_time = time.perf_counter() - self.elapsed_time

    def pause(self):
        if not self.paused:
            self.elapsed_time = time.perf_counter() - self.start_time
            self.paused = True

    def resume(self):
        if self.paused:
            self.paused = False
            self.start_time = time.perf_counter() - self.elapsed_time

    def stop(self):
        if not self.paused:
            self.elapsed_time = time.perf_counter() - self.start_time
            self.start_time = None
        return self.elapsed_time

    def reset(self):
        self.start_time = None
        self.elapsed_time = 0
        self.paused = False

# 使用Timer类
timer = Timer()

# 加载模型
print("Loading model...")
model = joblib.load('best_svr_model.pkl')

# 加载数据
print("Loading data...")
data = np.load('result_matrices.npz')
pulse_data = np.load('matrix.npz')


keys = sorted(data.keys())
pulse_keys = sorted(pulse_data.keys())

result_matrices = [data[key] for key in keys]
pulse_matrices = [pulse_data[pulse_key] for pulse_key in pulse_keys]
pulse_matrices = np.array(pulse_matrices)
pulse_matrices = pulse_matrices[10:, :]

data = np.load('mapped_matrix.npz')
mapped_matrix = [data[f'arr_{i}'] for i in range(len(data.files))]
mapped_matrix = np.array(mapped_matrix)
mapped_matrix = mapped_matrix[10:-1,:]

data.close()

# 初始化卡尔曼滤波器
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

predictions = []

biases = []
# 开始计时
timer.start()

for i, channel_matrix in enumerate(result_matrices):
    # 暂停计时
    timer.pause()

    # 将数据变形为（n*11，3）的形式
    reshaped_channel_matrix = channel_matrix.reshape(-1, 3)

    # 取出第二列作为特征
    X = reshaped_channel_matrix[:, 1]

    # 将 X 变形为 (473, 10)
    X = X.reshape(-1, 10)

    X /= 10000

    # 使用滑动窗口进行预测
    prediction = []
    bias = []


    for j in range(X.shape[0]):
        window_data = X[j].reshape(1, -1)
        # 继续计时
        timer.resume()
        pred = model.predict(window_data)  # 取出预测结果中的单个值
        # 暂停计时
        timer.pause()

        # 将预测值添加到window_data后面
        new_data = np.append(window_data, pred)

        # 使用卡尔曼滤波器
        filtered_data, _ = kf.filter(new_data)
        pred_filtered = filtered_data[-1]  # 取出滤波后的预测值

        pred_filtered *= 10000
        prediction.append(pred_filtered)

        # 如果存在下一个window_data，计算偏差
        if j + 1 < X.shape[0]:
            next_window_data = X[j + 1].reshape(1, -1)
            bias.append(pred_filtered - next_window_data[0][0] * 10000)

    # 保存预测结果
    predictions.append(prediction)

    # 保存偏差结果
    biases.append(bias)

# 记录结束时间
timer.stop()

# 将预测结果合并为一维数组
stacked_predictions = np.hstack(predictions)

# 计算并打印运行时间
print(f"Total elapsed time: {timer.elapsed_time} seconds.")

# 将偏差结果合并为一维数组
stacked_biases = np.hstack(biases)
np.savez('stacked_biases.npz', *stacked_biases)

# 使用 reshape 方法将结果数组的形状变为 (n, 24)
final_array = stacked_predictions.reshape(-1, 24)
final_array = final_array[:-1, :]

# 计算差值
svr_diff = pulse_matrices - final_array

# # 计算每个通道的最小值
# channel_mins = np.min(svr_diff, axis=0)
#
# # 从每个通道中减去最小值
# svr_diff = svr_diff - channel_mins

for i in range(24):
    plt.plot(stacked_biases[:, i], label=f'Channel {i+1}')


plt.legend(loc='upper right')

plt.title('Difference Matrix by Channel')
plt.xlabel('Index')
plt.ylabel('Difference')

plt.show()