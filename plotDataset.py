import numpy as np
import matplotlib.pyplot as plt

new_data = np.load('new_dataset.npy')

# 定义每个图形中要绘制的数字数量
num_digits_per_plot = 200

# 计算需要创建的图形数量
num_plots = len(new_data) // num_digits_per_plot + 1

# 遍历要创建的图形数量
for i in range(300000,300003):
    start_idx = i * num_digits_per_plot
    end_idx = min((i + 1) * num_digits_per_plot, len(new_data))

    fig, axs = plt.subplots(3)

    # 遍历当前图形要绘制的数字范围
    for sub_array in new_data[start_idx:end_idx]:
        # 绘制第一列的数字
        axs[0].plot(sub_array[:, 0])

        # 绘制第二列的数字
        axs[1].plot(sub_array[:, 1])

        # 绘制第三列的数字
        axs[2].plot(sub_array[:, 2])

    # 设置子图标题
    axs[0].set_title("第一列的数字")
    axs[1].set_title("第二列的数字")
    axs[2].set_title("第三列的数字")

    plt.tight_layout()

    plt.show()