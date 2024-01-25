import numpy as np
import matplotlib.pyplot as plt

data = np.load('new_dataset.npy')

column_index = 0

column_values = data[:, column_index]

plt.hist(column_values, bins=1000)  

plt.xlim(50000, np.max(column_values))
plt.ylim(0, 100)

plt.title("某一列的值的分布情况")
plt.xlabel("值")
plt.ylabel("频数")

plt.show()