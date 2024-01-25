import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

# 加载数据
print("Loading data...")
data = np.load('dataset.npy')

best_score = float('inf')
best_model = None
best_num_of_samples = None

np.random.seed(0)
np.random.shuffle(data)

data /= 10000

# 遍历 num_of_samples 的值
for num_of_samples in tqdm(range(10000, 20001, 10000)):
    print(f"\nProcessing with num_of_samples = {num_of_samples}")
    temp_data = data[:num_of_samples,:,:]

    # 提取传感器读数和真实气压值
    X = temp_data[:, :10, 0]  # 只提取前10个时间点的传感器读数
    Y = temp_data[:, :, 0]  # 真实气压值

    # 重新组织数据: 使用前10个时间点的数据作为特征，第11个时间点的数据作为目标变量
    X = np.reshape(X, (X.shape[0], -1))  # 这时 `X` 的形状应该是 `(n, 10)`
    Y = Y[:, -1]  # 只需要最后一个时间点的气压值作为目标变量

    # 将数据集分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 创建SVR模型
    svr = SVR(kernel='rbf', C=1e3, gamma=0.1, verbose=True)

    # 训练模型
    svr.fit(X_train, Y_train)

    # 用模型进行预测
    Y_pred = svr.predict(X_test)


    score = np.mean(np.abs(Y_pred - Y_test))
    print(f"Score for num_of_samples = {num_of_samples}: {score}")

    # 如果这个模型的评估标准比之前的模型都要好，那么保存这个模型
    if score < best_score:
        best_score = score
        best_model = svr
        best_num_of_samples = num_of_samples
        print(f"New best model found for num_of_samples = {num_of_samples} with score = {score}")

# 保存最佳模型
print("\nSaving best model...")
joblib.dump(best_model, 'best_svr_model.pkl')
print(f"Best model saved for num_of_samples = {best_num_of_samples} with score = {best_score}")