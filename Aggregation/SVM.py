# 导入所需的库
import pandas as pd
import numpy as np
from sklearn.svm import SVR  # 导入支持向量回归
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据
file_path = '######.xlsx'  # 文件路径
data = pd.read_excel(file_path)

# 打印数据的列名
print("Data columns:", data.columns)

# 定义特征和目标变量
features = ['不同聚合缓冲液', '适配体用量（uL）', 'MBA用量（mg）', '聚合时间（h）']
target = 'CAP的萃取量值（μg/g）'  # 确保列名与数据一致

# 对分类变量进行 One-Hot 编码
data_encoded = pd.get_dummies(data, columns=['不同聚合缓冲液'], prefix='Buffer')

# 更新特征列表（One-Hot 编码后的新列名）
features_encoded = [col for col in data_encoded.columns if col != target]

# 提取特征和目标变量
X = data_encoded[features_encoded]  # 特征
y = data_encoded[target]            # 目标变量

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化 SVM 回归模型
svm_regressor = SVR(kernel='rbf', C=1.0, gamma='scale',
                    epsilon=0.1)  # 可以根据需要调整参数

# 训练 SVM 模型
svm_regressor.fit(X_train_scaled, y_train)
print("\nSVM Model training completed.")

# 在测试集上评估 SVM 模型
y_pred_svm = svm_regressor.predict(X_test_scaled)
mse_svm = mean_squared_error(y_test, y_pred_svm)
rmse = np.sqrt(mse_svm)  # 计算均方根误差（RMSE）
r2_svm = r2_score(y_test, y_pred_svm)

# 定义贝叶斯优化的目标函数


def optimize_svm(buffer_te, buffer_tris_hcl, buffer_hepes, buffer_pbs, aptamer, mba, time):
    # 构造输入特征
    input_features = np.array(
        [[buffer_te, buffer_tris_hcl, buffer_hepes, buffer_pbs, aptamer, mba, time]])

    # 对输入特征进行标准化
    input_features_scaled = scaler.transform(input_features)

    # 预测CAP的萃取量值
    extraction_pred = svm_regressor.predict(input_features_scaled)

    # 叶斯优化的目标是最大化CAP的萃取量值
    return extraction_pred[0]


# 定义贝叶斯优化的参数范围
pbounds = {
    'buffer_te': (0, 1),  # TE
    'buffer_tris_hcl': (0, 1),  # Tris-HCl
    'buffer_hepes': (0, 1),  # Hepes
    'buffer_pbs': (0, 1),  # PBS
    'aptamer': (55, 105),  # 适配体用量（uL）的范围
    'mba': (14, 24),       # MBA用量（mg）的范围
    'time': (2, 12)        # 聚合时间（h）的范围
}

# 初始化贝叶斯优化器
optimizer = BayesianOptimization(
    f=optimize_svm, pbounds=pbounds, random_state=42)

# 运行贝叶斯优化
optimizer.maximize(init_points=30, n_iter=100)

# 打印最优的条件组合
print("\nOptimal conditions:")
optimal_conditions = optimizer.max['params']
print(optimizer.max)

# 提取最优条件
buffer_conditions = {
    'TE': optimal_conditions['buffer_te'],
    'Tris-HCl': optimal_conditions['buffer_tris_hcl'],
    'Hepes': optimal_conditions['buffer_hepes'],
    'PBS': optimal_conditions['buffer_pbs']
}

# 选择最优的聚合缓冲液
selected_buffer = max(buffer_conditions, key=buffer_conditions.get)
print(f"最优的聚合缓冲液: {selected_buffer}")

# 构造最优条件的输入特征
optimal_input = np.array([
    int(selected_buffer == 'TE'),
    int(selected_buffer == 'Tris-HCl'),
    int(selected_buffer == 'Hepes'),
    int(selected_buffer == 'PBS'),
    optimal_conditions['aptamer'],
    optimal_conditions['mba'],
    optimal_conditions['time']
]).reshape(1, -1)

# 对输入特征进行标准化
optimal_input_scaled = scaler.transform(optimal_input)

# 预测最优条件下的CAP的萃取量值
optimal_extraction = svm_regressor.predict(optimal_input_scaled)[0]

# 输出最优条件和预测的CAP的萃取量值
print("\n最优条件和预测的CAP的萃取量值：")
print(f"聚合缓冲液: {selected_buffer}")
print(f"适配体用量（uL）: {optimal_conditions['aptamer']:.2f}")
print(f"MBA用量（mg）: {optimal_conditions['mba']:.2f}")
print(f"聚合时间（h）: {optimal_conditions['time']:.2f}")
print(f"预测的CAP的萃取量值（μg/g）: {optimal_extraction:.2f}")

print("\nModel Evaluation on Test Set (SVM):")
print(f"Mean Squared Error (MSE): {mse_svm:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2_svm:.4f}")
