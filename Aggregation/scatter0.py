# 导入所需的库
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
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

# 初始化 XGBoost 回归模型
# xgb_regressor = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_regressor = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42
)

# 训练 XGBoost 模型
xgb_regressor.fit(X_train_scaled, y_train)
print("\nXGBoost Model training completed.")

# 在训练集和测试集上进行预测
y_pred_xgb_train = xgb_regressor.predict(X_train_scaled)
y_pred_xgb_test = xgb_regressor.predict(X_test_scaled)

# 评估模型
mse_xgb = mean_squared_error(y_test, y_pred_xgb_test)
rmse = np.sqrt(mse_xgb)  # 计算均方根误差（RMSE）
r2_xgb = r2_score(y_test, y_pred_xgb_test)

# 创建 DataFrame 存储实际值和预测值
scatter_data = pd.DataFrame({
    # 训练集实际值
    'Actual (Train)': np.concatenate([y_train, np.full(y_test.shape, np.nan)]),
    # 训练集预测值
    'Predicted (Train)': np.concatenate([y_pred_xgb_train, np.full(y_test.shape, np.nan)]),
    # 测试集实际值
    'Actual (Test)': np.concatenate([np.full(y_train.shape, np.nan), y_test]),
    # 测试集预测值
    'Predicted (Test)': np.concatenate([np.full(y_train.shape, np.nan), y_pred_xgb_test])
})

# 导出数据到 CSV 文件
scatter_data.to_csv('scatter_plot_XGB.csv', index=False)
print("数据已成功导出为 scatter_plot_XGB.csv！")


# 绘图
plt.figure(figsize=(6, 6))
plt.scatter(y_train, y_pred_xgb_train, color='#ff0000',
            alpha=0.6, label='Training', s=50)
plt.scatter(y_test, y_pred_xgb_test, color='#1f77b4',
            alpha=0.7, label='Test', s=50)

# 参考线 y = x
min_val = min(min(y_train), min(y_test))
max_val = max(max(y_train), max(y_test))
plt.plot([min_val, max_val], [min_val, max_val],
         linestyle='--', color='black', alpha=0.7)

# 设置轴标签
plt.xlabel("Actual CAP Extraction Amount (μg/g)", fontsize=12)
plt.ylabel("Predicted CAP Extraction Amount (μg/g)", fontsize=12)

# 添加图例
plt.legend(loc='upper left', fontsize=10)

# 在图上添加 R² 值
plt.text(max_val, min_val + (max_val - min_val) * 0.05,
         f'R² = {r2_xgb:.4f}', fontsize=12, color='black', ha='right')

# 设置刻度字体大小
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 调整布局
plt.tight_layout()

# 显示图像
plt.show()
