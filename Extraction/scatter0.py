# 导入所需的库
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor  # 导入 XGBoost 回归模型
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据
file_path = '######.xlsx'  # 文件路径
data = pd.read_excel(file_path)

# 打印数据的列名
print("Data columns:", data.columns)

# 定义特征和目标变量
features = ['不同萃取缓冲液', '解析液', 'Mg2+浓度( mmol/L)', '萃取时间(min)', '解析时间（min）']
target = 'CAP的萃取量值（μg/g）'  # 确保列名与数据一致

# 对分类变量进行 One-Hot 编码
data_encoded = pd.get_dummies(
    data, columns=['不同萃取缓冲液', '解析液'], prefix=['Buffer', 'Solvent'])

# 更新特征列表（One-Hot 编码后的新列名）
features_encoded = [col for col in data_encoded.columns if col != target]

# 提取特征和目标变量
X = data_encoded[features_encoded]  # 特征
y = data_encoded[target]            # 目标变量

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 初始化 XGBoost 回归模型
xgb_regressor = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=1.0,              # 用全部样本
    colsample_bytree=1.0,       # 用全部特征
    reg_alpha=0.0,              # 关闭正则项
    reg_lambda=0.0,
    random_state=42
)

# 训练模型
xgb_regressor.fit(X_train, y_train)
print("\nModel training completed.")

# 在训练集和测试集上进行预测
y_pred_train = xgb_regressor.predict(X_train)
y_pred_test = xgb_regressor.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)  # 计算均方根误差（RMSE）
r2 = r2_score(y_test, y_pred_test)

# 创建 DataFrame 存储实际值和预测值
scatter_data = pd.DataFrame({
    'Actual (Train)': np.concatenate([y_train, np.full(y_test.shape, np.nan)]),
    'Predicted (Train)': np.concatenate([y_pred_train, np.full(y_test.shape, np.nan)]),
    'Actual (Test)': np.concatenate([np.full(y_train.shape, np.nan), y_test]),
    'Predicted (Test)': np.concatenate([np.full(y_train.shape, np.nan), y_pred_test])
})

# 导出数据到 CSV 文件
scatter_data.to_csv('scatter_plot_data_XGBOOST.csv', index=False)
print("数据已成功导出为 scatter_plot_data_XGBOOST.csv！")

# --------------- 绘制散点图 ----------------

# 创建画布
plt.figure(figsize=(6, 6))

# 绘制训练集数据点（红色）
plt.scatter(y_train, y_pred_train, color='#ff0000',
            alpha=0.6, label='Training', s=50)

# 绘制测试集数据点（蓝色）
plt.scatter(y_test, y_pred_test, color='#1f77b4',
            alpha=0.7, label='Test', s=50)

# 绘制参考线 y = x
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
         f'R² = {r2:.4f}',
         fontsize=12, color='black', ha='right')

# 设置刻度字体大小
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 调整布局
plt.tight_layout()

# 显示图像
plt.show()

# 输出模型评估结果
print("\nModel Evaluation on Test Set (XgBoost):")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")
