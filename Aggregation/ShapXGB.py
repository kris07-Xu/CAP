# 导入所需的库
import pandas as pd
import shap
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

# 在测试集上评估 XGBoost 模型
y_pred_xgb = xgb_regressor.predict(X_test_scaled)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse = np.sqrt(mse_xgb)  # 计算均方根误差（RMSE）
r2_xgb = r2_score(y_test, y_pred_xgb)

# 定义贝叶斯优化的目标函数


def optimize_xgb(buffer_te, buffer_tris_hcl, buffer_hepes, buffer_pbs, aptamer, mba, time):
    # 构造输入特征
    input_features = np.array(
        [[buffer_te, buffer_tris_hcl, buffer_hepes, buffer_pbs, aptamer, mba, time]])

    # 对输入特征进行标准化
    input_features_scaled = scaler.transform(input_features)

    # 预测CAP的萃取量值
    extraction_pred = xgb_regressor.predict(input_features_scaled)

    # 贝叶斯优化的目标是最大化CAP的萃取量值
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
    f=optimize_xgb, pbounds=pbounds, random_state=42)

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
optimal_extraction = xgb_regressor.predict(optimal_input_scaled)[0]

# 输出最优条件和预测的CAP的萃取量值
print("\n最优条件和预测的CAP的萃取量值：")
print(f"聚合缓冲液: {selected_buffer}")
print(f"适配体用量（uL）: {optimal_conditions['aptamer']:.2f}")
print(f"MBA用量（mg）: {optimal_conditions['mba']:.2f}")
print(f"聚合时间（h）: {optimal_conditions['time']:.2f}")
print(f"预测的CAP的萃取量值（μg/g）: {optimal_extraction:.2f}")


print("\nModel Evaluation on Test Set (XGBoost):")
print(f"Mean Squared Error (MSE): {mse_xgb:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2_xgb:.4f}")



# 设置全局字体样式（可以加粗部分 SHAP 内部字体）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 14

# 定义英文特征名称
english_feature_names = {
    'Buffer-TE': 'Buffer-TE',
    'Buffer-Tris-HCl': 'Buffer-Tris-HCl',
    'Buffer-Hepes': 'Buffer-Hepes',
    'Buffer_PBS': 'Buffer-PBS',
    '适配体用量（uL）': 'Aptamer Volume (uL)',
    'MBA用量（mg）': 'MBA Amount (mg)',
    '聚合时间（h）': 'Polymerization Time (h)'
}
features_encoded_english = [
    english_feature_names.get(col, col).replace('_', '-') for col in features_encoded
]

# 创建 SHAP 解释器并计算 SHAP 值
explainer = shap.KernelExplainer(xgb_regressor.predict, X_train_scaled)
shap_values = explainer.shap_values(X_test_scaled)

# ---------- SHAP 摘要图 ----------

# 设置图背景透明
plt.figure(facecolor='none')

# 绘制 SHAP 摘要图
shap.summary_plot(shap_values, X_test, feature_names=features_encoded_english, show=False)

# 获取当前的轴对象
ax = plt.gca()

# 修改横纵轴刻度字体大小、加粗
for label in ax.get_xticklabels():
    label.set_fontsize(22)
    label.set_weight('bold')

for label in ax.get_yticklabels():
    label.set_fontsize(22)
    label.set_weight('bold')

# 修改坐标轴标签字体
ax.set_xlabel(ax.get_xlabel(), fontsize=22, fontweight='bold')
ax.set_ylabel(ax.get_ylabel(), fontsize=22, fontweight='bold')

# 获取 colorbar 对象
colorbar = plt.gcf().axes[-1]  # 最后一个轴通常是 colorbar

# 修改 colorbar 上的 tick（high/low 数字）的字体
colorbar.tick_params(labelsize=26, width=1.5)

# 修改 colorbar 的 high 和 low 标签字体大小
for label in colorbar.get_yticklabels():
    label.set_fontsize(26)
    label.set_weight('bold')
    label.set_fontname('Times New Roman')

# 修改 colorbar 的 'Feature value' 标签
for text in plt.gcf().findobj(match=plt.Text):
    if text.get_text() == 'Feature value':
        text.set_fontsize(26)
        text.set_weight('bold')
        text.set_fontname('Times New Roman')

# 设置坐标轴背景透明
plt.gca().set_facecolor('none')

# 保存图片
plt.savefig('shap_summary.png', dpi=300, transparent=True, bbox_inches='tight')

# 显示图形
plt.show()
