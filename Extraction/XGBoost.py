# 导入所需的库
import pandas as pd
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

# 在测试集上评估模型
y_pred = xgb_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # 计算均方根误差（RMSE）
r2 = r2_score(y_test, y_pred)


# 定义贝叶斯优化的目标函数
def optimize_xgb(buffer_te, buffer_tris_hcl, buffer_hepes, buffer_pbs,
                 solvent_methanol, solvent_5pct_water_methanol, solvent_10pct_water_methanol, solvent_acetonitrile, solvent_10pct_water_acetonitrile,
                 mg_concentration, extraction_time, desorption_time):
    # 构造输入特征
    input_features = np.array([
        [buffer_te, buffer_tris_hcl, buffer_hepes, buffer_pbs,
         solvent_methanol, solvent_5pct_water_methanol, solvent_10pct_water_methanol, solvent_acetonitrile, solvent_10pct_water_acetonitrile,
         mg_concentration, extraction_time, desorption_time]
    ])

    # 预测CAP的萃取量值
    extraction_pred = xgb_regressor.predict(input_features)

    # 贝叶斯优化的目标是最大化CAP的萃取量值
    return extraction_pred[0]


# 定义贝叶斯优化的参数范围
pbounds = {
    'buffer_te': (0, 1),  # TE
    'buffer_tris_hcl': (0, 1),  # Tris-HCl
    'buffer_hepes': (0, 1),  # Hepes
    'buffer_pbs': (0, 1),  # PBS
    'solvent_methanol': (0, 1),  # 甲醇
    'solvent_5pct_water_methanol': (0, 1),  # 5%水-甲醇
    'solvent_10pct_water_methanol': (0, 1),  # 10%水-甲醇
    'solvent_acetonitrile': (0, 1),  # 乙腈
    'solvent_10pct_water_acetonitrile': (0, 1),  # 10%水-乙腈
    'mg_concentration': (0, 25),  # Mg2+浓度范围
    'extraction_time': (15, 75),  # 萃取时间范围
    'desorption_time': (10, 35)  # 解析时间范围
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

solvent_conditions = {
    '甲醇': optimal_conditions['solvent_methanol'],
    '5%水-甲醇': optimal_conditions['solvent_5pct_water_methanol'],
    '10%水-甲醇': optimal_conditions['solvent_10pct_water_methanol'],
    '乙腈': optimal_conditions['solvent_acetonitrile'],
    '10%水-乙腈': optimal_conditions['solvent_10pct_water_acetonitrile']
}

# 选择最优的萃取缓冲液和解析液
selected_buffer = max(buffer_conditions, key=buffer_conditions.get)
selected_solvent = max(solvent_conditions, key=solvent_conditions.get)

# 构造最优条件的输入特征
optimal_input = np.array([
    int(selected_buffer == 'TE'),
    int(selected_buffer == 'Tris-HCl'),
    int(selected_buffer == 'Hepes'),
    int(selected_buffer == 'PBS'),
    int(selected_solvent == '甲醇'),
    int(selected_solvent == '5%水-甲醇'),
    int(selected_solvent == '10%水-甲醇'),
    int(selected_solvent == '乙腈'),
    int(selected_solvent == '10%水-乙腈'),
    optimal_conditions['mg_concentration'],
    optimal_conditions['extraction_time'],
    optimal_conditions['desorption_time']
]).reshape(1, -1)

# 预测最优条件下的CAP的萃取量值
optimal_extraction = xgb_regressor.predict(optimal_input)

# 输出最优条件和预测的CAP的萃取量值
print("\n最优条件和预测的CAP的萃取量值：")
print(f"萃取缓冲液: {selected_buffer}")
print(f"解析液: {selected_solvent}")
print(f"Mg2+浓度( mmol/L): {optimal_conditions['mg_concentration']:.2f}")
print(f"萃取时间(min): {optimal_conditions['extraction_time']:.2f}")
print(f"解析时间（min）: {optimal_conditions['desorption_time']:.2f}")
print(f"预测的CAP的萃取量值（μg/g）: {optimal_extraction[0]:.2f}")

print("\nModel Evaluation on Test Set(XGBoost):")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")
