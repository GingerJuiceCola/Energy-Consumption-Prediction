import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. 数据加载与预处理
# 加载数据（替换为你的数据路径）
df = pd.read_csv("D:/课程资料/数据挖掘/GS/数据/finaldata1_no0temp.csv")
# 转换日期格式（仅用于确保数据结构正确，不参与建模）
df["datetime"] = pd.to_datetime(df["datetime"])

# 2. 特征与目标变量划分
# 目标变量：能耗
target_col = "energia"
# 特征：排除日期列和目标列，其余均为特征
feature_cols = [col for col in df.columns if col not in ["datetime", target_col]]
X = df[feature_cols]  # 特征矩阵
y = df[target_col]    # 目标变量

# 3. 数据归一化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. 划分训练集与测试集（8:2分割）
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test set shape: X={X_test.shape}, y={y_test.shape}\n")

# 5. 定义模型评估函数（计算MAE、RMSE、R²）
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # 输出评估结果
    print(f"=== {model_name} Performance ===")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}\n")
    return mae, rmse, r2

# 6. 模型1：线性回归（Regression Analysis）
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)  # 训练模型
y_lr_pred = lr_model.predict(X_test)  # 测试集预测
# 评估线性回归模型
lr_mae, lr_rmse, lr_r2 = evaluate_model(y_test, y_lr_pred, "Linear Regression")

# 7. 模型2：随机森林回归（Random Forest）
rf_model = RandomForestRegressor(
    n_estimators=100,  # 决策树数量（默认100，简单易调）
    random_state=42
)
rf_model.fit(X_train, y_train)  # 训练模型
y_rf_pred = rf_model.predict(X_test)  # 测试集预测
# 评估随机森林模型
rf_mae, rf_rmse, rf_r2 = evaluate_model(y_test, y_rf_pred, "Random Forest Regression")

# 8. 模型性能对比（汇总结果）
print("=== Model Performance Comparison ===")
comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest Regression"],
    "MAE": [lr_mae, rf_mae],
    "RMSE": [lr_rmse, rf_rmse],
    "R² Score": [lr_r2, rf_r2]
})
print(comparison.round(4))
