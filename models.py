import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# 图表样式基础设置
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

# 1. 数据加载与预处理
# 加载数据
df = pd.read_csv("data/finaldata1_no0temp.csv")
# 日期格式转换
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
# 划分特征与目标（能耗列：energia）
target_col = "energia"
feature_cols = [col for col in df.columns 
                if col not in ["datetime", target_col] 
                and pd.api.types.is_numeric_dtype(df[col])]
# 确保对齐
X = df[feature_cols].dropna()
y = df.loc[X.index, target_col]
# 打印数据基础信息
print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
print(f"Data shape: {X.shape[0]} rows × {X.shape[1]} cols\n")
# 数据归一化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# 划分训练集/测试集（8:2，固定随机种子）
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows\n")

# 2. 模型评估函数（复用性强）
def evaluate(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{model_name}]")
    print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}\n")
    return mae, rmse, r2

# 3. 三模型训练与评估
# 3.1 线性回归
lr = LinearRegression().fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_mae, lr_rmse, lr_r2 = evaluate("Linear Regression", y_test, lr_pred)

# 3.2 随机森林
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mae, rf_rmse, rf_r2 = evaluate("Random Forest", y_test, rf_pred)

# 3.3 XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
xgb_params = {
    "objective": "reg:squarederror", "max_depth": 5, 
    "learning_rate": 0.1, "reg_alpha": 0.1, "verbosity": 0
}
# 训练
xgb_model = xgb.train(
    params=xgb_params, dtrain=dtrain, num_boost_round=100,
    evals=[(dtest, "test")], early_stopping_rounds=10, verbose_eval=False
)
xgb_pred = xgb_model.predict(dtest)
xgb_mae, xgb_rmse, xgb_r2 = evaluate("XGBoost", y_test, xgb_pred)

# 4. 模型性能汇总对比
# 4.1 表格对比
comp_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost"],
    "MAE": [lr_mae, rf_mae, xgb_mae],
    "RMSE": [lr_rmse, rf_rmse, xgb_rmse],
    "R2": [lr_r2, rf_r2, xgb_r2]
}).round(4).sort_values("R2", ascending=False)

print("=== Model Performance Summary ===")
print(comp_df)

# 4.2 可视化对比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
models = ["Linear Reg", "Random Forest", "XGBoost"]
# 子图1：MAE & RMSE
x = np.arange(3)
width = 0.35
ax1.bar(x-width/2, [lr_mae, rf_mae, xgb_mae], width, label="MAE", color="#1f77b4")
ax1.bar(x+width/2, [lr_rmse, rf_rmse, xgb_rmse], width, label="RMSE", color="#ff7f0e")
ax1.set_xlabel("Model")
ax1.set_ylabel("Error")
ax1.set_title("MAE & RMSE Comparison")
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()
# 子图2：R2
colors = ["#1f77b4", "#2ca02c", "#d62728"]
bars = ax2.bar(models, [lr_r2, rf_r2, xgb_r2], color=colors)
ax2.set_xlabel("Model")
ax2.set_ylabel("R2 Score")
ax2.set_title("R2 Comparison")
ax2.set_ylim(0, 1)
# 添加数值标签
for bar, val in zip(bars, [lr_r2, rf_r2, xgb_r2]):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, 
             f"{val:.4f}", ha="center")

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300)
plt.show()
