import pandas as pd

# 1. 加载数据
df = pd.read_csv("finaldata1_nomissing.csv")

# 2. 需检查的3个温度字段
check_fields = ["ESP32_temp", "WORKSTATION_CPU_TEMP", "WORKSTATION_GPU_TEMP"]

# 3. 过滤：仅保留3个字段均不为0的记录
for field in check_fields:
    if field in df.columns:
        df[field] = pd.to_numeric(df[field], errors="coerce").fillna(0)

# 核心过滤条件：所有检查字段都≠0
df_filtered = df[~(df[check_fields] == 0).any(axis=1)]

# 4. 保存新文件
df_filtered.to_csv("finaldata1_no0temp.csv", index=False)
