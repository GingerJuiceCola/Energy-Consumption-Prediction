import pandas as pd
import numpy as np

# 加载数据集
file_path = "finaldata1.csv"
df = pd.read_csv(file_path)

# 排除所有存在缺失值的行
df_cleaned = df.dropna(axis=0, how="any")

# 保存预处理后的数据集
output_path = "finaldata1_nomissing.csv"
df_cleaned.to_csv(output_path, index=False)
