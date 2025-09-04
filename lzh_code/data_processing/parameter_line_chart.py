import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 读取数据
df = pd.read_csv('filtered_core_params .csv')

# 时间字段
time_col = 'date' if 'date' in df.columns else 'year'

# 选出参数列
indicator_cols = [col for col in df.columns if col.lower() not in ['date', 'year', 'time', 'month']]

# 标准化
scaler = StandardScaler()
df_std = df.copy()
df_std[indicator_cols] = scaler.fit_transform(df[indicator_cols])

# 画图
plt.figure(figsize=(16,8))
for col in indicator_cols:
    plt.plot(df[time_col], df_std[col], label=col, marker='o')
plt.xlabel(time_col)
plt.ylabel('Normalized Parameter Values (Z-score)')
plt.title('All parameters are normalized and change trends over time')
plt.xticks(rotation=45)
plt.legend(loc='upper right', bbox_to_anchor=(1.16,1))
plt.tight_layout()
plt.show()
