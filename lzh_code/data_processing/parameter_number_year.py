import pandas as pd

# 读取数据
df = pd.read_csv('../data_water_quality/data_Pugh_Processing_2.csv')

# 取出所有污染物字段（排除'year'列）
param_cols = [col for col in df.columns if col != 'year']

# 创建一个DataFrame，用于存储每个指标每年的非空数据数量
data_avail = pd.DataFrame(index=param_cols, columns=sorted(df['year'].unique()))

# 填充每个指标每年有效数据条数
for year in df['year'].unique():
    sub = df[df['year'] == year]
    for col in param_cols:
        data_avail.loc[col, year] = sub[col].notna().sum()

# 将结果转为整数
data_avail = data_avail.astype(int)

# 筛选：每个指标每年都至少有X条数据（如设为3条）
min_count = 3
mask = (data_avail >= min_count).all(axis=1)
good_params = data_avail.index[mask].tolist()

print("每年都至少有%d条数据的指标有：" % min_count)
for col in good_params:
    print(col)

# 如果要保留这些指标的数据
df_good = df[['year'] + good_params]
df_good.to_csv('filtered_params.csv', index=False)

# 可视化展示每个指标每年数据条数（热力图）
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, max(6, len(good_params)//2)))
sns.heatmap(data_avail.loc[good_params], annot=True, fmt='d', cmap='Greens')
plt.xlabel('Year')
plt.ylabel('parameter_id')
plt.title('Heatmap of Annual Data Volume for Each Pollutant Indicator (After Filtering)')
plt.tight_layout()
plt.show()
