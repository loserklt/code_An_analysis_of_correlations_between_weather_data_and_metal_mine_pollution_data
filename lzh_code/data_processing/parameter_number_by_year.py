import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('../data_water_quality/data_Pugh_Processing_1.csv')

# 统计每一年数据条数
year_counts = df['year'].value_counts().sort_index()

# 设置颜色：大于等于5为绿色，其余为灰色
colors = ['green' if v >= 5 else 'red' for v in year_counts.values]

plt.figure(figsize=(10,5))
bars = plt.bar(year_counts.index.astype(str), year_counts.values, color=colors)
plt.xlabel('Year')
plt.ylabel('Number')
plt.title('Annual Statistics of Water Quality Data Entries')

# 标注数据值
for bar, count in zip(bars, year_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(count),
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
