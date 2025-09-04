import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_excel('../data_water_quality/data_Pugh_full.xlsx')

# 统计每个编号的非缺失数据条数
count_per_param = df.groupby('parameter_shortname')['sample_value'].count().reset_index()

# 编号升序排列
count_per_param = count_per_param.sort_values('parameter_shortname')

#筛选出数量大于35的指标
filtered = count_per_param[count_per_param['sample_value'] >= 35]
# 绘制可视化
plt.figure(figsize=(14,6))
plt.bar(count_per_param['parameter_shortname'].astype(str), count_per_param['sample_value'])
plt.xlabel('parameter_shortname')
plt.ylabel('number')
plt.title('Statistics of data entries corresponding to each indicator code')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,6))
plt.bar(filtered['parameter_shortname'].astype(str), filtered['sample_value'])
plt.xlabel('parameter_shortname')
plt.ylabel('number')
plt.title('Statistics of data entries corresponding to each indicator code(>=35)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
