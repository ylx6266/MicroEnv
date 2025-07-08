import pandas as pd

# 读取 CSV 文件
meta_test = pd.read_csv('region_colors.csv')
proj_ccf_me_test_copy = pd.read_csv('meta_test.csv', dtype={0: str})

# 根据 'region_id_me' 列合并两个 DataFrame
merged_df = pd.merge(meta_test, proj_ccf_me_test_copy, on='region_id_me', how='left')

# 保留 'region_name_ccf' 列的下划线前部分
#merged_df['region_name_ccf'] = merged_df['region_name_ccf'].str.split('_').str[0]

# 如果需要保存合并后的结果，可以使用下面的代码：
merged_df.to_csv('merged_color.csv', index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

# 读取 CSV 文件
merged_df = pd.read_csv('merged_color.csv')

# 获取 'region_name_ccf' 列的唯一值
unique_region_names = merged_df['region_name_ccf'].unique()

# 使用 Matplotlib 的颜色循环来生成颜色
color_cycle = cycle(plt.cm.tab20.colors)  # tab20 是一种有足够颜色的颜色映射
region_to_color = {region: next(color_cycle) for region in unique_region_names}

# 将 RGB 颜色转换为十六进制颜色
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

# 为每个 region_name_ccf 分配十六进制颜色
region_to_hex_color = {region: rgb_to_hex(color) for region, color in region_to_color.items()}

# 将颜色赋值到新的 'Color' 列
merged_df['Color'] = merged_df['region_name_ccf'].map(region_to_hex_color)

# 查看结果
print(merged_df[['region_name_ccf', 'Color']].drop_duplicates())

# 如果需要保存到新的 CSV 文件
merged_df.to_csv('merged_color.csv', index=False)