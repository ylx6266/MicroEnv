import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 读取CSV文件
df = pd.read_csv('merged_color.csv')

# 提取颜色列
colors = df['Color'].tolist()

# 将十六进制颜色码转换为RGB
rgb_colors = [mcolors.hex2color(color) for color in colors]

# 创建一个颜色映射
fig, ax = plt.subplots(figsize=(5, 1))

# 在颜色条上显示这些颜色
ax.imshow([rgb_colors], aspect='auto')
ax.set_xticks([])  # 不显示x轴刻度
ax.set_yticks([])  # 不显示y轴刻度

# 设置标题或其他标签
ax.set_title('Color Bar')

# 展示颜色条
plt.show()
