import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors  # For RGB to Hex conversion

df = pd.read_csv('merged_region_ccf-me.csv')

numeric_columns = df.select_dtypes(include=[np.number]).columns

grouped = df.groupby('region_id_me')[numeric_columns].mean()

grouped_vectors = grouped.iloc[:, 10:]
grouped_array = grouped_vectors.values

cos_sim_matrix = cosine_similarity(grouped_array)

# 进行层次聚类
linkage_matrix = linkage(cos_sim_matrix, method='ward')

# 根据聚类结果为热图排序
dendro = dendrogram(linkage_matrix, no_plot=True)
order = dendro['leaves']  # 获取聚类后的排序

# 对余弦相似度矩阵进行排序
sorted_cos_sim_matrix = cos_sim_matrix[order, :][:, order]

# 获取排序后的区域ID
sorted_region_ids = np.array(grouped.index)[order]

# 生成白到红的渐变色
colors = sns.color_palette("Reds", as_cmap=True)

# 创建一个gridspec布局，设置两列：一列用于绘制热图，另一列用于绘制树状图
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(1, 2, width_ratios=[0.1, 0.9])  # Make dendrogram column smaller

# 画出聚类树状图
ax_dendro = fig.add_subplot(gs[0, 0])
dendrogram(linkage_matrix, ax=ax_dendro, orientation='left', labels=sorted_region_ids)
ax_dendro.set_xticks([])  # 隐藏 x 轴刻度
ax_dendro.set_yticks([])  # 隐藏 y 轴刻度
ax_dendro.invert_yaxis()  # Flip the y-axis vertically

# 画出热图
ax_heatmap = fig.add_subplot(gs[0, 1])
sns.heatmap(sorted_cos_sim_matrix, cmap=colors, annot=False, xticklabels=sorted_region_ids, yticklabels=sorted_region_ids, ax=ax_heatmap, cbar=True)


# 隐藏热图的 x 和 y 轴刻度标签
ax_heatmap.set_xticks([])  
ax_heatmap.set_yticks([])

# 添加标题
# ax_heatmap.set_title('Cosine Similarity Heatmap of Average Vectors by Region (Clustering)', fontsize=16)

# 调整布局，使两个图形紧凑而不重叠
fig.tight_layout()

# 保存热图图像
plt.savefig('cosine_similarity_heatmap_with_cluster_dendrogram_flipped.png', dpi=1200, bbox_inches='tight')  
plt.close()


#######################################################################################################


# 保存绘制热图所用的直接数据到 CSV 文件
heatmap_data = pd.DataFrame(sorted_cos_sim_matrix, index=sorted_region_ids, columns=sorted_region_ids)
heatmap_data.to_csv('heatmap_data.csv', index=True, header=True)  # Save cosine similarity matrix as CSV

# Optionally, save region colors to CSV (for reference)
region_colors = pd.DataFrame({
    'Region ID': sorted_region_ids,
    'Color': [mcolors.to_hex(color) for color in colors(range(len(sorted_region_ids)))]  # Convert RGB color to hex using matplotlib
})
region_colors.to_csv('region_colors.csv', index=False)  # Save region colors as CSV
