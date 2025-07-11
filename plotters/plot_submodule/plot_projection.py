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

linkage_matrix = linkage(cos_sim_matrix, method='ward')
dendro = dendrogram(linkage_matrix, no_plot=True)
order = dendro['leaves']  
sorted_cos_sim_matrix = cos_sim_matrix[order, :][:, order]

sorted_region_ids = np.array(grouped.index)[order]

colors = sns.color_palette("Reds", as_cmap=True)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(1, 2, width_ratios=[0.1, 0.9])  # Make dendrogram column smaller

ax_dendro = fig.add_subplot(gs[0, 0])
dendrogram(linkage_matrix, ax=ax_dendro, orientation='left', labels=sorted_region_ids)
ax_dendro.set_xticks([])  
ax_dendro.set_yticks([]) 
ax_dendro.invert_yaxis()  # Flip the y-axis vertically

ax_heatmap = fig.add_subplot(gs[0, 1])
sns.heatmap(sorted_cos_sim_matrix, cmap=colors, annot=False, xticklabels=sorted_region_ids, yticklabels=sorted_region_ids, ax=ax_heatmap, cbar=True)


ax_heatmap.set_xticks([])  
ax_heatmap.set_yticks([])

# ax_heatmap.set_title('Cosine Similarity Heatmap of Average Vectors by Region (Clustering)', fontsize=16)
fig.tight_layout()

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
