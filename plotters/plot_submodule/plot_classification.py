import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('merged_region_classification.csv')

count_matrix = pd.crosstab(df['region_id_me'], df['super_class'])
proportion_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)

ordered_region_ids = [
234, 332, 297, 299, 287, 401, 291, 289, 290, 293, 284, 288, 295, 298, 258, 285, 286, 292, 310, 321, 386, 
137, 170, 185, 20, 296, 231, 236, 51, 50, 54, 197, 300, 85, 198, 77, 256, 259, 78, 72, 73, 257, 135, 340, 
35, 83, 30, 88, 215, 81, 80, 84, 79, 82, 177, 211, 389, 29, 34, 32, 31, 33, 201, 40, 189, 216, 387, 205, 388, 
196, 203, 86, 200, 76, 45, 192, 37, 38, 41, 190, 191, 39, 36, 188, 10, 118, 382, 385, 207, 208, 206, 209, 383, 
294, 240, 396, 243, 399, 237, 283, 400, 241, 238, 239, 397, 398, 166, 262, 265, 16, 19, 186, 264, 157, 159, 17, 
161, 164, 160, 156, 158, 162, 263, 43, 122, 119, 120, 179, 390, 181, 393, 21, 261, 18, 165, 87, 182, 183, 180, 184, 
111, 114, 47, 250, 252, 22, 249, 248, 42, 44, 46, 230, 251, 307, 308, 391, 392, 394, 395, 187, 270, 58, 60, 67, 68, 
56, 61, 57, 59, 124, 126, 1, 123, 125, 274, 93, 91, 95, 115, 116, 110, 105, 106, 92, 94, 90, 107, 108, 112, 301, 194, 
302, 303, 304, 335, 337, 338, 233, 52, 232, 306, 333, 334, 74, 260, 269, 12, 266, 89, 268, 346, 121, 55, 13, 62, 11, 
75, 117, 53, 195, 235, 109, 336, 113, 309, 442, 446, 448, 451, 441, 444, 436, 440, 445, 450, 453, 454, 384, 437, 435, 
455, 439, 449, 415, 438, 417, 418, 217, 223, 97, 102, 452, 443, 447, 314, 313, 317, 312, 315, 213, 311, 319, 322, 316, 
318, 343, 342, 347, 351, 344, 349, 345, 350, 353, 354, 136, 138, 144, 145, 146, 267, 320, 139, 140, 142, 141, 143, 174, 
169, 172, 348, 171, 178, 173, 176, 219, 220, 218, 222, 242, 471, 478, 479, 466, 468, 474, 476, 469, 480, 463, 481, 475, 
482, 477, 429, 428, 430, 366, 367, 361, 352, 355, 276, 470, 365, 465, 360, 282, 363, 277, 280, 358, 362, 364, 356, 359, 279, 
278, 281, 96, 271, 339, 275, 272, 273, 409, 402, 405, 403, 406, 407, 404, 408, 370, 381, 374, 375, 380, 373, 376, 378, 377, 379, 
66, 372, 63, 369, 71, 69, 70, 65, 64, 371, 221, 224, 229, 99, 100, 103, 226, 101, 225, 227, 228, 98, 104, 244, 247, 245, 246, 462, 
458, 457, 456, 461, 459, 460, 167, 129, 130, 127, 128, 6, 341, 133, 134, 131, 132, 331, 330, 357, 48, 15, 210, 5, 9, 7, 8, 431, 434, 
432, 433, 472, 473, 464, 467, 412, 413, 416, 411, 420, 168, 204, 424, 427, 410, 414, 212, 422, 426, 214, 423, 425, 49, 152, 148, 149, 
4, 28, 153, 26, 154, 155, 150, 151, 3, 254, 325, 326, 324, 323, 328, 327, 329, 305, 24, 253, 255

]

missing_region_ids = [region_id for region_id in ordered_region_ids if region_id not in proportion_matrix.index]

for region_id in missing_region_ids:
    proportion_matrix.loc[region_id] = [0] * proportion_matrix.shape[1]

proportion_matrix = proportion_matrix.loc[ordered_region_ids]


#scaler = StandardScaler()
#normalized_data = scaler.fit_transform(proportion_matrix.T)

#linked = linkage(normalized_data, method='ward')
#dendro = dendrogram(linked, no_plot=True)
#ordered_cell_types = [proportion_matrix.columns[i] for i in dendro['leaves']]
#proportion_matrix = proportion_matrix[ordered_cell_types]

plt.figure(figsize=(10, 8))
sns.heatmap(proportion_matrix, annot=False, cmap='Blues', fmt='.2f', cbar=True)

plt.title('Cell Type Proportions for Each Region', fontsize=14)
plt.xlabel('Cell Type', fontsize=12)
plt.ylabel('Region ID', fontsize=12)

plt.savefig('cell_type_proportions_heatmap2.png', bbox_inches='tight', dpi=1200)

