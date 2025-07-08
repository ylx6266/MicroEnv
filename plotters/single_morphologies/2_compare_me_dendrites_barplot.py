import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append('../../../')
from config import mRMR_f3, mRMR_f3me, moranI_score, load_features, standardize_features

import warnings
warnings.filterwarnings("ignore")


# compare the statistics of ME features and original dendritic features
feat_file = '../../../data/mefeatures_100K.csv'
rname = 'Calyx of Mushroom Body'

df, fnames = load_features(feat_file, feat_type='full')
df_cp = df[df['soma_region'].str.startswith("MB_CA")]
fn22_me = [col for col in df.columns if col.endswith('_me')]
fn22 = [col[:-3] for col in fn22_me]

num_neurons = df_cp.shape[0]
print("Number of Neurons:", num_neurons)

df_coords = df_cp[['soma_x', 'soma_y', 'soma_z']]  # to um
#df_me = df_cp[fn22_me]; standardize_features(df_me, fn22_me)
#df_de = df_cp[fn22]; standardize_features(df_de, fn22)
#df_de = df_de.loc[:, df_de.std() != 0]

df_me = df_cp[fn22_me].copy()
standardize_features(df_me, fn22_me)
df_de = df_cp[fn22].copy()
standardize_features(df_de, fn22)
df_de = df_de.loc[:, df_de.std() != 0]
valid_feats = df_de.columns.tolist()
df_me = df_me[[f + '_me' for f in valid_feats]]
fn22 = valid_feats  

use_subset = True
nsel = 5000
if use_subset and (df_me.shape[0] > nsel):
    random.seed(1024)
    sel_ids = np.array(random.sample(range(df_me.shape[0]), nsel))
    coords = df_coords.iloc[sel_ids].values
    mes = df_me.iloc[sel_ids].values
    des = df_de.iloc[sel_ids].values
        
else:
    coords = df_coords.values
    mes = df_me.values
    des = df_de.values

moranI_me = np.array(moranI_score(coords, mes, reduce_type='all'))
moranI_de = np.array(moranI_score(coords, des, reduce_type='all'))

print(f'Avg Moran Index for ME and DE are {moranI_me.mean():.2f}, {moranI_de.mean():.2f}')

sns.set_theme(style='ticks', font_scale=1.6)
fig, ax = plt.subplots(figsize=(10,8))

# Processing the feature labels
pf2label = {
    'AverageBifurcationAngleRemote': 'Avg. Bif. Angle Remote',
    'AverageBifurcationAngleLocal': 'Avg. Bif. Angle Local',
    'AverageContraction': 'Avg. Contraction',
    'AverageFragmentation': 'Avg. Fragmentation',
    'AverageParent-daughterRatio': 'Avg. PD ratio',
    'Bifurcations': 'Bifurcations',
    'Branches': 'Branches',
    'HausdorffDimension': 'Hausdorff Dimension',
    'MaxBranchOrder': 'Max. Branch Order',
    'Length': 'Total Length',
    'MaxEuclideanDistance': 'Max. Euc. Distance',
    'MaxPathDistance': 'Max. Path Distance',
    'Volume': 'Volume',
    'OverallDepth': 'Overall Depth',
    'OverallHeight': 'Overall Height',
    'OverallWidth': 'Overall Width',
    'Stems': 'Stems',
    'Tips': 'Tips',
    'pc11': 'PC11',
    'pc12': 'PC12',
    'pc13': 'PC13',
    'pca_vr1': 'PCA_variance_ratio1',
    'pca_vr2': 'PCA_variance_ratio2',
    'pca_vr3': 'PCA_variance_ratio3',
}
fn_ticks = []
for name in fn22:
    if name in pf2label:
        fn_ticks.append(pf2label[name])
    else:
        fn_ticks.append(name)
#index = np.arange(len(moranI_me))
#moranI_me = np.insert(moranI_me, 0, np.nan)
#moranI_de = np.insert(moranI_de, 0, np.nan)
#fn_ticks.insert(0, '')

bar_width = 0.35
index = np.arange(len(moranI_me))

bar1 = plt.bar(index, moranI_me, bar_width, color='indianred', label="Microenvironment", edgecolor='none')

bar2 = plt.bar(index, moranI_de, bar_width, color='blue', alpha=0.4, label="Morphology", edgecolor='none')

plt.ylabel("Moran's Index")
plt.xlabel('Morphological feature')



plt.xticks(ticks=index, labels=fn_ticks, rotation=52, ha='right', rotation_mode='anchor')
plt.xlim(-bar_width/2, len(fn_ticks)-0.5)
plt.ylim(0, 1)

# Style adjustments
axes = plt.gca()
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.spines['bottom'].set_linewidth(2)
axes.spines['left'].set_linewidth(2)
axes.xaxis.set_tick_params(width=2, direction='in', labelsize=17)
axes.yaxis.set_tick_params(width=2, direction='in')

plt.subplots_adjust(left=0.12, bottom=0.41)

# Adding legend
#plt.legend(frameon=False, loc='upper right')


plt.text(
    x=len(fn_ticks)/2, y=plt.ylim()[1]*1,
    s=f'(N = {num_neurons})',
    ha='center', va='bottom', fontsize=18
)
plt.title(f'{rname}', fontsize=24, pad=30)


# Save plot as PNG
plt.savefig(f'MoranI_improvement_of_{rname}.png', dpi=1200)
plt.close()

print()
