import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

file_path1 = "/mnt/h/fly/dataframe_data/classification.csv"
file_path2 = "/mnt/h/fly/dataframe_data/mefeatures_100K.csv"
file_path3 = "/mnt/h/fly/dataframe_data/proj_ccf-me_test_copy.csv"
#file_path3 = "/mnt/h/fly/dataframe_data/proj_test_copy.csv"
file_path4 = "/mnt/h/fly/dataframe_data/proj_test_copy.csv"
file_path5 = "/mnt/h/fly/dataframe_data/meta_data.csv"

Optic_Lobe = ['AME_L','AME_R','LA_L','LA_R','LO_L','LO_R','LOP_L','LOP_R','ME_L','ME_R','OCG']
Lateral_Complex = ['BU_L','BU_R','GA_L','GA_R']
Lateral_Horn = ['LH_L','LH_R']
Periesophageal_Neuropils = ['CAN_L','AMMC_L','FLA_L','CAN_R','AMMC_R','FLA_R','SAD','PRW'] 
Mushroom_body = ['MB_CA_L','MB_CA_R','MB_VL_R','MB_VL_L','MB_ML_L','MB_ML_R','MB_PED_L','MB_PED_R']
Inferior_Neuropils = ['ICL_L','ICL_R','IB_L','IB_R','ATL_L','ATL_R','CRE_L','CRE_R','SCL_L','SCL_R']
Ventromedial_Neuropils = ['VES_L','VES_R','GOR_L','GOR_R','SPS_L','SPS_R','IPS_L','IPS_R','EPA_L','EPA_R']
Antennal_Lobe = ['AL_L','AL_R']
Superior_Neuropils = ['SLP_L','SLP_R','SIP_L','SIP_R','SMP_L','SMP_R']
Ventrolateral_Neuropils = ['AVLP_L','AVLP_R','PVLP_L','PVLP_R','WED_L','WED_R','PLP_L','PLP_R','AOTU_L','AOTU_R']
Central_complex = ['NO','PB','EB','FB']
Gnathal_Ganglia = ['GNG']

CNS = ['LO_L','LO_R','LOP_L','LOP_R','BU_L','BU_R','GA_L','GA_R','LH_L','LH_R','CAN_L','AMMC_L','FLA_L','CAN_R','AMMC_R','FLA_R',
       'MB_CA_L','MB_CA_R','MB_VL_R','MB_VL_L','MB_ML_L','MB_ML_R','MB_PED_L','MB_PED_R','ICL_L','ICL_R','IB_L','IB_R','ATL_L','ATL_R','CRE_L','CRE_R','SCL_L','SCL_R',
       'VES_L','VES_R','GOR_L','GOR_R','SPS_L','SPS_R','IPS_L','IPS_R','EPA_L','EPA_R','AL_L','AL_R','SLP_L','SLP_R','SIP_L','SIP_R','SMP_L','SMP_R',
       'AVLP_L','AVLP_R','PVLP_L','PVLP_R','WED_L','WED_R','PLP_L','PLP_R','AOTU_L','AOTU_R','NO','PB','EB','FB','SAD','PRW','GNG','OCG']
PNS = ['ME_L','ME_R','LA_L','LA_R','OCG','AME_L','AME_R']

region = 'Ocella'

meta_data_df = pd.read_csv(file_path5)
#meta_data_df = meta_data_df[meta_data_df["Soma_region"].isin(Ventromedial_Neuropils)]
meta_data_df = meta_data_df[meta_data_df["Soma_region"].isin(["OCG"])]
classification_df = pd.read_csv(file_path1)
me_feature_df = pd.read_csv(file_path2)
me_feature_df = pd.merge(me_feature_df, meta_data_df, on='Name', how='left').dropna()
proj_ccf_me_df = pd.read_csv(file_path3)
proj_ccf_df = pd.read_csv(file_path4)


me_columns = [col for col in me_feature_df.columns if col.endswith('_me')]
single_columns = [item.replace('_me', '') for item in me_columns]
ccf_me_columns = proj_ccf_me_df.select_dtypes(include=['number']).columns.tolist()
ccf_me_columns = [col for col in ccf_me_columns if col != 'Name']
ccf_columns = proj_ccf_df.select_dtypes(include=['number']).columns.tolist()
ccf_columns = [col for col in ccf_columns if col != 'Name']

merge_me_ccf_me = pd.merge(proj_ccf_me_df, me_feature_df, on='Name', how='left').dropna()
merge_me_ccf = pd.merge(proj_ccf_df, me_feature_df, on='Name', how='left').dropna()

X = merge_me_ccf_me[me_columns + single_columns]
y = merge_me_ccf_me[ccf_me_columns]

X_part1 = X.iloc[:, :len(me_columns)] 
X_part2 = X.iloc[:, len(me_columns):]  

scaler_X1 = MinMaxScaler()
scaler_X2 = MinMaxScaler()

X_part1 = pd.DataFrame(scaler_X1.fit_transform(X_part1), columns=X_part1.columns, index=X_part1.index)
X_part2 = pd.DataFrame(scaler_X2.fit_transform(X_part2), columns=X_part2.columns, index=X_part2.index)
y = np.log1p(y)

num_rows = X_part1.shape[0]
#########################################################################
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

distance_part1 = pdist(X_part1, metric='euclidean')
distance_part2 = pdist(X_part2, metric='euclidean')
distance_y = pdist(y, metric='euclidean')

#distance_part1 = pdist(X_part1, metric='cosine')
#distance_part2 = pdist(X_part2, metric='cosine')
#distance_y = pdist(y, metric='cosine')

print(f"The number of Neurons: {num_rows}")

corr_part1_y, p_value_part1_y = pearsonr(distance_part1, distance_y)
print(f"Correlation between MicroEnv and Projection pattern: {corr_part1_y}, p = {p_value_part1_y}")

corr_part2_y, p_value_part2_y = pearsonr(distance_part2, distance_y)
print(f"Correlation between Morpho and Projection pattern: {corr_part2_y}, p = {p_value_part2_y}")

#exit()
###################################################################
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, gaussian_kde
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

# Custom colormap
colors = [(1, 1, 1), (0, 0, 1)]
cmap_name = 'white_to_blue'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from matplotlib.colors import LinearSegmentedColormap

def plot_density_correlation(x_data, y_data, x_label, region, num_rows, output_file, vmax=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, gaussian_kde

    #white_blue_cmap = LinearSegmentedColormap.from_list("white_to_blue", ["white", "blue"])

    corr_coeff, _ = pearsonr(x_data, y_data)

    heatmap, xedges, yedges = np.histogram2d(x_data, y_data, bins=30)

    if vmax is None:
        vmax = np.percentile(heatmap[heatmap > 0], 99) if np.any(heatmap > 0) else 1

    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(
    heatmap.T,
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    origin='lower',
    aspect='auto',
    cmap=plt.get_cmap("viridis"),
    norm=LogNorm(vmin=1, vmax=vmax)  
    )


    slope, intercept = np.polyfit(x_data, y_data, 1)
    line_x = np.linspace(min(x_data), max(x_data), 1000)
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, color='red', linewidth=3, label=f'Coeff = {corr_coeff:.3f}')
    #ax.legend(fontsize=60)

    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.xaxis.offsetText.set_visible(False)
    ax.yaxis.offsetText.set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_title(f"{region}\n(N = {num_rows})", fontsize=36)
    ax.set_xlabel(x_label, fontsize=32)
    ax.set_ylabel("Projection pattern", fontsize=32)
    ax.legend(fontsize=40)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    kde_x = gaussian_kde(x_data)
    kde_y = gaussian_kde(y_data)

    x_vals = np.linspace(min(x_data), max(x_data), 1000)
    y_vals = np.linspace(min(y_data), max(y_data), 1000)

    kde_x_vals = kde_x(x_vals)
    kde_y_vals = kde_y(y_vals)

    viridis = plt.get_cmap("viridis")
    ax_xdist = ax.inset_axes([0, 1.02, 1, 0.2], sharex=ax)
    ax_xdist.fill_between(x_vals, 0, kde_x_vals, color=viridis(0.8), alpha=0.6)
    ax_xdist.axis('off')

    ax_ydist = ax.inset_axes([1.02, 0, 0.2, 1], sharey=ax)
    ax_ydist.fill_betweenx(y_vals, 0, kde_y_vals, color=viridis(0.8), alpha=0.6)
    ax_ydist.axis('off')

    plt.savefig(output_file, format='tiff', dpi=300, bbox_inches='tight')
    plt.close()

#################################################################

#ME vs proj
plot_density_correlation(
    x_data=distance_part1,
    y_data=distance_y,
    x_label="MicroEnv feature",
    region=region,
    num_rows=num_rows,
    output_file=f"{region}_density.tiff"
)

#Dendrite vs proj
plot_density_correlation(
    x_data=distance_part2,
    y_data=distance_y,
    x_label="Morphological feature",
    region=region,
    num_rows=num_rows,
    output_file=f"{region}_density_dendrite.tiff"
)