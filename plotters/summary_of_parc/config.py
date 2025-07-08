##########################################################
#Author:          Yufeng Liu
#Create time:     2024-04-30
#Description:               
##########################################################
import numpy as np
import pandas as pd
import pickle
import pysal.lib as pslib
from esda.moran import Moran



mRMR_f3 = ['AverageFragmentation', 'pca_vr1', 'Stems'] #MIQ


mRMR_f3me = ['AverageFragmentation_me', 'pca_vr1_me', 'Stems_me']  #MIQ

__FEAT24D__ = [
    'Stems', 'Bifurcations', 'Branches', 'Tips', 'OverallWidth', 'OverallHeight',
    'OverallDepth', 'Length', 'Volume', 'MaxEuclideanDistance', 'MaxPathDistance',
    'MaxBranchOrder', 'AverageContraction', 'AverageFragmentation',
    'AverageParent-daughterRatio', 'AverageBifurcationAngleLocal',
    'AverageBifurcationAngleRemote', 'HausdorffDimension',
    'pc11', 'pc12', 'pc13', 'pca_vr1', 'pca_vr2', 'pca_vr3'
]

__Subregion_id__ = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "997", "998", "999"]
#list(map(str, range(1, 419)))

BS7_COLORS = {
    'AME_R': 'lightseagreen',
'LO_R': 'darkviolet',
'NO': 'goldenrod',
'BU_R': 'firebrick',
'PB': 'teal',
'LH_R': 'mediumvioletred',
'LAL_R': 'indigo',
'SAD': 'darkslateblue',
'CAN_R': 'plum',
'AMMC_R': 'tomato',
'ICL_R': 'orchid',
'VES_R': 'dodgerblue',
'IB_R': 'chartreuse',
'ATL_R': 'peru',
'CRE_R': 'crimson',
'MB_PED_R': 'seashell',
'MB_VL_R': 'midnightblue',
'MB_ML_R': 'slategray',
'FLA_R': 'darkorange',
'LOP_R': 'aqua',
'EB': 'lightcoral',
'AL_R': 'yellowgreen',
'ME_R': 'saddlebrown',
'FB': 'lightsteelblue',
'SLP_R': 'rosybrown',
'SIP_R': 'deepskyblue',
'SMP_R': 'yellow',
'AVLP_R': 'magenta',
'PVLP_R': 'mediumslateblue',
'WED_R': 'sienna',
'PLP_R': 'darkcyan',
'AOTU_R': 'crimson',
'GOR_R': 'firebrick',
'MB_CA_R': 'springgreen',
'SPS_R': 'royalblue',
'IPS_R': 'forestgreen',
'SCL_R': 'darkkhaki',
'EPA_R': 'indianred',
'GNG': 'fuchsia',
'PRW': 'lavender',
'GA_R': 'blueviolet',
'AME_L': 'darkorchid',
'LO_L': 'crimson',
'BU_L': 'slateblue',
'LH_L': 'hotpink',
'LAL_L': 'gold',
'CAN_L': 'lawngreen',
'AMMC_L': 'tomato',
'ICL_L': 'mediumslateblue',
'VES_L': 'dodgerblue',
'IB_L': 'teal',
'ATL_L': 'chartreuse',
'CRE_L': 'orchid',
'MB_PED_L': 'darkslategray',
'MB_VL_L': 'aqua',
'MB_ML_L': 'firebrick',
'FLA_L': 'lightseagreen',
'LOP_L': 'lightsteelblue',
'AL_L': 'darkviolet',
'ME_L': 'mediumvioletred',
'SLP_L': 'rosybrown',
'SIP_L': 'deepskyblue',
'SMP_L': 'yellow',
'AVLP_L': 'magenta',
'PVLP_L': 'mediumslateblue',
'WED_L': 'sienna',
'PLP_L': 'darkcyan',
'AOTU_L': 'crimson',
'GOR_L': 'firebrick',
'MB_CA_L': 'springgreen',
'SPS_L': 'royalblue',
'IPS_L': 'forestgreen',
'SCL_L': 'darkkhaki',
'EPA_L': 'indianred',
'GA_L': 'fuchsia',
'LA_R': 'royalblue',
'LA_L': 'darkorange',
'OCG': 'limegreen'

}

def load_features(mefile, scale=1., feat_type='mRMR', flipLR=False, standardize=True):
    df = pd.read_csv(mefile, index_col=1).dropna()

    if feat_type == 'full':
        cols = df.columns
        fnames = [fname for fname in cols if fname[-3:] == '_me']
    elif feat_type == 'mRMR':
        # Features selected by mRMR
        fnames = mRMR_f3me
    elif feat_type == 'PCA':
        fnames = ['pca_feat1', 'pca_feat2', 'pca_feat3']
    elif feat_type == 'single':
        fnames = mRMR_f3
    else:
        raise ValueError("Unsupported feature types")

    if standardize:
        # standardize
        tmp = df[fnames]
        tmp = (tmp - tmp.mean()) / (tmp.std() + 1e-10)
        df[fnames] = tmp

    # scaling the coordinates to CCFv3-25um space
#    df['soma_x'] = df['soma_x']/scale - 85.8770447/2   
#    df['soma_y'] = df['soma_y']/scale - 63.0468216/2
#    df['soma_z'] = df['soma_z']/scale + 3.3968091/2
    df['soma_x'] = df['soma_x']/scale - 85.877   
    df['soma_y'] = df['soma_y']/scale - 63.046
    df['soma_z'] = df['soma_z']/scale + 3.416
    # we should remove the out-of-region coordinates
    zdim,ydim,xdim = (808, 383, 274)   # dimensions for CCFv3-25um atlas
    
    #changed##################################################
    in_region = (df['soma_x'] >= 0) & (df['soma_x'] < zdim) & \
                (df['soma_y'] >= 0) & (df['soma_y'] < ydim) & \
                (df['soma_z'] >= 0) & (df['soma_z'] < xdim)   
    #########################################################     
    df = df[in_region]
    print(f'Filtered out {in_region.shape[0] - df.shape[0]}')

    if flipLR:
        # mirror right hemispheric points to left hemisphere
        zdim2 = zdim // 2
        nzi = np.nonzero(df['soma_z'] < zdim2)
        loci = df.index[nzi]
        df.loc[loci, 'soma_z'] = zdim - df.loc[loci, 'soma_z']

    return df, fnames

def standardize_features(dfc, feat_names, epsilon=1e-8):
    fvalues = dfc[feat_names]
    fvalues = (fvalues - fvalues.mean()) / (fvalues.std() + epsilon)
    dfc.loc[:, feat_names] = fvalues.values

def normalize_features(dfc, feat_names, epsilon=1e-8):
    fvalues = dfc[feat_names]
    fvalues = (fvalues - fvalues.min()) / (fvalues.max() - fvalues.min() + epsilon)
    dfc.loc[:, feat_names] = fvalues.values

def gini_coeff(points):
    points = np.array(points)
    n = len(points)
    diff_sum = np.sum(np.abs(points[:, None] - points))
    return diff_sum / (2 * n * np.sum(points))

def moranI_score(coords, feats, eval_ids=None, reduce_type='average', threshold=5):
    """
    The coordinates should be in `um`, and as type of numpy.array
    The feats should be standardized
    """
    # spatial coherence
    weights = pslib.weights.DistanceBand.from_array(coords, threshold=threshold)
    avgI = []
    if eval_ids is None:
        eval_ids = range(feats.shape[1])
    for i in eval_ids:
        moran = Moran(feats[:,i], weights)
        avgI.append(moran.I)
    
    if reduce_type == 'average':
        avgI = np.mean(avgI)
    elif reduce_type == 'max':
        avgI = np.max(avgI)
    elif reduce_type == 'all':
        return avgI
    else:
        raise NotImplementedError
    return avgI

def get_me_ccf_mapper(me2ccf_file):
    # load the me to ccf correspondence file
    with open(me2ccf_file, 'rb') as fm2c:
        me2ccf = pickle.load(fm2c)
    # get the reverse map
    ccf2me = {}
    for k, v in me2ccf.items():
        if v in ccf2me:
            ccf2me[v].append(k)
        else:
            ccf2me[v] = [k]

    return me2ccf, ccf2me

