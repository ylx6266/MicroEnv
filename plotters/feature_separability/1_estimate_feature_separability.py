##########################################################
#Author:          Yufeng Liu
#Create time:     2024-03-25
#Description:               
##########################################################
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import random
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist
from scipy import stats

from file_io import load_image
from anatomy.anatomy_config import TEST_REGION, REGION314, MASK_CCF25_FILE
from anatomy.anatomy_core import parse_ana_tree
from math_utils import get_exponent_and_mantissa



def cortical_separability(mefile, regions, sname, disp_type, cnt_thres=20):
    
    laminations = ['1', '2/3', '4', '5', '6a', '6b']
    
    # All sub-regions
    sregions = []
    for region in regions:
        for lam in laminations:
            sregions.append(region + lam)

    t0 = time.time()
    # load the microenvironment features and their affiliated meta informations
    df = pd.read_csv(mefile, index_col=0)
    dfc = df[df.soma_region.isin(sregions)]

    rnames, rcnts = np.unique(dfc.soma_region, return_counts=True)
    # do not take into consideration of low-count regions
    rnames = rnames[rcnts > cnt_thres]
    # re-select the neurons
    dfc = dfc[dfc.soma_region.isin(rnames)]
    r671_names = dfc.soma_region
    lams = [r1[len(r2):] for r1, r2 in zip(r671_names, dfc.region_name_r316)]

    #feat_names = [fname for fname in dfc.columns if fname[-3:] == '_me']
    __FN__ = ('AverageFragmentation', 'AverageBifurcationAngleLocal', 'MaxEuclideanDistance')
    feat_names = [fname+'_me' for fname in __FN__]
    fnames = feat_names + ['region_name_r316', 'soma_region']
    dfc = dfc[fnames]
    dfc['lamination'] = lams
    print(f'--> Time used in loading and processing data: {time.time() - t0:.2f} seconds')

    # rename the columns 
    sns.set_theme(style='ticks', font_scale=1.6)
    rmapper = {
        'Length_me': 'Length (µm)',
        'AverageFragmentation_me': 'Avg. Fragmentation',
        'AverageContraction_me': 'Straightness'
    }
    dfc.rename(columns=rmapper, inplace=True)

    # visualize using pairplot
    if disp_type == 'l':
        hue = 'lamination'
        leg_title = 'Lamination'
        hue_order = laminations
        figname = f'lamination_{sname}.png'
    elif disp_type == 'r316':
        hue = 'region_name_r316'
        leg_title = 'Region'
        hue_order = regions
        figname = f's-regions_{sname}.png'
    elif disp_type == 'r671':
        hue = 'soma_region'
        leg_title = 'Region'
        hue_order = rnames
        figname = f'sl_regions_{sname}.png'

    g = sns.pairplot(dfc, vars=['Length (µm)', 'Avg. Fragmentation', 'Straightness'], hue=hue, plot_kws={'marker':'.'}, 
                     kind='scatter', diag_kws={'common_norm':False},
                     hue_order=hue_order)
    handles = g._legend_data.values()
    labels = g._legend_data.keys()
    g._legend.remove()
    g.fig.legend(handles=handles, labels=labels, markerscale=3, loc='center right',
                 labelspacing=0.4, handletextpad=-0.2, fontsize=15,
                 title=leg_title)
    
    plt.savefig(figname, dpi=300)
    plt.close()
    

    #mat_mean = dfc[[fname+'_me' for fname in __FN__] + ['soma_region']].groupby('soma_region').mean()
    #mat_mean = StandardScaler().fit_transform(mat_mean)


def get_struct(id_path, bstructs):
    for idx in id_path[::-1]:
        if idx in bstructs:
            return idx
    else:
        return 0

def dsmatrix_of_all(mefile, ds_file1, ds_file2, metric='euclidean', cnt_thres=20):
    __FN__ = ('AverageFragmentation', 'pca_vr1', 'Stems')
    feat_names = [fname+'_me' for fname in __FN__]
    random.seed(1024)

    # helper functions
    def dsmatrix(mefile, ds_file1, ds_file2, cnt_thres, feat_names, metric):
        df = pd.read_csv(mefile, index_col=1)
        #print(df.columns)
        na_mask = df.soma_region.isna()
        # get the salient regions
        ana_tree = parse_ana_tree(keyname='id')
        salient_names = [ana_tree[idx]['acronym'] for idx in TEST_REGION]
        # in regions
        in_region = df.soma_region.isin(salient_names)
        fnames = feat_names + ['soma_region']
        # remove regions with insufficient neurons
        dfc = df[~na_mask & in_region][fnames]
        rs, rcs = np.unique(dfc.soma_region, return_counts=True)
        rcs_m = rcs >= cnt_thres
        rs = rs[rcs_m]
        rcs = rcs[rcs_m]
        dfc = dfc[dfc.soma_region.isin(rs)]
        # standardize
        tmp = dfc[feat_names].copy()
        tmp = (tmp - tmp.mean()) / tmp.std()
        if metric == 'cosine':
            tmp = tmp / (np.linalg.norm(tmp.values, axis=1, keepdims=True) + 1e-10)

        # estimate DS matrix
        t0 = time.time()
        dsm1 = np.zeros((len(rs), len(rs)))
        dsm2 = np.zeros((len(rs), len(rs)))
        for irs, rsi in enumerate(rs):
            rsm1 = tmp[dfc.soma_region == rsi]
            for jrs in range(irs, len(rs)):
                rsj = rs[jrs]
                rsm2 = tmp[dfc.soma_region == rsj]
                pdists = pairwise_distances(rsm1, rsm2, metric=metric)
                pm = pdists.mean()
                ps = pdists.std()
                dsm1[irs, jrs] = pm
                dsm1[jrs, irs] = pm
                dsm2[irs, jrs] = ps
                dsm2[jrs, irs] = ps
                
            print(f'[{irs}/{len(rs)}]: {time.time() - t0:.2f} seconds')

        d1 = pd.DataFrame(dsm1, columns=rs, index=rs)
        d2 = pd.DataFrame(dsm2, columns=rs, index=rs)
        # to file
        d1.to_csv(ds_file1)
        d2.to_csv(ds_file2)

        return d1, d2
    # --- End of helper functions --- #


    # --------- Inter-neuronal similarity ----------- #
    if os.path.exists(ds_file1) and os.path.exists(ds_file2):
        d1 = pd.read_csv(ds_file1, index_col=0)
        d2 = pd.read_csv(ds_file2, index_col=0)
    else:
        d1, d2 = dsmatrix(mefile, ds_file1, ds_file2, cnt_thres, feat_names, metric)
    print(d1.values.max(), d1.values.min())
       
    # Map the regions to brain structures
    '''
    bstructs = {
                688: 'CTX' 
                623: 'CNU',# STR + PAL
                512: 'CB',
                343: 'BS'   #: IB(549(TH)+HY), MB, HB
                }
    '''
    bstructs = {
        2:"AME_R",
3:"LO_R",
4:"NO",
5:"BU_R",
6:"PB",
7:"LH_R",
8:"LAL_R",
9:"SAD",
10:"CAN_R",
11:"AMMC_R",
12:"ICL_R",
13:"VES_R",
14:"IB_R",
15:"ATL_R",
16:"CRE_R",
17:"MB_PED_R",
18:"MB_VL_R",
19:"MB_ML_R",
20:"FLA_R",
22:"LOP_R",
23:"EB",
24:"AL_R",
25:"ME_R",
26:"FB",
27:"SLP_R",
28:"SIP_R",
29:"SMP_R",
30:"AVLP_R",
31:"PVLP_R",
32:"WED_R",
33:"PLP_R",
34:"AOTU_R",
35:"GOR_R",
36:"MB_CA_R",
37:"SPS_R",
38:"IPS_R",
39:"SCL_R",
40:"EPA_R",
49:"GNG",
50:"PRW",
51:"GA_R",
52:"AME_L",
53:"LO_L",
54:"BU_L",
55:"LH_L",
56:"LAL_L",
57:"CAN_L",
58:"AMMC_L",
59:"ICL_L",
60:"VES_L",
61:"IB_L",
62:"ATL_L",
63:"CRE_L",
64:"MB_PED_L",
65:"MB_VL_L",
66:"MB_ML_L",
67:"FLA_L",
69:"LOP_L",
70:"AL_L",
71:"ME_L",
72:"SLP_L",
73:"SIP_L",
74:"SMP_L",
75:"AVLP_L",
76:"PVLP_L",
77:"WED_L",
78:"PLP_L",
79:"AOTU_L",
80:"GOR_L",
81:"MB_CA_L",
82:"SPS_L",
83:"IPS_L",
84:"SCL_L",
85:"EPA_L",
86:"GA_L",
997:"LA_R",
998:"LA_L",
999:"OCG"
    }
    

    if 1:
        keepr = []
        ana_tree_n = parse_ana_tree(keyname='name')
        struct_ids = bstructs.keys()
        for region in d1.index:
            id_path = ana_tree_n[region]['structure_id_path']
            sid = get_struct(id_path, struct_ids)
            if sid == 0:
                keepr.append('')
            else:
                keepr.append(bstructs[sid])
        keepr = np.array(keepr)
        print(keepr)
        keepr_nz = (keepr!='').nonzero()[0]
        keepr = keepr[keepr_nz]
        d1 = d1.iloc[keepr_nz, keepr_nz]
        d2 = d2.iloc[keepr_nz, keepr_nz]
        keepr_s = pd.Series(keepr, name='structure')
        lut = dict(zip(np.unique(keepr_s), sns.hls_palette(len(np.unique(keepr_s)), l=0.5, s=0.8)))
        col_colors = keepr_s.map(lut)

        '''
        if metric == 'cosine':
            vmin, vmax = 0, 1
        elif metric == 'euclidean':
            vmin, vmax = 1, 4
        g = sns.clustermap(d1.reset_index(drop=True), cmap='hot_r', row_colors=col_colors, 
                           vmin=vmin, vmax=vmax, metric='euclidean') 
        for label in np.unique(keepr_s):
            g.ax_col_dendrogram.bar(0, 0, color=lut[label],
                                    label=label, linewidth=0)
        g.ax_col_dendrogram.legend(loc="center", ncol=2)
        reordered_ind = g.dendrogram_col.reordered_ind
        
        plt.savefig('tmp.png')
        plt.close()
        '''

        sns.set_theme(style='ticks', font_scale=1.7)
        # mean vs std
        m_intra = np.diagonal(d1).astype(float)
        s_intra = np.diagonal(d2).astype(float)
        smr = s_intra / m_intra
        df_mss = pd.DataFrame(np.array([m_intra, s_intra, smr, keepr]).transpose(), columns=('mean', 'std', 'sm ratio', 'struct')).astype({'mean': float, 'std': float, 'sm ratio': float, 'struct': str})
        g1 = sns.lmplot(data=df_mss, x='mean', y='std', 
                        scatter_kws={'color': 'black', 's': 12, 'alpha': 0.7},
                        line_kws={'color': 'red'})
        r, p = stats.pearsonr(df_mss['mean'], df_mss['std'])
        ax_g1 = plt.gca()
        ax_g1.text(0.55, 0.16, r'$R={:.2f}$'.format(r), transform=ax_g1.transAxes)
        e, m = get_exponent_and_mantissa(p)
        ax_g1.text(0.55, 0.08, r'$P={%.1f}x10^{%d}$' % (m, e),
                transform=ax_g1.transAxes)
        plt.savefig('regional_mean_vs_std.png', dpi=1200)
        plt.close()

        g2 = sns.boxplot(data=df_mss, x='struct', y='sm ratio', width=0.35, color='black', 
                         fill=False, order=sorted(keepr))
        plt.ylabel('Coefficient of Variance')
        plt.xlabel('Brain area')
        # rotate xticks
        plt.xticks(rotation=45, rotation_mode='anchor', ha='right', va='top')
        plt.subplots_adjust(left=0.15, bottom=0.3)
        plt.savefig('regional_CV.png', dpi=1200)
        plt.close()

        sns.boxplot(data=df_mss, x='struct', y='std')
        plt.savefig('regional_std.png', dpi=1200)
        plt.close()



    #---- Section 2: interareal similarity
    if 0:
        # plot the correlation between distance and similarity
        df = pd.read_csv(mefile, index_col=0)
        tmp = df[feat_names].copy()
        tmp = (tmp - tmp.mean()) / tmp.std()
        dfc = df.copy()
        dfc[feat_names] = tmp
        nsel = 10000
        sel_ids = random.sample(range(dfc.shape[0]), nsel)
        df_sel = dfc.iloc[sel_ids]
        pdists = pdist(df_sel[feat_names])
        cdists = pdist(df_sel[['soma_x', 'soma_y', 'soma_z']] /1000.)
        ## for debug
        cm = cdists < 1
        pdists = pdists[cm]
        cdists = cdists[cm]
        #
        nsel2 = 50000 
        sel_ids2 = random.sample(range(pdists.shape[0]), nsel2)
        cdists_sel = cdists[sel_ids2]
        pdists_sel = pdists[sel_ids2]
        dfpd = pd.DataFrame(np.array([cdists_sel, pdists_sel]).transpose(), columns=['cdist', 'pdist'])
        sns.kdeplot(data=dfpd, x='cdist', y='pdist', fill=True, levels=500, thresh=0.005)
        stride = 0.05

        p75s = []
        for cs1 in np.arange(0, 1., stride):
            cs2 = cs1 + stride
            csm = (cdists_sel > cs1) & (cdists_sel <= cs2)
            pts = pdists_sel[csm]
            p75s.append(np.percentile(pts, 50))
        #import ipdb; ipdb.set_trace()
        plt.plot(np.arange(0, 1, stride)+stride//2, p75s, 'o-r')
        
        plt.savefig('tmp3.png')
        plt.close()


    # ------ Section 4: comparison between ds_inter and ds_intra ------- #
    if 1:
        from scipy.stats import ranksums

        # Function to convert p-value to stars
        def p_value_to_stars(p):
            if p < 0.0001:
                return '****'
            elif p < 0.001:
                return '***'
            elif p < 0.01:
                return '**'
            elif p < 0.05:
                return '*'
            else:
                return 'n.s.'  # not significant

        ds_intra = np.diagonal(d1)
        ds_inter = d1.values[np.triu_indices_from(d1, k=1)]
        print(ds_intra.mean(), ds_inter.mean())
        # Create a DataFrame
        df_ds = pd.DataFrame({
            "Values": np.hstack((ds_intra, ds_inter)),
            "Category": ['Intra-region'] * len(ds_intra) + ['Inter-region'] * len(ds_inter)
        })

        g2 = sns.boxplot(data=df_ds, x='Category', y='Values', width=0.2, color='black',
                         fill=False, order=['Inter-region', 'Intra-region'])
        # Annotate test results
        # Perform the Wilcoxon Rank Sum Test
        stat, p_value = ranksums(ds_intra, ds_inter)
        
        y_max = df_ds['Values'].max()  # Adjust spacing for the annotation

        # Drawing a line between categories
        x_values = [0, 1]  # Adjust these based on the number of categories
        line_height = y_max + 0.1  # Adjust the line height as needed
        plt.plot(x_values, [line_height] * 2, 'k-', linewidth=2)  # 'k-' is for black solid line
        
        # Annotate the plot with the significance level as stars
        stars = p_value_to_stars(p_value)   # two-sided Wilcoxon rank sum test
        print(f'p_value={p_value}')
        e, m = get_exponent_and_mantissa(p_value)
        plt.annotate(stars + r'$ (p={%.1f}x10^{%d})$' % (m, e), xy=(0.5, 0.995), xycoords='axes fraction', 
                     ha='center', va='top', fontsize=15, color='black')
        plt.xlim(-0.5, 1.5)
        plt.ylim(0.75, 7.7)

        fig = plt.gcf()
        fig.set_size_inches(6,6)
        ax = plt.gca()
        __LABEL_FONTS__ = 25

        plt.xlabel(f'', fontsize=__LABEL_FONTS__)
        plt.ylabel(f'Distance of feature', fontsize=__LABEL_FONTS__)
        lw = 2
        ax.spines['left'].set_linewidth(lw)
        ax.spines['bottom'].set_linewidth(lw)
        ax.spines['right'].set_linewidth(lw)
        ax.spines['top'].set_linewidth(lw)
        ax.xaxis.set_tick_params(width=lw, direction='in', labelsize=__LABEL_FONTS__ - 4)
        ax.yaxis.set_tick_params(width=lw, direction='in', labelsize=__LABEL_FONTS__ - 4)
        plt.legend(loc='upper right', frameon=False, fontsize=__LABEL_FONTS__)
        plt.savefig('inter-and-intra-region_distance.png', dpi=1200)
        plt.close()


def analyze_dsmatrix_vs_parcellations(dsmean_file, parc_file, atlas_file=None):
    # load the ds matrix
    dsmean = pd.read_csv(dsmean_file, index_col=0)
    # load the standard CCFv3 atlas
    if atlas_file is None:
        atlas = load_image(MASK_CCF25_FILE)
    else:
        atlas = load_image(atlas_file)
    # load the annotation file
    ana_tree = parse_ana_tree(keyname='name')
    # load parcellation file
    parc = load_image(parc_file)

    # 
    ds_intra = np.diagonal(dsmean)
    ds_thr = 1.  #setting threshod 1.9
    high_ds = ds_intra > ds_thr
    high_ds_ids = high_ds.nonzero()[0]
    high_regs = dsmean.index[high_ds_ids]
    #print(high_regs)
    for ireg, reg in enumerate(high_regs):
        idx = ana_tree[reg]['id']
        atlas_mask = atlas == idx
        parc_mask = parc[atlas_mask]
        print(f'{reg}/ [DS: {ds_intra[high_ds_ids[ireg]]:.2f}]: {atlas_mask.sum()}, {np.unique(parc_mask)}')
#    import ipdb
#    ipdb.set_trace()
#    print()



if __name__ == '__main__':
    mefile = '../../../data/mefeatures_100K.csv'

#    if 0:
#        region_dict = {
#            'all':['FRP', 'MOp', 'MOs', 'SSp-n', 'SSp-bfd', 'SSp-ll', 'SSp-m', 'SSp-ul', 'SSp-tr',
#                   'SSp-un', 'SSs', 'GU', 'VISC', 'AUDd', 'AUDp', 'AUDpo', 'AUDv', 'VISal', 
#                   'VISam', 'VISl', 'VISp', 'VISpl', 'VISpm', 'VISli', 'VISpor', 'ACAd', 'ACAv',
#                   'PL', 'ILA', 'ORBl', 'ORBm', 'ORBvl', 'AId', 'AIp', 'AIv', 'RSPagl', 'RSPd', 
#                   'RSPv', 'VISa', 'VISrl', 'TEa', 'PERI', 'ECT'],
#            'SSp':['SSp-n', 'SSp-bfd', 'SSp-ll', 'SSp-m', 'SSp-ul', 'SSp-tr', 'SSp-un'],
#            'VIS':['VISal', 'VISam', 'VISl', 'VISp', 'VISpl', 'VISpm', 'VISli', 'VISpor', 'VISa', 'VISrl'],
#            'AUD':['AUDd', 'AUDp', 'AUDpo', 'AUDv'],
#            'AI':['AId', 'AIp', 'AIv']
#        }
        
#        for sname, regions in region_dict.items():
#            for disp_type in ('r316', 'l'):
#                cortical_separability(mefile, regions, sname, disp_type)

#    if 0:
#        regions = ['FRP', 'MOp', 'SSp-n', 'SSs', 'GU', 'AUDd', 'VISp', 'ACAd', 'PL', 'ILA', 'ORBl', 'AId',
#                'RSPd', 'TEa', 'PERI', 'ECT']
#        regions = ['SSp']
#        cortical_separability(mefile, regions, sname='r671', disp_type='r671')

    if 1:
        metric = 'euclidean'
        ds_file1 = f'dsmean_{metric}.csv'
        ds_file2 = f'dsstd_{metric}.csv'
        dsmatrix_of_all(mefile, ds_file1, ds_file2, metric=metric)

    if 1:
        dsmean_file = 'dsmean_euclidean.csv'
        parc_file = '../../../intermediate_data/parc_full.nrrd'
        analyze_dsmatrix_vs_parcellations(dsmean_file, parc_file)
    
    if 1:
        dsmean_file = 'dsstd_euclidean.csv'
        parc_file = '../../../intermediate_data/parc_full.nrrd'
        analyze_dsmatrix_vs_parcellations(dsmean_file, parc_file)

