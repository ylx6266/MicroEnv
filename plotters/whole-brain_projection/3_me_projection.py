##########################################################
#Author:          Yufeng Liu
#Create time:     2024-08-21
#Description:               
##########################################################
import os, glob
import random
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.spatial import distance


from anatomy.anatomy_config import MASK_CCF25_FILE, TEST_REGION, \
                                   BSTRUCTS4, BSTRUCTS7, BSTRUCTS13
from anatomy.anatomy_core import parse_ana_tree, get_struct_from_id_path
from projection import Projection
from swc_handler import get_soma_from_swc
from file_io import load_image


sys.path.append('../../../')
from config import get_me_ccf_mapper

import sys
sys.setrecursionlimit(20000)

def get_dataset(dataset, axon_dir):
    datasets = {
        'test': '',
        'test_1um': 'test_1um',
        'test_0_15um': 'test_0_15um',
        'test_0_5um': 'test_0_5um',
        'test_1_5um': 'test_1_5um',
        'hip': 'ion_hip_2um',
        'pfc': 'ion_pfc_2um',
        'mouselight': 'mouselight_2um',
        'seuallen': 'seu-allen1876_2um',
    }
    if dataset == 'all':
        axon_files = []
        for d, v in datasets.items():
            axon_files.extend(sorted(glob.glob(os.path.join(axon_dir, v, '*.swc'))))
    else:
        axon_files = sorted(glob.glob(os.path.join(axon_dir, datasets[dataset], '*.swc')))
    return axon_files


def get_nearest_coordinate(x, y, z, me_atlas):
    valid_coords = np.array(np.where(me_atlas != 0)).T
    
    if len(valid_coords) == 0:
        return 0
    
    distances = distance.cdist([(x, y, z)], valid_coords, 'euclidean')
    min_distance = distances.min()
    
    if min_distance < 2:
        nearest_idx = np.argmin(distances)
        nearest_coord = valid_coords[nearest_idx]
        x, y, z = nearest_coord
        rid_me = me_atlas[x, y, z]
        return rid_me
    else:
        return 0


#def get_nearest_coordinate(x, y, z, me_atlas):
#    valid_coords = np.array(np.where(me_atlas != 0)).T  # Get the coordinates of non-zero voxels
    
#    if len(valid_coords) == 0:
#        return None  
#    distances = distance.cdist([(x, y, z)], valid_coords, 'euclidean')
#    nearest_idx = np.argmin(distances)
#    nearest_coord = valid_coords[nearest_idx]
#    rid_me = me_atlas[tuple(nearest_coord)]  

#    return rid_me


def get_meta_information(axon_dir, dataset, me_atlas_file, me2ccf_file, meta_file):
    # load the ccf-me atlas
    me_atlas = load_image(me_atlas_file)
    
    me2ccf, ccf2me = get_me_ccf_mapper(me2ccf_file)

    # get the names
    ana_tree = parse_ana_tree()
    #bstructs7 = set(list(BSTRUCTS7))
    bstructs13 = set(list(BSTRUCTS13))

    axon_files = get_dataset(dataset, axon_dir)
    df_meta = []
    fns = []
    nfiles = 0
    for axon_file in axon_files:
        #print(axon_file)
        filename = os.path.split(axon_file)[-1][:-4]
        try:
            soma = get_soma_from_swc(axon_file)
        except:
            continue
        sloc = np.array(list(map(float, soma[2:5])))
        #x, y, z = np.floor(sloc / 2000. - ([89.13698204753916/2, 56.428678169921284/2, 8.691020752748784/2])).astype(int) 
        x, y, z = np.floor(sloc / 1000. + [-85.877, -63.046, 3.416]).astype(int) 
        #########################
        try:
            rid_me = me_atlas[x,y,z]        # !!!
#            if rid_me == 0:
#                rid_me = get_nearest_coordinate(x, y, z, me_atlas)  
        except IndexError:
            rid_me = 0
            
            
        #print(rid_me)
        ########################
         #map to ccf-atlas
        if rid_me in me2ccf:
            rid_ccf = me2ccf[rid_me]
            rname_ccf = ana_tree[rid_ccf]['acronym']
            rname_me = f'{rname_ccf}-R{rid_me-min(ccf2me[rid_ccf])+1}'
            # get the brain structure
            id_path = ana_tree[rid_ccf]['structure_id_path']
            #sid7 = get_struct_from_id_path(id_path, BSTRUCTS7)
            sid13 = get_struct_from_id_path(id_path, BSTRUCTS13)
            #sname7 = ana_tree[sid7]['acronym']
            sname13 = ana_tree[sid13]['acronym']

            df_meta.append([sloc[0], sloc[1], sloc[2], rid_ccf, rname_ccf, rid_me, 
                            rname_me, sid13, sname13])
        else:
            df_meta.append([sloc[0], sloc[1], sloc[2], 0, '', 0, '', 0, ''])
        fns.append(filename)
        
        nfiles += 1
        if nfiles % 10 == 0:
            print(nfiles)

    df_meta = pd.DataFrame(df_meta, index=fns, columns=['x','y','z','region_id_ccf', 'region_name_ccf',
                           'region_id_me', 'region_name_me', 'struct13_id', 'struct13_name'])
    df_meta.to_csv(meta_file, index=True)
 
def preprocess_proj(proj_file, thresh, normalize=False, remove_empty_regions=True, 
                    is_me=False, me2ccf=None, ccf2me=None, ana_tree=None, meta=None, ccf_regions=None,
                    keep_structures=None):
    projs = pd.read_csv(proj_file, index_col=0)
    #print(projs)
    projs = projs[projs.index.isin(meta.index)] # remove neurons according meta information

    projs.columns = projs.columns.astype(int)
    if not is_me:
        # for ME projection, we use the extracted regions for CCF projection directly, no need and incorrect
        # to independently thresholding like this.
        projs[projs < thresh] = 0
    else:
        projs[projs < thresh/4] = 0
        
    # to log space
    log_projs = np.log(projs+1)
    #print(log_projs)
    # keep only projection in salient regions
    col_mapper = {}
    skeys = []
    for rid_ in log_projs.columns:
        if rid_ < 0:
            rid = -rid_
            name_prefix = ''
        else:
            rid = rid_
            name_prefix = ''
        
        if is_me:
            rid_ccf = me2ccf[rid]
            rname = f'{name_prefix}{ana_tree[rid_ccf]["acronym"]}-R{rid-min(ccf2me[rid_ccf])+1}'
        else:
            rid_ccf = rid
            rname = f'{name_prefix}{ana_tree[rid_ccf]["acronym"]}'
        
        if rid_ccf not in TEST_REGION:
            # discard regions not int salient regions, or root
            continue

        # get the brain structures
        id_path = ana_tree[rid_ccf]['structure_id_path']
        sid13 = get_struct_from_id_path(id_path, BSTRUCTS13)
        if sid13 == 0:
            # this is unclassified top granularity of brain regions, just skip
            continue
        sname13 = ana_tree[sid13]['acronym']
        skeys.append(f'{sname13}*{rname}')

        col_mapper[rid_] = rname
    
    # extract and rename
    log_projs = log_projs[log_projs.columns[log_projs.columns.isin(col_mapper.keys())]].rename(columns=col_mapper)
    
    # sort by structure
    log_projs = log_projs.iloc[:, np.argsort(skeys)]
    bstructs = np.array([skey.split('*')[0] for skey in skeys])[np.argsort(skeys)]
    
    # remove zeroing neurons
    if normalize:
        log_projs = log_projs / log_projs.sum(axis=1).values.reshape(-1,1)
    # remove zero columns
    if not is_me:
        if remove_empty_regions:
            col_mask = log_projs.sum() != 0
            log_projs = log_projs[log_projs.columns[col_mask]]
            
            bstructs = bstructs[col_mask]
            print("Test!!!")
            print(col_mask)
        # keep only permitted regions in `keep_structures`
        nzbs = np.nonzero(pd.Series(bstructs).isin(keep_structures))[0]
        
        print(nzbs)
        log_projs = log_projs.iloc[:,nzbs]
        
        print(log_projs)
        bstructs = bstructs[nzbs]
        print()
        
    else:
        # keep only the regions kept in CCF regions
        resv_regs = []
        for rn in log_projs.columns:
            ccf_rn = '-'.join(rn.split('-')[:-1])
            if ccf_rn in ccf_regions:
                resv_regs.append(True)
            else:
                resv_regs.append(False)
        col_mask = np.array(resv_regs)
        log_projs = log_projs[log_projs.columns[col_mask]]
        bstructs = bstructs[col_mask]

    # map the index to name

    return log_projs, bstructs
    
# NOT working!
def cbar_for_row_colors(g, uniq_cls, cm_name, loc='center left', bbox_to_anchor=(1,0.5)):
    cmap = mpl.colormaps.get_cmap(cm_name)
    norm = plt.Normalize(vmin=0, vmax=len(uniq_cls))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # remove the axis from the colorbar
    sm.set_array([])
    # Add colorbar to the plot
    #import ipdb; ipdb.set_trace()
    g.cax.figure.colorbar(sm, ax=g.ax_heatmap, orientation='vertical', ticks=range(len(uniq_cls)))
    # Adjusting the colorbar labels
    #g.cax.yaxis.set_tick_params(labelsize=10)
    g.cax.set_yticklabels(uniq_cls)

def get_region_map(regs, all_subregs, diff_hemisphere=False):
    subregs = []
    if diff_hemisphere:
        regs = [hemi+rn for hemi in ('ipsi_', 'contra_') for rn in regs]
        
    for cur_subreg in all_subregs:
        if '-'.join(cur_subreg.split('-')[:-1]) in regs:                   #'-'.join(cur_subreg.split('-')[:-1]) in regs
            subregs.append(cur_subreg)
    return subregs

def analyze_proj(proj_ccf_file, proj_me_file, meta_file, me2ccf_file, thresh=100, min_neurons=10):
    random.seed(1024)
    np.random.seed(1024)

    print('load the meta data...')
    meta = pd.read_csv(meta_file, index_col=0,)
    # skip neurons in regions containing few neurons
    meta = meta[~meta.region_name_ccf.isna()]
    reg_names, reg_cnts = np.unique(meta.region_name_ccf, return_counts=True)
    print(reg_cnts)
    keep_names = reg_names[reg_cnts > min_neurons]
    print(keep_names)
    meta = meta[meta.region_name_ccf.isin(keep_names)]
    

    print('load the tree ontology')
    ana_tree = parse_ana_tree()
    print('load the me2ccf mapping file')
    me2ccf, ccf2me = get_me_ccf_mapper(me2ccf_file)
    
    print('load the projection on CCF space data...')
    proj_ccf, bstructs_ccf = preprocess_proj(proj_ccf_file, thresh, me2ccf=me2ccf, ccf2me=ccf2me, 
                                             is_me=False, ana_tree=ana_tree, meta=meta,  
                                             keep_structures=('FB','EB','PB','NO','AMMC_R','AMMC_L','FLA_R','FLA_L','CAN_R','CAN_L','PRW','SAD','GNG','AL_R','AL_L','LH_R','LH_L','MB_CA_R','MB_CA_L','MB_PED_R',
 'MB_PED_L','MB_VL_R','MB_VL_L','MB_ML_R','MB_ML_L','BU_R','BU_L','GA_R','GA_L','LAL_R','LAL_L','SLP_R','SLP_L','SIP_R','SIP_L','SMP_R','SMP_L',
'CRE_R','CRE_L','SCL_R','SCL_L','ICL_R','ICL_L','IB_R','IB_L','ATL_R','ATL_L','VES_R','VES_L','EPA_R','EPA_L','GOR_R','GOR_L','SPS_R','SPS_L',
'IPS_R','IPS_L','AOTU_R','AOTU_L','AVLP_R','AVLP_L','PVLP_R','PVLP_L','PLP_R','PLP_L','WED_R','WED_L','ME_R','ME_L','AME_R','AME_L','LO_R','LO_L',
'LOP_R','LOP_L','LA_R','LA_L','OCG'                                             
))
    desired_columns2 = [
    'AL_L', 'AL_R', 'AME_L', 'AME_R', 'AMMC_L', 'AMMC_R', 'AOTU_L', 'AOTU_R', 
    'ATL_L', 'ATL_R', 'AVLP_L', 'AVLP_R', 'BU_L', 'BU_R', 'CAN_L', 'CAN_R', 
    'CRE_L', 'CRE_R', 'EPA_L', 'EPA_R', 'FB', 'FLA_L', 'FLA_R', 'GA_L', 'GA_R', 
    'GNG', 'GOR_L', 'GOR_R', 'IB_L', 'IB_R', 'ICL_L', 'ICL_R', 'IPS_L', 'IPS_R', 
    'LAL_L', 'LAL_R', 'LA_L', 'LA_R', 'LH_L', 'LH_R', 'LOP_L', 'LOP_R', 'LO_L', 
    'LO_R', 'MB_CA_L', 'MB_CA_R', 'MB_ML_L', 'MB_ML_R', 'MB_PED_L', 'MB_PED_R', 
    'MB_VL_L', 'MB_VL_R', 'ME_L', 'ME_R', 'NO', 'OCG', 'PB', 'PLP_L', 'PLP_R', 
    'PRW', 'PVLP_L', 'PVLP_R', 'SAD', 'SCL_L', 'SCL_R', 'SIP_L', 'SIP_R', 'SLP_L', 
    'SLP_R', 'SMP_L', 'SMP_R', 'SPS_L', 'SPS_R', 'VES_L', 'VES_R', 'WED_L', 'WED_R'
    ]
    
    desired_columns = ['LA_L','LA_R','ME_L','ME_R','LO_L','LO_R','LOP_L','LOP_R','PLP_L','PLP_R','PVLP_L','PVLP_R','AOTU_L','AOTU_R']
    current_columns = proj_ccf.columns
    missing_columns = [col for col in desired_columns if col not in current_columns]
    for col in missing_columns:
        proj_ccf[col] = 0
    proj_ccf = proj_ccf[desired_columns]
    bstructs_ccf = desired_columns
#'FB','EB','PB','NO','AMMC_R','AMMC_L','FLA_R','FLA_L','CAN_R','CAN_L','PRW','SAD','GNG','AL_R','AL_L','LH_R','LH_L','MB_CA_R','MB_CA_L','MB_PED_R',
# 'MB_PED_L','MB_VL_R','MB_VL_L','MB_ML_R','MB_ML_L','BU_R','BU_L','GA_R','GA_L','LAL_R','LAL_L','SLP_R','SLP_L','SIP_R','SIP_L','SMP_R','SMP_L',
#'CRE_R','CRE_L','SCL_R','SCL_L','ICL_R','ICL_L','IB_R','IB_L','ATL_R','ATL_L','VES_R','VES_L','EPA_R','EPA_L','GOR_R','GOR_L','SPS_R','SPS_L',
#'IPS_R','IPS_L','AOTU_R','AOTU_L','AVLP_R','AVLP_L','PVLP_R','PVLP_L','PLP_R','PLP_L','WED_R','WED_L','ME_R','ME_L','AME_R','AME_L','LO_R','LO_L',
#'LOP_R','LOP_L','LA_R','LA_L','OCG'                                             
    print('load the projection on CCF-ME space data...')
    proj_me, bstructs_me = preprocess_proj(proj_me_file, thresh, me2ccf=me2ccf, ccf2me=ccf2me, 
                                            is_me=True, ana_tree=ana_tree, meta=meta, ccf_regions=proj_ccf.columns)
    #print(np.unique(bstructs_me))
    print(proj_me)

    if 0:
        print(f'Visualize projection matrix of neurons...')
        # col_colors
        uniq_bs = np.unique(bstructs_ccf)
        lut_col = {bs: plt.cm.rainbow(each, bytes=False)
                  for bs, each in zip(uniq_bs, np.linspace(0, 1, len(uniq_bs)))}
        col_colors_ccf = np.array([lut_col[bs] for bs in bstructs_ccf])
        # row colors
        regnames = meta.region_name_ccf.loc[proj_ccf.index]
        uniq_rn = np.unique(regnames)
        lut_row = {rn: plt.cm.rainbow(each, bytes=False)
                  for rn, each in zip(uniq_rn, np.linspace(0, 1, len(uniq_rn)))}
        row_colors_ccf = np.array([lut_row[rn] for rn in regnames])
        cmap = plt.get_cmap('bwr')
        g1 = sns.clustermap(proj_ccf, col_cluster=False, col_colors=col_colors_ccf, row_colors=row_colors_ccf, 
                            cmap=cmap)
        plt.savefig('proj_ccf_neurons.png', dpi=300)
        plt.close()

        # for me
        reordered_ind = g1.dendrogram_row.reordered_ind
        proj_me = proj_me.iloc[reordered_ind]
        col_colors_me = np.array([lut_col[bs] for bs in bstructs_me])
        row_colors_me = row_colors_ccf[reordered_ind]
        g2 = sns.clustermap(proj_me, col_cluster=False, col_colors=col_colors_me, row_colors=row_colors_me,
                            row_cluster=False, cmap=cmap)
        plt.savefig('proj_me_neurons.png', dpi=300)
        plt.close()

    
    # group by regions
    proj_ccf_r = proj_ccf.copy()
    proj_ccf_r['region_ccf'] = meta.region_name_ccf.loc[proj_ccf_r.index]
    proj_ccf_r = proj_ccf_r.groupby('region_ccf').mean()
    # for me
    proj_me_r = proj_me.copy()
    proj_me_r['region_me'] = meta.region_name_me.loc[proj_me_r.index]
    proj_me_r = proj_me_r.groupby('region_me').mean()
    print(proj_ccf_r.shape, proj_me_r.shape)
    #print(proj_ccf_r.index)
    #print(proj_me_r)
    #import ipdb; ipdb.set_trace()

    if 1:
        print(f'Visualize projection matrix of regions')
        cmap = plt.get_cmap('bwr')
        cmap = mpl.colors.ListedColormap(cmap(np.linspace(0.3, 1., 256)))

        # col_colors
        uniq_bs = np.unique(bstructs_ccf)
        rnd_ids = np.arange(len(uniq_bs))
        random.shuffle(rnd_ids)
        uniq_bs_rnd = uniq_bs[rnd_ids]
        print(uniq_bs, uniq_bs_rnd)
        lut_col = {bs: plt.cm.rainbow(each, bytes=False)
                  for bs, each in zip(uniq_bs_rnd, np.linspace(0, 1, len(uniq_bs_rnd)))}
        col_colors_ccf = np.array([lut_col[bs] for bs in bstructs_ccf])
        
        # plotting
        g1 = sns.clustermap(proj_ccf_r, cmap=cmap, col_cluster=False, col_colors=col_colors_ccf, 
                            row_cluster=False, yticklabels=1, vmin=0, vmax=10, figsize=(20,20),
                            xticklabels=1, cbar_pos={0.08,0.05,0.02,0.15})
        #cbar_for_row_colors(g1, uniq_bs, cm_name='rainbow')
        proj_ccf_r.to_csv('proj_ccf_r.csv', index=True)
        plt.savefig('proj_ccf_regions.png', dpi=300)
        plt.close()

        col_colors_me = np.array([lut_col[bs] for bs in bstructs_me])
        g2 = sns.clustermap(proj_me_r, cmap=cmap, col_cluster=False, row_cluster=False, 
                            col_colors=col_colors_me, yticklabels=1, vmin=0, vmax=11, 
                            figsize=(100,100), xticklabels=1)
        g2.ax_heatmap.tick_params(left=True, right=False, labelleft=True, labelright=False)
        g2.ax_heatmap.set_ylabel('Source subregions', fontsize=18)
        g2.ax_heatmap.yaxis.set_label_position("left")
        g2.ax_heatmap.set_xlabel('Target subregions', fontsize=18)
        plt.setp(g2.ax_heatmap.get_xticklabels(), rotation=45, rotation_mode='anchor',
                     ha='right')

        #g2.cax.set_position([0.05,0.05,0.03,0.15]) # why not working?!
        g2.cax.set_visible(False)
        plt.subplots_adjust(bottom=0.18)
        plt.savefig('proj_me_regions.png', dpi=300)
        plt.close()

        ###### turn off  this part ########
        if 0:
            # Analyze the divergence of source, target and source-target suregions
            src_cc_means = []
            ipsi_cc_means = []
            contra_cc_means = []
            st_cc_means = []
            for reg in proj_ccf_r.index:
                src_regs = get_region_map([reg], proj_me_r.index, diff_hemisphere=False)
                if len(src_regs) == 1:
                    continue
                # source divergence
                corrs_src = proj_me_r.loc[src_regs].transpose().corr()
                src_cc_mean = corrs_src.values[np.triu_indices_from(corrs_src, k=1)].mean()
                src_cc_means.append(src_cc_mean)
            
            # target
            for reg in proj_ccf_r.columns:
                tgt_regs = get_region_map([reg], proj_me_r.columns, diff_hemisphere=False)
                if len(tgt_regs) == 1:
                    continue
                
                if tgt_regs[0].startswith('ipsi'):
                    corrs_ipsi = proj_me_r.loc[:,tgt_regs].corr()
                    corrs_ipsi.fillna(0, inplace=True)
                    ipsi_cc_mean = corrs_ipsi.values[np.triu_indices_from(corrs_ipsi, k=1)].mean()
                    ipsi_cc_means.append(ipsi_cc_mean)
                else:
                    corrs_contra = proj_me_r.loc[:,tgt_regs].corr()
                    corrs_contra.fillna(0, inplace=True)
                    contra_cc_mean = corrs_contra.values[np.triu_indices_from(corrs_contra, k=1)].mean()
                    contra_cc_means.append(contra_cc_mean)
                # source-target pairs
                #for reg_s in proj_ccf_r.index:
                #    st_regs = get_region_map([reg_s], proj_me_r.index, diff_hemisphere=False)
                #    if len(st_regs) == 1:
                #        continue

                #    corrs_st = proj_me_r.loc[st_regs, tgt_regs].transpose().corr()
                #    st_cc_mean = corrs_st.values[np.triu_indices_from(corrs_st, k=1)].mean()
                #    st_cc_means.append(st_cc_mean)

                
            sns.set_theme(style='ticks', font_scale=1.5)
            # visualization
            df_corrs = pd.DataFrame({
                'Source': pd.Series(src_cc_means),
                'Target-ipsi': pd.Series(ipsi_cc_means),
                'Target-contra': pd.Series(contra_cc_means)
            })
            plt.figure(figsize=(6,6))
            sns.stripplot(data=df_corrs, alpha=0.75, legend=False)
            ax_pp = sns.pointplot(data=df_corrs, linestyle="none", errorbar=None, marker='_', 
                                  markersize=20, markeredgewidth=3, color='red')
            # annotate
            avg_ccs = df_corrs.mean()
            for xc, yc in zip(range(df_corrs.shape[0]), avg_ccs):
                # The first container is the axis, skip
                txt = f'{yc:.2f}'
                ax_pp.text(xc+0.3, yc, txt, ha='center', va='center', color='red')

            plt.ylabel("Correlation between projections of subregions")
            plt.savefig('correlations_between_subregions.png', dpi=300)
            plt.close()

            print()    

        
    if 0:
        cmap = plt.get_cmap('Reds')
        cmap = mpl.colors.ListedColormap(cmap(np.linspace(0, 0.8, 256)))
        #cmap = plt.get_cmap('bwr')
        #cmap = mpl.colors.ListedColormap(cmap(np.linspace(0.3, 1., 256)))

        print(f'--> Projection heatmap for subregions')
        r_src, r_tgts = 'AVLP_R', ['AVLP_R']
        #r_src, r_tgts = 'PRE', ['ENTm2', 'ENTm3', 'PAR']
        #r_tgts = [hemi+rn for hemi in ('contra_', 'ipsi_') for rn in r_tgts]    #changed
        
        #proj_ccf_s = proj_ccf_r.loc[r_src, r_tgts]
        #g3 = sns.clustermap(proj_ccf_s, cmap=cmap, col_cluster=False, row_cluster=False, yticklabels=1, 
        #               xticklabels=1, vmin=0, vmax=11)
        #plt.savefig(f'proj_ccf_{r_src}.png', dpi=300)
        #plt.close()
        
        #CA3s = ['CA3-R7', 'CA3-R2', 'CA3-R6', 'CA3-R11', 'CA3-R4', 'CA3-R3', 'CA3-R10', 'CA3-R1', 'CA3-R12'
        #        'CA3-R8', 'CA3-R9', 'CA3-R5'] 
        ME_L = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] 
        ME_R = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        LOP_L = [0,1,2,3,4,5,6,7,8,9,10,11]
        LOP_R = [0,1,2,3,4,5,6,7,8,9,10,11,12]
        LO_L = [0,1,2,3,4,5,6,7,8,9,10]
        LO_R = [0,1,2,3,4,5,6,7,8]
        LAL_L = [0]
        LAL_R = [0]
        AL_L = [0,1,2,3,4,5,6,7,8,9,10,11,12]
        AL_R = [0,1,2,3,4,5,6,7,8,9]
        MB_CA_R = [0,1,2,3,4,5]
        MB_CA_L = [0,1,2,3,4,5]
        MB_VL_R = [0,1,2,3,4]
        MB_VL_L = [0,1,2,3]
        MB_PED_L = [0,1,2]
        MB_PED_R = [0]
        LA_R = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        LA_L = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        AMMC_L = [0,1,2,3,4,5,6,7,8]
        AMMC_L2 = [0,1,2,3,4]
        AMMC_R = [0,1,2,3,4,5]
        WED_L = [0,1,2,3,4,5,6]
        WED_R = [0,1,2,3,4,5,6]
        AVLP_L = [0,1,2,3,4,5,6,7,8]
        AVLP_R = [0,1,2,3,4,5,6,7,8,9]
        
        #PVLPs = [1]
        for r_tgt in r_tgts:
            # for me
            #print(r_tgt)
            rmes = np.array(get_region_map([r_src], proj_me_r.index, diff_hemisphere=False))
            #print(rmes)
            tgts_me = np.array(get_region_map([r_tgt], proj_me_r.columns, diff_hemisphere=False))
#            if r_src == 'LOP_L':
#                rmes = rmes[LOP_L]
#            elif r_src == 'ME_L':
#                rmes = rmes[ME_L]
#            elif r_src == 'ME_R':
#                rmes = rmes[ME_R]
#            elif r_src == 'LOP_R':
#                rmes = rmes[LOP_R]
#            elif r_src == 'LO_R':
#                rmes = rmes[LO_R]
#            elif r_src == 'LO_L':
#                rmes = rmes[LO_L]
#            elif r_src == 'LAL_L':
#                rmes = rmes[LAL_L]
#            elif r_src == 'LAL_R':
#                rmes = rmes[LAL_R]
#            elif r_src == 'MB_CA_R':
#                rmes = rmes[MB_CA_R]
#            elif r_src == 'MB_CA_L':
#                rmes = rmes[MB_CA_L]
#            elif r_src == 'LA_L':
#                rmes = rmes[LA_L]
#            elif r_src == 'LA_R':
#                rmes = rmes[LA_R]
#            elif r_src == 'AVLP_R':
#                rmes = rmes[AVLP_R]
#            elif r_src == 'AVLP_L':
#                rmes = rmes[AVLP_L]
#            elif r_src == 'AMMC_R':
#                rmes = rmes[AMMC_R]
#            elif r_src == 'AMMC_L':
#                rmes = rmes[AMMC_L2]
                
#            if 'ME_L' in r_tgt:
#                tgts_me = tgts_me[ME_L]
#            elif 'LOP_L' in r_tgt:
#                tgts_me = tgts_me[LOP_L]
#            elif 'ME_R' in r_tgt:
#                tgts_me = tgts_me[ME_R]
#            elif 'LOP_R' in r_tgt:
#                tgts_me = tgts_me[LOP_R]
#            elif 'LO_R' in r_tgt:
#                tgts_me = tgts_me[LO_R]
#            elif 'LO_L' in r_tgt:
#                tgts_me = tgts_me[LO_L]
#            elif 'AL_L' in r_tgt:
#                tgts_me = tgts_me[AL_L]
#            elif 'AL_R' in r_tgt:
#                tgts_me = tgts_me[AL_R]
#            elif 'MB_CA_R' in r_tgt:
#                tgts_me = tgts_me[MB_CA_R]
#            elif 'MB_CA_L' in r_tgt:
#                tgts_me = tgts_me[MB_CA_L]
#            elif 'MB_VL_R' in r_tgt:
#                tgts_me = tgts_me[MB_VL_R]
#            elif 'MB_VL_L' in r_tgt:
#                tgts_me = tgts_me[MB_VL_L]
#            elif 'MB_PED_R' in r_tgt:
#                tgts_me = tgts_me[MB_PED_R]
                #print(tgts_me)
#            elif 'MB_PED_L' in r_tgt:
#                tgts_me = tgts_me[MB_PED_L]
#            elif 'AVLP_L' in r_tgt:
#                tgts_me = tgts_me[AVLP_L]
#            elif 'AVLP_R' in r_tgt:
#                tgts_me = tgts_me[AVLP_R]
#            elif 'AMMC_L' in r_tgt:
#                tgts_me = tgts_me[AMMC_L]
#            elif 'AMMC_R' in r_tgt:
#                tgts_me = tgts_me[AMMC_R]

            proj_me_s = proj_me_r.loc[rmes, tgts_me].transpose()
            g4 = sns.clustermap(proj_me_s, cmap=cmap, col_cluster=False, row_cluster=False, 
                           yticklabels=1, xticklabels=1, vmin=0, vmax=11)
            ax4 = g4.ax_heatmap
            
            # mark the high projection subregions
            markers = proj_me_s.copy(); markers.iloc[:,:] = 0
            for isubg, subg in enumerate(proj_me_s.columns):
                if proj_me_s.shape[0] > 1:  
                    top2i = np.argpartition(proj_me_s.iloc[:, isubg], -2)[-2:]
                    i_high = np.nonzero(proj_me_s.iloc[top2i, isubg] > 1)[0]
                    if i_high.shape[0] > 0:
                        markers.iloc[top2i[i_high], isubg] = 1
                else:
                    markers.iloc[0, isubg] = 1
                   
            ys, xs = np.nonzero(markers)
            ax4.plot(xs+0.5, ys+0.5, 'ko', markersize=8)

            # customize the ticks and labels
            tick_font = 16
            label_font = 22
            g4.ax_heatmap.tick_params(axis='y', labelsize=tick_font, rotation=0, direction='in')
            g4.ax_heatmap.tick_params(axis='x', direction='in')
            plt.setp(g4.ax_heatmap.get_xticklabels(), rotation=45, rotation_mode='anchor', 
                     ha='right', fontsize=tick_font)
            g4.ax_heatmap.set_xlabel('Source subregions', fontsize=label_font)
            g4.ax_heatmap.set_ylabel(f'Target subregions', fontsize=label_font)
            g4.ax_heatmap.set_aspect('equal')

            plt.subplots_adjust(left=0, right=0.75, bottom=0.13)
            
            # configuring the colorbar
            g4.cax.set_position([0.02, 0.1, 0.03, 0.15])
            g4.cax.tick_params(axis='y', labelsize=tick_font)
            g4.cax.set_ylabel(r'$ln(L+1)$', fontsize=label_font)
            
            plt.savefig(f'proj_me_{r_src}_{r_tgt}', dpi=300)
            plt.close()
            #import ipdb; ipdb.set_trace()


if __name__ == '__main__':


    atlas_file = None 
    me_atlas_file = '../../../intermediate_data/parc_full.nrrd'
    me2ccf_file = '../../../intermediate_data/parc_full.nrrd.pkl'
    axon_dir = '/home/ylx/fly/axon_file'
    axon_dir_with_projected_soma = '/home/ylx/fly/axon_file/axon_sink_soma'
    dataset = 'test'

    if dataset == 'all':
        me_proj_csv = './data/proj_ccf-me.csv'
        proj_csv = './data/proj.csv'
    else:
        me_proj_csv = f'./data/proj_ccf-me_{dataset}.csv'
        proj_csv = f'./data/proj_{dataset}.csv'
    meta_file = f'./data/meta_{dataset}.csv'

    if 0:
        # generate the projection matrix
        use_me = False
        axon_files = get_dataset(dataset, axon_dir)

        if use_me:
            PJ = Projection(resample_scale=1000., atlas_file=me_atlas_file)
            PJ.calc_proj_matrix(axon_files, proj_csv=me_proj_csv)
        else:
            PJ = Projection(resample_scale=1000., atlas_file=atlas_file)
            PJ.calc_proj_matrix(axon_files, proj_csv=proj_csv)

    if 0:
        # get the meta information for the neurons
        #get_meta_information(axon_dir, dataset, me_atlas_file, me2ccf_file, meta_file)
        get_meta_information(axon_dir_with_projected_soma, dataset, me_atlas_file, me2ccf_file, meta_file)
        
    if 1:
        proj_csv = './data/filter_data/proj_multipolar.csv'
        me_proj_csv = './data/filter_data/me_proj_multipolar.csv'
        analyze_proj(proj_csv, me_proj_csv, meta_file, me2ccf_file, thresh=1000, min_neurons=20)

