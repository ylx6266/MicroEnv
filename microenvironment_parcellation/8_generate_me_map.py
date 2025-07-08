#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : generate_me_map.py
#   Author       : Yufeng Liu
#   Date         : 2023-02-09
#   Description  : 
#
#================================================================
import os
import sys
import random
import numpy as np
import numbers
import pickle
import pandas as pd
from skimage import exposure, filters, measure
from skimage import morphology
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from scipy.optimize import curve_fit
from scipy.spatial import distance_matrix
import matplotlib
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from fil_finder import FilFinder2D
#import astropy.units as u
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import umap

from image_utils import get_mip_image, image_histeq
from file_io import load_image, save_image
from anatomy.anatomy_config import MASK_CCF25_FILE, MASK_CCF25_R314_FILE, ANATOMY_TREE_FILE
from anatomy.anatomy_vis import get_brain_outline2d, get_section_boundary_with_outline, \
                                get_brain_mask2d, get_section_boundary, detect_edges2d
from anatomy.anatomy_core import parse_ana_tree
from plotters.neurite_arbors import NeuriteArbors

from config import mRMR_f3 as __MAP_FEATS__, standardize_features, __Subregion_id__

#if __name__ == '__main__':
    # customize the fonts
    #plt.rcParams['font.family'] = 'Helvetica'
    #plt.rcParams['font.weight'] = 'light'


# features selected by mRMR
__MAP_FEATS__ = ('AverageFragmentation', 'pca_vr1', 'Stems')  

def process_features(mefile, scale=25., with_comment=False, use_me_feature=True):
    if with_comment:
        df = pd.read_csv(mefile, index_col=1, comment='#')
    else:
        df = pd.read_csv(mefile, index_col=1)

    if use_me_feature:
        # Manually rename them to remove the '_me' suffix. This is not an elegant way, but
        # rather a temporal way historically
        df.drop(list(__MAP_FEATS__), axis=1, inplace=True)

        mapper = {}
        for mf in __MAP_FEATS__:
            mapper[f'{mf}_me'] = mf
        df.rename(columns=mapper, inplace=True)
    
    # We would like to use tortuosity, which is  opposite of contraction
    #df.loc[:, 'AverageContraction'] = 1 - df['AverageContraction']

    feat_names = [fn for fn in __MAP_FEATS__]
    # scaling the coordinates to CCFv3-25um space
    df['soma_x'] /= scale
    df['soma_y'] /= scale
    df['soma_z'] /= scale
    # we should remove the out-of-region coordinates
    xdim,ydim,zdim = (808, 383, 275)   # dimensions for fly atlas
    in_region = (df['soma_x'] >= 0) & (df['soma_x'] < xdim) & \
                (df['soma_y'] >= 0) & (df['soma_y'] < ydim) & \
                (df['soma_z'] >= 0) & (df['soma_z'] < zdim)
    df = df[in_region]
    print(f'Filtered out {in_region.shape[0] - df.shape[0]}')

    return df, feat_names

def plot_section_outline(mask, axis=0, sectionX=None, ax=None, with_outline=True, outline_color='orange', b_scale=0.5):
    boundary_mask2d = get_section_boundary(mask, axis=axis, v=1, c=sectionX)
    sh, sw = boundary_mask2d.shape[:2]
    if ax is None:
        fig, ax = plt.subplots()
        brain_mask2d = get_brain_mask2d(mask, axis=axis, v=1)
        im = np.ones((sh, sw, 4), dtype=np.uint8) * 255
        im[~brain_mask2d] = 0#1

    # show boundary
    b_indices = np.where(boundary_mask2d)
    ax.scatter(b_indices[1], b_indices[0], s=b_scale, c='black', alpha=0.5, edgecolors='none')
    # intra-brain regions
        
    if with_outline:
        outline_mask2d = get_brain_outline2d(mask, axis=axis, v=1)
        o_indices = np.where(outline_mask2d)
        ax.scatter(o_indices[1], o_indices[0], s=1.0, c=outline_color, alpha=1.0, edgecolors='none')
    
    if ax is None:
        return fig, ax
    else:
        return ax
    

def process_mip(mip, mask, sectionX=None, axis=0, figname='temp.png', mode='composite', with_outline=True, outline_color='orange', pt_scale=2, b_scale=0.5):
    # get the mask
    brain_mask2d = get_brain_mask2d(mask, axis=axis, v=1)

    #if axis==1: cv2.imwrite('temp.png', mip); sys.exit()
    im = np.ones((mip.shape[0], mip.shape[1], 4), dtype=np.uint8) * 255
    # default size is 6.4 x 4.8
    scale = np.sqrt(np.prod(mip.shape[:2]) / 456 / 320)
    wi, hi = np.round(6.4 * scale, 2), np.round(4.8 * scale, 2)

    fig, ax = plt.subplots(figsize=(wi, hi))
    width, height = fig.get_size_inches() * fig.get_dpi()
    width = int(width)
    height = int(height)

    canvas = FigureCanvas(fig)
    im = ax.imshow(im)
    fig.patch.set_visible(False)
    ax.axis('off')
    
    bg_mask = mip.sum(axis=-1) == 0
    fg_mask = ~bg_mask
    fg_indices = np.where(fg_mask)
    if mode == 'composite':
        fg_values = mip[fg_indices] / 255.
        cmap = None
    elif mode == 'single':  # single feature
        fg_values = mip[fg_indices][:,0] / 255.
        cmap = 'coolwarm'
    elif mode == 'soma':    # soma density
        fg_values = mip[fg_indices] / 255.
        cmap = None
    else:
        raise NotImplementedError
    
    if len(fg_indices[0]) > 0:
        ax.scatter(fg_indices[1], fg_indices[0], c=fg_values, s=pt_scale, edgecolors='none', cmap=cmap)
    plot_section_outline(mask, axis=axis, sectionX=sectionX, ax=ax, with_outline=with_outline, outline_color=outline_color, b_scale=b_scale)

    plt.savefig(figname, dpi=300)
    plt.close('all')

    #canvas.draw()       # draw the canvas, cache the renderer
    #img_buffer = canvas.tostring_rgb()
    #out = np.frombuffer(img_buffer, dtype=np.uint8).reshape(height, width, 3)
    #return out

def get_me_mips(mefile, shape3d, histeq, flip_to_left, mode, findex, axids=(2,), thickX2=20, disp_right_hemi=False, use_me_feature=True):
    df, feat_names = process_features(mefile, use_me_feature=use_me_feature)
    
    c = len(feat_names)
    zdim, ydim, xdim = shape3d
    zdim2, ydim2, xdim2 = zdim//2, ydim//2, xdim//2
    memap = np.zeros((zdim, ydim, xdim, c), dtype=np.uint8)
    xyz = np.floor(df[['soma_x', 'soma_y', 'soma_z']].to_numpy()).astype(np.int32)
    # normalize to uint8
    fvalues = df[feat_names]
    fmin, fmax = fvalues.min(), fvalues.max()
    fvalues = ((fvalues - fmin) / (fmax - fmin) * 255).to_numpy()
    if histeq:
        for i in range(fvalues.shape[1]):
            fvalues[:,i] = image_histeq(fvalues[:,i])[0]
    
    if flip_to_left:
        # flip z-dimension, so that to aggregate the information to left or right hemisphere
        right_hemi_mask = xyz[:,2] < zdim2
        xyz[:,2][right_hemi_mask] = zdim - xyz[:,2][right_hemi_mask]
        # I would also like to show the right hemisphere
        if disp_right_hemi:
            xyz2 = xyz.copy()
            xyz2[:,2] = zdim - xyz2[:,2]
            # concat
            xyz = np.vstack((xyz, xyz2))
            # also for the values
            fvalues = np.vstack((fvalues, fvalues))
    
    debug = False
    if debug: #visualize the distribution of features
        g = sns.histplot(data=fvalues, kde=True)
        plt.savefig('fvalues_distr_histeq.png', dpi=300)
        plt.close('all')

    if mode == 'composite':
        memap[xyz[:,2], xyz[:,1], xyz[:,0]] = fvalues
    elif mode == 'single':
        memap[xyz[:,2], xyz[:,1], xyz[:,0]] = fvalues[:,findex].reshape(-1,1)
    elif mode == 'soma':
        memap[xyz[:,2]-1, xyz[:,1]-1, xyz[:,0]-1] = [255,0,0]
    else:
        raise NotImplementedError
    
    # keep only values near the section plane
    mips = []
    for axid in axids:
        print(f'--> Processing axis: {axid}')
        cur_mips = []
        for sid in range(thickX2, shape3d[axid], 2*thickX2):
            cur_memap = memap.copy()
            if thickX2 != -1:
                if axid == 0:
                    cur_memap[:sid-thickX2] = 0
                    cur_memap[sid+thickX2:] = 0
                elif axid == 1:
                    cur_memap[:,:sid-thickX2] = 0
                    cur_memap[:,sid+thickX2:] = 0
                else:
                    cur_memap[:,:,:sid-thickX2] = 0
                    cur_memap[:,:,sid+thickX2:] = 0
            print(cur_memap.mean(), cur_memap.std())
            
            mip = get_mip_image(cur_memap, axid)
            cur_mips.append(mip)
        mips.append(cur_mips)
    return mips

def generate_me_maps(mefile, outfile, histeq=True, flip_to_left= False, mode='composite', findex=0, fmt='svg', axids=(2,), use_me_feature=True):
    '''
    @param mefile:          file containing microenviron features
    @param outfile:         prefix of output file
    @param histeq:          Whether or not to use histeq to equalize the feature values
    @param flip_to_left:    whether map points at the right hemisphere to left hemisphere
    @param mode:            [composite]: show 3 features; otherwise separate feature
    @param findex:          index of feature to display
    '''
    if mode != 'composite':
        fname = __MAP_FEATS__[findex]
        prefix = f'{outfile}_{fname}'
    else:
        prefix = f'{outfile}'
    
    mask = load_image(MASK_CCF25_FILE)  # x,y,z order! 
    shape3d = mask.shape
    thickX2 = 20
    mips = get_me_mips(mefile, shape3d, histeq, flip_to_left, mode, findex, axids=axids, thickX2=thickX2, use_me_feature=use_me_feature)
    for axid, cur_mips in zip(axids, mips):
        for imip, mip in enumerate(cur_mips):
            figname = f'{prefix}_mip{axid}_{imip:02d}.{fmt}'
            sectionX = thickX2 * (2 * imip + 1)
            process_mip(mip, mask, axis=axid, figname=figname, mode=mode, sectionX=sectionX, with_outline=False)
            if not figname.endswith('svg'):
                # load and remove the zero-alpha block
                img = cv2.imread(figname, cv2.IMREAD_UNCHANGED)
                wnz = np.nonzero(img[img.shape[0]//2,:,-1])[0]
                ws, we = wnz[0], wnz[-1]
                hnz = np.nonzero(img[:,img.shape[1]//2,-1])[0]
                hs, he = hnz[0], hnz[-1]
                img = img[hs:he+1, ws:we+1]
                # set the alpha of non-brain region as 0
                img[img[:,:,-1] == 1] = 0
                if axid != 0:   # rotate 90
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    # concatenate with the sectional atlas for the whole brain
                    outscale = 3
                    atlas2d = colorize_atlas2d_cv2(axid, sectionX, outscale=outscale, annot=False, fmt='png')
                    atlas2d = cv2.resize(atlas2d, img.shape[:2][::-1])
                    img[:,:,:3] = atlas2d[:,:,:3]  # Replace the entire image with atlas2d
                
                cv2.imwrite(figname, img)

       

def colorize_atlas2d_cv2(axid=2, sectionX=420, outscale=3, annot=False, fmt='svg'):
    mask = load_image(MASK_CCF25_FILE)
    ana_dict = parse_ana_tree()
    
    boundaries = get_section_boundary(mask, axis=axid, c=sectionX, v=1)
    section = np.take(mask, sectionX, axid)
    out = np.ones((*section.shape, 3), dtype=np.uint8) * 255
    values = np.unique(section)
    print(f'Dimension of axis={axid} is: {section.shape}, with {len(values)-1} regions')

    if annot:
        centers = []
        rnames = []
        c2 = out.shape[0] // 2
        right_mask = boundaries.copy()
        right_mask.fill(False)
        right_mask[:c2] = True
        for v in values:
            if v == 0: continue
            rname = ana_dict[v]['acronym']

            # center of regions,
            cur_mask = section == v
            out[:,:,:3][cur_mask] = ana_dict[v]['rgb_triplet']

            if rname in ['root', 'fiber tracts']:   # no texting is necessary
                continue
            if axid != 0:
                cur_mask = cur_mask & right_mask #only left hemisphere
            cc = cv2.connectedComponents(cur_mask.astype(np.uint8))
            for i in range(cc[0] - 1):
                cur_mask = cc[1] == (i+1)
                if cur_mask.sum() < 5:
                    continue
                indices = np.where(cur_mask)
                xmean = (indices[0].min() + indices[0].max()) // 2
                ymean = int(np.median(indices[1][indices[0] == xmean]))
                centers.append((xmean, ymean))
                rnames.append(rname)
    else:
        for v in values:
            if v == 0: continue
            cur_mask = section == v
            out[:,:,:3][cur_mask] = ana_dict[v]['rgb_triplet']
    # mixing with boudnary
    alpha = 0.2
    out[:,:,:3][boundaries] = (0 * alpha + out[boundaries][:,:3] * (1 - alpha)).astype(np.uint8)
    #out[:,:,3][boundaries] = int(alpha * 255)

    figname = f'atlas_axis{axid}.{fmt}'
    if outscale != 1:
        out = cv2.resize(out, (0,0), fx=outscale, fy=outscale, interpolation=cv2.INTER_CUBIC)
    # we would like to rotate the image, so that it can be better visualized
    if axid != 0:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)

    so1, so2 = out.shape[:2]
    # annotation if required
    if annot:
        figname = f'atlas_axis{axid}_section{sectionX}_annot.{fmt}'
        shift = 20
        for center, rn in zip(centers, rnames):
            sx, sy = center[1]*outscale, center[0]*outscale
            if axid != 0:
                # rotate accordingly
                new_center = (so2-sy-shift, sx)
            else:
                new_center = (sx-shift, sy)
            cv2.putText(out, rn, new_center, cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 1)

    if figname.endswith('svg'):
        # save to `svg` vectorized file, using plt
        fig, ax = plt.subplots()
        ax.imshow(out)
        fig.patch.set_visible(False)
        ax.axis('off')
        plt.savefig(figname, dpi=300)
        plt.close('all')
    else:
        cv2.imwrite(figname, out)
    
    return out

def find_regional_representative(mefile, region='IC', swcdir='', color='magenta'):
    random.seed(1024)
    df = pd.read_csv(mefile, index_col=0).dropna()
    keys = [f'{key}_me' for key in __MAP_FEATS__]
    #import ipdb; ipdb.set_trace()
    tmp = df[keys]
    dfr = df.copy().dropna()
    dfr[keys] = (tmp - tmp.mean()) / tmp.std()
    # keep only neurons from the target region
    rmask = dfr.region_name_r316 == region
    dfr = dfr[rmask][keys]
    print(f'Number of neurons in region {region}: {dfr.shape[0]}')
    medians = dfr.median()
    # find out the neurons with closest features
    dists2m = np.linalg.norm(dfr - medians, axis=1)
    min_id = dists2m.argmin()
    min_dist = dists2m[min_id]
    min_name = dfr.index[min_id]
    min_brain = df.loc[min_name, 'brain_id']
    print(f'The neuron {min_name} has distance {min_dist:.4f} to the median of the region {region}')
    print(df[rmask][keys].iloc[min_id], min_brain)
    
    plot_morphology = True
    if plot_morphology:
        if 0:
            # plot the neurons
            nsamples = 50
            for swc_name in random.sample(list(df[rmask].index), nsamples):
                brain_id = df.loc[swc_name, 'brain_id']
                swcfile = os.path.join(swcdir, str(brain_id), f'{swc_name}_stps.swc')
                na = NeuriteArbors(swcfile)
                out_name = f'{region}_{brain_id}_{swc_name}'
                na.plot_morph_mip(type_id=None, color=color, figname=out_name, out_dir='.', show_name=False)
        if 1:   # to match previous selection
            #brain_id, swc_name = 191797, '6987_19683_7370'
            brain_id, swc_name = 201584, '8378_11445_10616'
            swcfile = os.path.join(swcdir, str(brain_id), f'{swc_name}_stps.swc')
            na = NeuriteArbors(swcfile)
            out_name  = f'{region}_{brain_id}_{swc_name}'
            na.plot_morph_mip(type_id=None, color=color, figname=out_name, out_dir='.', show_name=False)

def plot_inter_regional_features(mefile, regions=('IC', 'SIM')):
    df = pd.read_csv(mefile, index_col=0).dropna()
    keys = ['region_name_r316'] + [f'{key}_me' for key in __MAP_FEATS__]
    dfr = df[keys][df['region_name_r316'].isin(regions)]
    # axes instance
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # plot
    n1, n2, n3 = keys[-3:]
    dfr1 = dfr[dfr['region_name_r316'] == regions[0]]
    dfr2 = dfr[dfr['region_name_r316'] == regions[1]]
    sc1 = ax.scatter(dfr1[n3], dfr1[n2], dfr1[n1]/1000, s=12, c='magenta', marker='o', alpha=.75, label=regions[0])
    sc2 = ax.scatter(dfr2[n3], dfr2[n2], dfr2[n1]/1000, s=12, c='cyan', marker='o', alpha=1., label=regions[1])

    label_size = 22
    ax.set_xlabel('Avg. Fragmentation', fontsize=label_size, labelpad=8)
    ax.set_ylabel('Avg. Straightness', fontsize=label_size, labelpad=10)
    ax.set_zlabel('Total Length (mm)', fontsize=label_size, labelpad=10)

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_yticks([0.8,0.84,0.88,0.92,0.96])

    # legend
    plt.legend(bbox_to_anchor=(0.6,0.6), fontsize=label_size, markerscale=3., handletextpad=0.2, frameon=False)
    fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    ax.set_box_aspect(None, zoom=0.85)  # to avoid Z label cutoff
    #hide the gridline
    ax.grid(False)
    elev, azim = 30, -15
    ax.view_init(elev, azim)
    # save
    plt.savefig("IC_SIM_features.png", bbox_inches='tight', dpi=300)
    plt.close()

def plot_MOB_features(mefile, rname='MOB', r316=False):
    df = pd.read_csv(mefile, index_col=0).dropna()
    keys = [f'{key}_me' for key in __MAP_FEATS__]
    if r316:
        rkey = 'region_name_r316'
    else:
        rkey = 'soma_region'

    if type(rname) is list:
        dfr = df[keys][df[rkey].isin(rname)]
        out_prefix = 'tmp'
    else:
        dfr = df[keys][df[rkey] == rname]
        out_prefix = rname

    # We handling the coloring
    dfc = dfr.copy()
    for i in range(3):
        tmp = dfc.iloc[:,i]
        dfc.iloc[:,i] = image_histeq(tmp.values)[0] / 255.
    dfc[dfc > 1] = 1.
        
    # axes instance
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # plot
    n1, n2, n3 = keys[-3:]
    print(n1, n2, n3)
    sc = ax.scatter(dfr[n3], dfr[n2], dfr[n1]/1000, s=10, c=dfc.values, marker='o', alpha=.75)
    label_size = 22
    ax.set_xlabel('Fragmentation', fontsize=label_size, labelpad=10)
    ax.set_ylabel('Straightness', fontsize=label_size, labelpad=10)
    ax.set_zlabel('Total Length (mm)', fontsize=label_size, labelpad=10)

    ax.tick_params(axis='both', which='major', labelsize=14)

    # legend
    plt.legend(bbox_to_anchor=(0.6,0.6), fontsize=label_size, markerscale=3., handletextpad=0.2)
    fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    ax.set_box_aspect(None, zoom=0.85)  # to avoid Z label cutoff
    if rname in ['MOB', 'ACB']:
        ax.set_zlim3d(0.5, 2)
    ax.get_legend().remove()
    # Hide grid lines
    ax.grid(False)
    if rname == 'ACB':
        ax.view_init(30, 30)

    # save
    if '/' in rname:
        rname = rname.replace('/', '_')
    plt.savefig(f"{out_prefix}_features.png", bbox_inches='tight')
    plt.close()

def plot_region_feature_in_ccf_space(mefile, rname='ME_R', r316=False, flipLR=False):
    df = pd.read_csv(mefile, index_col=1).dropna()
    keys = [f'{key}_me' for key in __MAP_FEATS__]
    if r316:
        rkey = 'region_name_r316'
    else:
        rkey = 'soma_region'  #region_name_r671

    if type(rname) is list:
        sel_mask = df[rkey].isin(rname)
        out_prefix = 'tmp'
    else:
        sel_mask = df[rkey] == rname
        out_prefix = rname
    
    dfr = df[keys][sel_mask]
    coords = df[['soma_x', 'soma_y', 'soma_z']][sel_mask].values 
    if flipLR:
        zdim = 404 * 2. / 1000
        right = np.nonzero(coords[:,2] > zdim/2)[0]
        coords[right, 2] = zdim - coords[right, 2]

    # We handling the coloring
    dfc = dfr.copy()
    for i in range(3):
        tmp = dfc.iloc[:,i]
        dfc.iloc[:,i] = image_histeq(tmp.values)[0] / 255.
    dfc[dfc > 1] = 1.
        
    # axes instance
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # plot
    sc = ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=3, c=dfc.values, marker='o', alpha=.75)
    label_size = 22
    ax.set_xlabel('LR axis (μm)', fontsize=label_size, labelpad=10)
    ax.set_ylabel('AP axis (μm)', fontsize=label_size, labelpad=10)
    ax.set_zlabel('DV axis (μm)', fontsize=label_size, labelpad=10)
    #ax.set_xlim(ax.get_xlim()[::-1])
    #ax.set_zlim(ax.get_zlim()[::-1])
    ax.set_xlim([0, 900])
    ax.set_ylim([0, 900])
    ax.set_zlim([0, 900])
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_axis_off()


    # legend
    plt.legend(bbox_to_anchor=(0.6,0.6), fontsize=label_size, markerscale=3., handletextpad=0.2)
    fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    ax.set_box_aspect(None, zoom=0.85)  # to avoid Z label cutoff
    ax.get_legend().remove()
    # Hide grid lines
    ax.grid(False)
    #ax.view_init(0, 30)
    plt.show()

    # save
#    if '/' in out_prefix:
#        out_prefix = out_prefix.replace('/', '_')
#    plt.savefig(f"{out_prefix}_features_ccf.png", bbox_inches='tight')
#    plt.close()

def plot_region_feature_sections(mefile, rname='ME_R', r316=False, flipLR=False, thickX2=40, feat_type='me'):
    df = pd.read_csv(mefile, index_col=1).dropna()
    if feat_type == 'me':
        keys = [f'{key}_me' for key in __MAP_FEATS__]
    elif feat_type == 'single':
        keys = __MAP_FEATS__
    elif feat_type == 'global_pca':
        keys = ['pca_feat1', 'pca_feat2', 'pca_feat3']
    elif feat_type == 'local_me_pca':
        keys = [key for key in df.columns if key.endswith('_me')]
    elif feat_type == 'local_single_pca':
        keys = [key[:-3] for key in df.columns if key.endswith('_me')]
        if len(keys) == 0:
            from config import __FEAT24D__
            keys = __FEAT24D__
    else:
        raise ValueError
    

    if r316:
        rkey = 'region_name_r316'
        mask = load_image(MASK_CCF25_R314_FILE)
    else:
        rkey = 'soma_region'
        mask = load_image(MASK_CCF25_FILE) 
    ana_tree = parse_ana_tree(keyname='name')

    if type(rname) is list:
        sel_mask = df[rkey].isin(rname)
        rmask = np.zeros_like(mask)
        for ri in rname:
            idx = ana_tree[ri]['id']
            rmask = rmask | (mask == idx)
            
        out_prefix = 'tmp'
    else:
        sel_mask = df[rkey] == rname
        idx = ana_tree[rname]['id']
        
        rmask = mask == idx
        out_prefix = rname
    
    dfr = df[keys][sel_mask]
    if (feat_type == 'local_me_pca') or (feat_type == 'local_single_pca'):
        # do pca feature reduction
        pca = PCA(n_components=3, whiten=True)
        dfr = pd.DataFrame(pca.fit_transform(dfr), columns=('pca_feat1', 'pca_feat2', 'pca_feat3'))

    #coords = (df[['soma_x', 'soma_y', 'soma_z']][sel_mask].values / 2) + np.array([-85.8770447/2, -63.0468216/2, 3.3968091/2])
    coords = (df[['soma_x', 'soma_y', 'soma_z']][sel_mask].values) + np.array([-85.877, -63.046, 3.416])
    #print(coords)
    if flipLR:
        zdim = 404
        zcoord = zdim * 2 
        right = np.nonzero(coords[:,2] > zcoord/2)[0]
        coords[right, 2] = zcoord - coords[right, 2]
        rmask[zdim//2:] = 0

    # We handling the coloring
    dfc = dfr.copy()
    for i in range(3):
        tmp = dfc.iloc[:,i]
        dfc.iloc[:,i] = image_histeq(tmp.values)[0]
    dfc[dfc > 255] = 255

    # get the boundary of region
    nzcoords = rmask.nonzero()
    nzcoords_t = np.array(nzcoords).transpose()   
    zmin, ymin, xmin = nzcoords_t.min(axis=0)
    zmax, ymax, xmax = nzcoords_t.max(axis=0)
    print(zmin,ymin,xmin)
    print(zmax,ymax,xmax)
    sub_mask = rmask[zmin-1:zmax+1, ymin-1:ymax+1, xmin-1:xmax+1]
    memap = np.zeros((*sub_mask.shape, 3), dtype=np.uint8)
    
    coords_s = np.floor(coords).astype(int)
    #memap[coords_s[:,0]-zmin, coords_s[:,1]-ymin, coords_s[:,2]-xmin] = dfc.values
    valid_coords = (
    (coords_s[:, 0] >= zmin) & (coords_s[:, 0] <= zmax) &
    (coords_s[:, 1] >= ymin) & (coords_s[:, 1] <= ymax) &
    (coords_s[:, 2] >= xmin) & (coords_s[:, 2] <= xmax)
    )
    valid_coords_s = coords_s[valid_coords]
    memap[valid_coords_s[:, 0] - zmin, valid_coords_s[:, 1] - ymin, valid_coords_s[:, 2] - xmin] = dfc.values[:len(valid_coords_s)]


    mips = []
    shape3d = mask.shape
    axid = 2
    dmax, dmin = nzcoords_t.max(axis=0)[axid], nzcoords_t.min(axis=0)[axid]
    for sid in range(0, dmax-dmin-thickX2-1, thickX2):       #(0, dmax-dmin-thickX2-1, thickX2*2)
        sid = sid + thickX2
        cur_memap = memap.copy()
        if axid == 0:
            cur_memap[:sid-thickX2] = 0
            cur_memap[sid+thickX2:] = 0
        elif axid == 1:
            cur_memap[:,:sid-thickX2] = 0
            cur_memap[:,sid+thickX2:] = 0
        elif axid == 2:
#            cur_memap[:,:,:sid-thickX2] = 0
#            cur_memap[:,:,sid+thickX2:] = 0
             cur_memap[:,:,:sid-3] = 0
             cur_memap[:,:,sid+3:] = 0         #very important
        print(cur_memap.mean(), cur_memap.std())

        mip = get_mip_image(cur_memap, axid)
        
        figname = f'{out_prefix}_{feat_type}_section{sid:03d}.png'
        print(mip.shape, sub_mask.shape)
        process_mip(mip, sub_mask, axis=axid, figname=figname, sectionX=sid, with_outline=False, pt_scale=5, b_scale=1)
        # load and remove the zero-alpha block
        img = cv2.imread(figname, cv2.IMREAD_UNCHANGED)
        wnz = np.nonzero(img[img.shape[0]//2,:,-1])[0]
        ws, we = wnz[0], wnz[-1]
        hnz = np.nonzero(img[:,img.shape[1]//2,-1])[0]
        hs, he = hnz[0], hnz[-1]
        img = img[hs:he+1, ws:we+1]
        # set the alpha of non-brain region as 0
        img[img[:,:,-1] == 1] = 0
        if axid != 0:   # rotate 90
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            #
        cv2.imwrite(figname, img)
 

def plot_region_clusters_on_sections(mefile=None, rname='MOB', r316=False, flipLR=False, thickX2=10, feat_type='me', seed=1024, projection_file=None):
    if feat_type == 'me':
        df = pd.read_csv(mefile, index_col=1).dropna()
        keys = [f'{key}_me' for key in __MAP_FEATS__]
    elif feat_type == 'single':
        df = pd.read_csv(mefile, index_col=1).dropna()
        keys = __MAP_FEATS__
    elif feat_type == 'projection':
        df = pd.read_csv(projection_file, index_col=0).dropna()
        keys = __Subregion_id__
        #df[keys] = df[keys] / 1000
    else:
        raise ValueError

    if r316:
        rkey = 'region_name_r316'
        mask = load_image(MASK_CCF25_R314_FILE)
    else:
        rkey = 'soma_region'
        mask = load_image(MASK_CCF25_FILE)
    ana_tree = parse_ana_tree(keyname='name')

    if type(rname) is list:
        sel_mask = df[rkey].isin(rname)
        rmask = np.zeros_like(mask)
        for ri in rname:
            idx = ana_tree[ri]['id']
            rmask = rmask | (mask == idx)
            
        out_prefix = 'tmp'
    else:
        sel_mask = df[rkey] == rname
        idx = ana_tree[rname]['id']
        rmask = mask == idx
        out_prefix = rname
    
    dfr = df[keys][sel_mask]

    coords = (df[['soma_x', 'soma_y', 'soma_z']][sel_mask].values) + np.array([-85.877, -63.046, 3.416])
    if flipLR:
        zdim = 162
        zcoord = zdim * 5.
        right = np.nonzero(coords[:,2] > zcoord/2)[0]
        coords[right, 2] = zcoord - coords[right, 2]
        rmask[zdim//2:] = 0

    # do clustering
    print(f'--> Processing for region {rname} with {dfr.shape[0]} samples')
    # standardize
    
    standardize_features(dfr, dfr.columns)
    reducer = umap.UMAP(random_state=seed)
    embedding = reducer.fit_transform(dfr)
    umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])  
    
    
    # clustering
    nclusters = 8

    db = SpectralClustering(n_clusters=nclusters, random_state=seed).fit(embedding)
    labels = db.labels_
    # Output UMAP clustering information
    cluster_info = pd.DataFrame({'Name': dfr.index, 'Cluster': labels})
    print(cluster_info)
    # Save cluster_info to CSV
    output_csv = f'{out_prefix}_{feat_type}_cluster_info.csv'
    cluster_info.to_csv(output_csv, index=False)
    print(f'Cluster information saved to {output_csv}')
    name_column = cluster_info.iloc[:, 0]
    umap_df['Name'] = name_column
    umap_df.to_csv('UMAP_cord.csv', index=False)
    # Plot UMAP 2D scatter
    plt.figure(figsize=(10, 8))
    custom_colors = [
    '#FF0000', 
    '#00FF00',  
    '#0000FF', 
    '#FFFF00',  
    '#800080', 
    '#00FFFF', 
    '#FFA500', 
    '#FFC0CB',
    '#FFFF00',  
    '#800080', 
    '#00FFFF', 
    '#FFA500'
     ]
    for i, cluster in enumerate(np.unique(labels)):
        plt.scatter(
            embedding[labels == cluster, 0], 
            embedding[labels == cluster, 1], 
            label=f'Cluster {cluster}', 
            color=custom_colors[i % len(custom_colors)],  # Assign colors cyclically
            alpha=0.5
        )
    plt.title(f'{feat_type}_umap_plot')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.grid(False)
    # Save the plot
    plt.savefig(f'{out_prefix}_{feat_type}_umap_plot.png')
    # sort the label, so that they were more likely consistent across feat_types.
    sorted_labels = np.zeros_like(labels)
    unique_labels = sorted(set(labels))
    # sorting criterion. It's not guaranted to be consistent across feat_types
    means = [coords[labels == label].mean(axis=0) for label in unique_labels]
    random.seed(seed)
    #random.shuffle(means)
    sorted_indices = np.argsort([mean[1] for mean in means])
    # map the original labels to sorted labels
    for new_label, old_label in enumerate(sorted_indices):
        sorted_labels[labels == unique_labels[old_label]] = new_label
    labels = sorted_labels

    #colors = {ie: plt.cm.rainbow(each, bytes=True) for ie, each in enumerate(np.linspace(0, 1, len(unique_labels)))}
#    custom_colors = [
#    '#FF0000', 
#    '#00FF00',  
#    '#0000FF' 
#     ]
     
    all_colors = np.array([custom_colors[label] for label in labels])
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)])

    rgb_colors = np.array([hex_to_rgb(c) for c in all_colors])
    rgb_colors = np.clip(rgb_colors, 0, 255)
    dfc = dfr.copy()
    dfc.iloc[:, :3] = rgb_colors
    
#    colors = {i: custom_colors[i] for i in range(nclusters)}
#    all_colors = np.array([colors[i] for i in labels])

#    dfc = dfr.copy()
#    dfc.iloc[:,:3] = all_colors[:,:3]
#    dfc[dfc > 255] = 255

    # get the boundary of region
    nzcoords = rmask.nonzero()
    nzcoords_t = np.array(nzcoords).transpose()
    zmin, ymin, xmin = nzcoords_t.min(axis=0)
    zmax, ymax, xmax = nzcoords_t.max(axis=0)
    sub_mask = rmask[zmin-1:zmax+1, ymin-1:ymax+1, xmin-1:xmax+1]
    memap = np.zeros((*sub_mask.shape, 3), dtype=np.uint8)
    
    coords_s = np.floor(coords).astype(int)
    valid_coords = (
    (coords_s[:, 0] >= zmin) & (coords_s[:, 0] <= zmax) &
    (coords_s[:, 1] >= ymin) & (coords_s[:, 1] <= ymax) &
    (coords_s[:, 2] >= xmin) & (coords_s[:, 2] <= xmax)
    )
    valid_coords_s = coords_s[valid_coords]
    memap[valid_coords_s[:, 0] - zmin, valid_coords_s[:, 1] - ymin, valid_coords_s[:, 2] - xmin] = dfc.values[:len(valid_coords_s), :3]
    #memap[valid_coords_s[:, 0] - zmin, valid_coords_s[:, 1] - ymin, valid_coords_s[:, 2] - xmin] = dfc.values[:len(valid_coords_s)]
    #memap[coords_s[:,0]-zmin, coords_s[:,1]-ymin, coords_s[:,2]-xmin] = dfc.values

    mips = []
    shape3d = mask.shape
    axid = 1
    dmax, dmin = nzcoords_t.max(axis=0)[axid], nzcoords_t.min(axis=0)[axid]
    for sid in range(0, dmax-dmin-thickX2-1, thickX2):
        sid = sid + thickX2
        cur_memap = memap.copy()
        if axid == 0:
            cur_memap[:sid-1] = 0
            cur_memap[sid+1:] = 0
        elif axid == 1:
            cur_memap[:,:sid-1] = 0
            cur_memap[:,sid+1:] = 0
        elif axid == 2:
            cur_memap[:,:,:sid-1] = 0
            cur_memap[:,:,sid+1:] = 0
        print(cur_memap.mean(), cur_memap.std())

        mip = get_mip_image(cur_memap, axid)
        
        figname = f'{out_prefix}_{feat_type}_section{sid:03d}.png'
        print(mip.shape, sub_mask.shape)
        process_mip(mip, sub_mask, axis=axid, figname=figname, sectionX=sid, with_outline=False, pt_scale=3, b_scale=0.5)
        # load and remove the zero-alpha block
        img = cv2.imread(figname, cv2.IMREAD_UNCHANGED)
        wnz = np.nonzero(img[img.shape[0]//2,:,-1])[0]
        ws, we = wnz[0], wnz[-1]
        hnz = np.nonzero(img[:,img.shape[1]//2,-1])[0]
        hs, he = hnz[0], hnz[-1]
        img = img[hs:he+1, ws:we+1]
        # set the alpha of non-brain region as 0
        img[img[:,:,-1] == 1] = 0
        if axid != 0:   # rotate 90
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            #
        cv2.imwrite(figname, img)
    

def plot_parcellations(parc_file, ccf_tree_file=ANATOMY_TREE_FILE, ccf_atlas_file=MASK_CCF25_FILE):
    thickX2 = 10
    axid = 2
    parc = load_image(parc_file)
    # flip
    zdim2 = parc.shape[0] // 2
    if parc.shape[0] % 2 == 0:
        parc[:zdim2] = parc[zdim2:][::-1]
    else:
        parc[:zdim2] = parc[zdim2 + 1:][::-1]
    ana_tree = parse_ana_tree(ccf_tree_file)
    ccf25 = load_image(ccf_atlas_file)
    shape3d = parc.shape
    prefix = os.path.splitext(os.path.split(parc_file)[-1])[0]
    for isid, sid in enumerate(range(thickX2, shape3d[axid], 2*thickX2)):
        figname = f'{prefix}_{isid:02d}.png'
        section = np.take(parc, sid, 2)
        ccf25s = np.take(ccf25, sid, 2)
        vuniq = np.unique(ccf25s)
        # coloring with CCF color scheme
        out = np.ones((*section.shape, 4), dtype=np.uint8) * 255
        out3 = out[:,:,:3]
        for vi in vuniq:
            if vi == 0: continue
            rmask = ccf25s == vi
            #print(rmask.sum())
            out3[rmask] = ana_tree[vi]['rgb_triplet']

        # draw the sub-parcellation
        parc_edges = detect_edges2d(section)
        ccf25_edges = detect_edges2d(ccf25s)
        print("parc_edges shape:", parc_edges.shape)
        print("ccf25_edges shape:", ccf25_edges.shape)

        extra_edges = parc_edges ^ ccf25_edges
        #extra_edges[:zdim2] = extra_edges[zdim2:][::-1]
        if extra_edges.shape[0] % 2 == 0:
            extra_edges[:zdim2] = extra_edges[zdim2:][::-1]
        else:
            extra_edges[:zdim2] = extra_edges[zdim2 + 1:][::-1]
        out[extra_edges] = (0,0,255,255)
        # draw the original ccf outline
        out[ccf25_edges] = (0,0,0,128)
        # zeroing the background
        # rotate
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(figname, out)
        print()


    

if __name__ == '__main__':
    mefile = './data/mefeatures_100K.csv'
    mapfile = 'microenviron_map'
    scale = 1.
    flip_to_left = True
    mode = 'soma'  # 'compsite'/'single'/'soma'
    axids = (2,)
    findex = 0
    fmt = 'png'
    use_me_feature = True

    if 0:
        generate_me_maps(mefile, outfile=mapfile, flip_to_left=flip_to_left, mode=mode, findex=findex, fmt=fmt, axids=axids, use_me_feature=use_me_feature)

    if 1:
        mefile = './data/mefeatures_130K.csv'
        swcdir = '/home/ylx/fly/data_rename'
        projection_file = './data/mefile_with_projection.csv'

        #find_regional_representative(mefile, region=region, swcdir=swcdir, color=color)
        #plot_inter_regional_features(mefile)
        rname = ['AVLP_R', 'LAL_R', 'AL_L', 'SMP_R', 'AVLP_L', 'MB_ML_R', 
                 'CRE_L', 'SLP_R', 'GNG', 'CRE_R', 'SMP_L', 'EB', 'PRW', 'AL_R', 'MB_VL_L', 'GA_L', 'AOTU_R', 'MB_ML_L', 
                 'SIP_R', 'LA_R', 'MB_VL_R', 'FLA_R', 'LAL_L', 'FLA_L', 'MB_PED_R', 'SAD', 'AMMC_R', 'AOTU_L', 'PVLP_R', 
                 'GA_R', 'SIP_L', 'MB_PED_L', 'AMMC_L', 'LO_L', 'LOP_L', 'LO_R', 'PLP_R', 'LOP_R', 'ME_L', 'MB_CA_L', 'ME_R', 
                 'SLP_L', 'PLP_L', 'LH_L', 'SPS_L', 'IPS_R', 'SPS_R', 'PVLP_L', 'SCL_R', 'ICL_R', 'NO', 'WED_L', 'MB_CA_R', 'WED_R', 
                 'ICL_L', 'VES_L', 'LA_L', 'IB_R', 'ATL_L', 'FB', 'LH_R', 'IB_L', 'IPS_L', 'VES_R', 'EPA_L', 'GOR_R', 'PB', 'GOR_L', 
                 'EPA_R', 'SCL_L', 'ATL_R', 'AME_R', 'AME_L', 'CAN_R', 'OCG', 'BU_R', 'CAN_L', 'BU_L']
        #rname = ['MB_CA_L','MB_CA_R','MB_VL_R','MB_VL_L','MB_ML_L','MB_ML_R','MB_PED_L','MB_PED_R']
        #plot_MOB_features(mefile, rname)  #plot selected features
        plot_region_feature_in_ccf_space(mefile, rname)
        #plot_region_feature_sections(mefile, rname, feat_type='local_me_pca')
        #plot_region_feature_sections(mefile, rname, feat_type='local_single_pca')
        #plot_region_clusters_on_sections(mefile=mefile, rname=rname, feat_type='single')
        #plot_region_clusters_on_sections(mefile=mefile, rname=rname, feat_type='me')
        #plot_region_clusters_on_sections(mefile=None, rname=rname, feat_type='projection', projection_file=projection_file)
        
        
    if 0:
        parc_file = 'intermediate_data/annotation_25.nrrd'
        plot_parcellations(parc_file)
