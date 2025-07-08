##########################################################
#Author:          Yufeng Liu
#Create time:     2024-02-04
#Description:               
##########################################################

import os
import glob
import time
import json
import random
import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
from multiprocessing.pool import Pool
from collections import defaultdict
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.sparse import csr_matrix
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import Voronoi, distance_matrix
import matplotlib as mpl
import matplotlib.cm as cm
import cv2
import cc3d
from skimage.morphology import ball as morphology_ball
from skimage.morphology import cube as morphology_cube
from skimage.morphology import erosion
from skimage.filters.rank import majority as majority_filter
from sklearn.neighbors import KNeighborsClassifier

import leidenalg as lg
import igraph as ig

from image_utils import get_mip_image, image_histeq
from file_io import load_image, save_image
from anatomy.anatomy_config import MASK_CCF25_FILE, MASK_CCF25_R314_FILE, REGION314, REGION671, TEST_REGION
from anatomy.anatomy_vis import detect_edges2d
from math_utils import min_distances_between_two_sets

import sys
from generate_me_map import process_mip
from shape_normalize import shape_normalized_scaling
from config import mRMR_f3, mRMR_f3me, gini_coeff, moranI_score, load_features


def reorder_mask_using_cc(sub_mask, cc_mask, sub_fg_mask, min_pts_per_parc):
    if cc_mask is None:
        cc_mask = cc3d.connected_components(sub_mask, connectivity=26)
        # we would like to re-estimate the small communities/components
        cc_ids, cc_cnts = np.unique(cc_mask, return_counts=True)
    # reordering the mask indices using connected components
    # Firstly, found out the background mask
    bg_ids = np.unique(cc_mask[~sub_fg_mask])
    fg_ids, fg_cnts = np.unique(cc_mask[sub_fg_mask], return_counts=True)
    argsort_ids = np.argsort(fg_cnts)[::-1]
    fg_ids = fg_ids[argsort_ids]
    fg_cnts = fg_cnts[argsort_ids]
    sub_mask[~sub_fg_mask] = 0
    
    # The original CCFv3 atlas does not ensure a connected components
    idict = {}
    cur_id = 0
    for i, fg_id in enumerate(fg_ids):
        cur_fg = cc_mask == fg_id
        sub_id = sub_mask[cur_fg][0]
        if cur_fg.sum() < min_pts_per_parc:
            if sub_id in idict:
                cur_id = idict[sub_id]
            else:
                cur_id += 1
                idict[sub_id] = cur_id
        else:
            cur_id += 1
            if sub_id not in idict:
                idict[sub_id] = cur_id
                
        sub_mask[cur_fg] = cur_id
        #print(cur_fg.sum(), idict)
    #print(np.unique(sub_mask, return_counts=True))

def random_colorize(coords, values, shape3d, color_level):
    """
    coords in shape of [N,3], in order of ZYX
    """
    # map the communities to different colors using randomized color map
    norm = mpl.colors.Normalize(values.min(), vmax=values.max())
    vals = np.linspace(0,1,color_level)
    np.random.shuffle(vals)
    cmap = cm.colors.ListedColormap(cm.jet(vals))
    #cmap = cm.bwr
    smapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = np.floor(smapper.to_rgba(values) * 255).astype(np.uint8)
    zyx = np.floor(coords).astype(np.int32)

    # intialize map
    pmap = np.zeros((*shape3d, colors.shape[-1]), dtype=np.uint8)
    
    # Fill pmap with colors
    for idx, coord in enumerate(zyx):
        z, y, x = coord
        if 0 <= z < shape3d[0] and 0 <= y < shape3d[1] and 0 <= x < shape3d[2]:
            pmap[z, y, x, :] = colors[idx]
    

    return pmap


class BrainParcellation:
    def __init__(self, mefile, scale=1., feat_type='mRMR', flipLR=False, seed=1024, r314_mask=True, 
                 debug=False, out_mask_dir='./output', out_vis_dir='./vis', shape_normalize=True,
                 min_neurons_per_region=40, min_max_width=9):
        """
        @args feat_type: 
                    - "full": complete set of L-Measure features
                    - "mRMR": the top 3 features selected using mRMR from the complete set
                    - "PCA": the top 3 features selected using PCA from the complete set
        """
        self.df, self.fnames = load_features(mefile, scale=scale, feat_type=feat_type, flipLR=flipLR)
#        print(self.df)
        print("length is:",len(self.df))
        self.feat_type = feat_type
        self.seed = seed
        np.random.seed(seed)
        self.flipLR = flipLR
        self.debug = debug
        self.shape_normalize = shape_normalize
        self.min_neurons_per_region = min_neurons_per_region
        self.min_max_width = min_max_width

        self.r314_mask = r314_mask
        if self.r314_mask:
            mask_file = MASK_CCF25_R314_FILE
        else:
            mask_file = MASK_CCF25_FILE
        self.mask = load_image(mask_file)  # z,y,x order!
        lmask = self.mask.copy() # left mask
        lmask[:self.mask.shape[0]//2] = 0
        self.lmask = lmask

        if not os.path.exists(out_mask_dir):
            os.mkdir(out_mask_dir)
        if not os.path.exists(out_vis_dir):
            os.mkdir(out_vis_dir)
        self.out_vis_dir = out_vis_dir
        self.out_mask_dir = out_mask_dir



    def visualize_on_ccf(self, dfp, mask, prefix='temp'):
        shape3d = mask.shape
        zdim2, ydim2, xdim2 = shape3d[0]//2, shape3d[1]//2, shape3d[2]//2    #world cord  x y z

        crds = dfp[['soma_x', 'soma_y', 'soma_z']]
        values = dfp['parc']
        ncolors = values.max()
        pmap = random_colorize(crds.to_numpy(), values, shape3d, ncolors)
        
        thickX2 = 10
        axid = 2
        for isec, sec in enumerate(range(thickX2, shape3d[axid], thickX2*2)):
            print(f'--> Processing section: {sec}')
            cur_map = pmap.copy()
            cur_map[:,:,:isec*2*thickX2] = 0
            cur_map[:,:,(isec*2+2)*thickX2:] = 0
            print(cur_map.mean(), cur_map.std())

            mip = get_mip_image(cur_map, axid)
            figname = os.path.join(self.out_vis_dir, f'{prefix}_mip{axid}_{isec:02d}.png')
            process_mip(mip, mask, axis=axid, figname=figname, mode='composite', sectionX=sec, with_outline=False, pt_scale=5, b_scale=0.5)
            # load and remove the zero-alpha block
            img = cv2.imread(figname, cv2.IMREAD_UNCHANGED)
            wnz = np.nonzero(img[img.shape[0]//2,:,-1])[0]
            ws, we = wnz[0], wnz[-1]
            hnz = np.nonzero(img[:,img.shape[1]//2,-1])[0]
            hs, he = hnz[0], hnz[-1]
            img = img[hs:he+1, ws:we+1]
            if axid != 0:   # rotate 90
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            # set the alpha of non-brain region as 0
            img[img[:,:,-1] == 1] = 0
            img = img[:,::-1]
            cv2.imwrite(figname, img)

    @staticmethod
    def remove_small_parcs(sub_mask, sub_fg_mask, cc_mask=None, cc_ids=None, cc_cnts=None, min_pts=0, use_cc_ids=False):
        if cc_mask is None:
            cc_mask = cc3d.connected_components(sub_mask, connectivity=26)
            # we would like to re-estimate the small communities/components
            cc_ids, cc_cnts = np.unique(cc_mask, return_counts=True)
        
        have_small_parcs = False
        for cc_id, cc_cnt in zip(cc_ids, cc_cnts):
            if cc_cnt < min_pts:
                try:
                    interp_mask = interp_mask | (cc_mask == cc_id)
                except UnboundLocalError:
                    interp_mask = cc_mask == cc_id
                have_small_parcs = True

        if have_small_parcs:
            if sub_fg_mask.sum() == interp_mask.sum():
                # no sufficient data to interpolate
                sub_mask[sub_fg_mask] = 1
                cc_mask = sub_mask.copy()
                cc_ids, cc_cnts = np.unique(cc_mask, return_counts=True)
                return sub_mask, cc_mask, cc_ids, cc_cnts

            if use_cc_ids:
                reorder_mask_using_cc(sub_mask, cc_mask, sub_fg_mask, min_pts_per_parc)

            anchor_mask = sub_fg_mask ^ interp_mask
            anchor_coords = np.array(np.where(anchor_mask)).transpose()
            anchor_values = sub_mask[anchor_mask]
            interp = NearestNDInterpolator(anchor_coords, anchor_values)
            predv = interp(*np.where(interp_mask))
            sub_mask[interp_mask] = predv

            # re-estimation
            cc_mask = cc3d.connected_components(sub_mask, connectivity=26)
            # we would like to re-estimate the small communities/components
            cc_ids, cc_cnts = np.unique(cc_mask, return_counts=True)

        return sub_mask, cc_mask, cc_ids, cc_cnts
        
    def save_colorized_images(self, cmask, mask, reg_sub_parc, out_image_file, thickX2=10, plot_regid=False):
        zdim, ydim, xdim = mask.shape
        zdim2, ydim2, xdim2 = zdim // 2, ydim // 2, xdim // 2
        # visualize
        prefix = os.path.splitext(os.path.split(out_image_file)[-1])[0]
        fprefix = os.path.join(self.out_vis_dir, prefix)

        for i, dim in zip(range(3), (zdim, ydim, xdim)):
            if i != 2: continue

            for isec, sec in enumerate(range(thickX2, dim, thickX2)):
                k = sec
                m2d0 = np.take(cmask, k, i)
                # overlay the boundaries
                m2d1 = np.take(mask, k, i)
                edges = detect_edges2d(m2d1)
                p1 = m2d0.copy()
                p1[edges] = np.array([0,0,0,255])
                outfile = f'{fprefix}_axid{i}_{isec:02d}.png'
                p1 = cv2.flip(p1, 0)
                # write the regid
                if plot_regid:
                    reg2d = cv2.flip(np.take(reg_sub_parc, k, i), 0)
                    for cur_regid in np.unique(reg2d):
                        if cur_regid == 0:
                            continue
                        cur_mask = reg2d == cur_regid
                        yy, xx = np.nonzero(cur_mask)
                        yc, xc = int(yy.mean()), int(xx.mean())
                        center = (xc, yc)
                        print(isec, cur_regid, center)
                        cv2.putText(p1, str(cur_regid), center, cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,255), 2)

                cv2.imwrite(outfile, p1)
                if i != 0:
                    #print(f'Rotate by 90 degree')
                    os.system(f'convert {outfile} -rotate 90 {outfile}')

        print()

    def do_leiden(self, coords, feats, reg_mask, max_n_neighbors, shape_normalize, n_jobs):
        t0 = time.time()
        n_neighbors = min(max_n_neighbors, coords.shape[0])
        if shape_normalize:
            coords_t = shape_normalized_scaling(reg_mask, coords, visualize=self.debug)
        else:
            coords_t = coords

        A = kneighbors_graph(coords_t, 
                             n_neighbors=n_neighbors, include_self=True, 
                             mode='distance', metric='euclidean', n_jobs=n_jobs)

        dist_th = A[A>0].max()# + 1e-5   # to ensure all included
        if self.debug:
            print(f'Threshold for graph construction: {dist_th:.4f} <<-- {time.time() - t0:.2f} seconds')
        
        A.setdiag(0)
        if self.debug:
            print(f'[Neighbors generation]: {time.time() - t0:.2f} seconds')
        
        A_csr = csr_matrix(A)
        sources, targets = A_csr.nonzero()
        # estimate the edge weights
        dists = A_csr[sources, targets]
        wd = np.squeeze(np.asarray(np.exp(-dists/dist_th)))

        if self.debug:
            print(f'wd[mean/max/min]: {wd.mean():.2f}, {wd.max():.2f}, {wd.min():.2f}')
            print(f'Total and avg number of edges: {wd.shape[0]}, {wd.shape[0]/feats.shape[0]:.2f}')

        fs = feats[sources]
        ft = feats[targets]
        par2 = 3
        if self.feat_type == 'full':
            # The mean pairwise distances of full set of features (`feat_type == full`) are about 3x
            # (6.576) compared to that of mRMR3 (2.180) and PCA3 (2.236).
            par2 = par2 / 3.
        wf = np.exp(-par2*np.linalg.norm(fs - ft, axis=1))

        if self.debug:
            print(f'wf[mean/max/min]: {wf.mean():.2f}, {wf.max():.2f}, {wf.min():.2g}')
            print(f'[weights estimation]: {time.time() - t0:.2f} seconds')

        weights = wf

        g = ig.Graph(list(zip(sources, targets)), directed=False)
        g.es['weight'] = weights
        if self.debug:
            print(f'[Graph initialization]: {time.time() - t0: .2f} seconds')
        
        ### Step 3: Apply the Leiden Algorithm
        partition = lg.find_partition(g, lg.ModularityVertexPartition, weights='weight', seed=self.seed)
        if self.debug:
            print(f'[Partition]: {time.time() - t0: .2f} seconds')

        # in case of isolated points
        cm = [i for i in partition.membership]
        nii = max(partition.membership)
        for ii in range(len(cm), coords.shape[0]):
            nii += 1
            cm.append(nii)
        
        community_memberships = np.array([i+1 for i in cm]) # re-indexing starting from 1, to differentiate with background
        #print('This is for test !!!')
        #print(community_memberships)
        #print(len(community_memberships))
        return coords_t, community_memberships, partition

    def community_detection(self, coords, feats, reg_mask, n_jobs=2):
        # or try to use radius_neighbors_graph
        # the radius are in 25um space

        t0 = time.time()
        coords = coords.values.astype(np.float64)

        # do silhoutte-based adaptive parameter tuning
        max_sil_score = -1
        if 1:
            start_id = 50
            end_id = 250
            step_id = 50
            for n_neighbors in range(start_id, end_id+step_id, step_id):
                for shape_normalize in [True, False]:
                    coords_t, community_memberships, partition = self.do_leiden(
                            coords, feats, reg_mask, n_neighbors, shape_normalize, n_jobs
                    )

                    # estimate the silhoutte score
                    n_sample = 1000
                    if len(community_memberships) > n_sample:
                        # select only a subset of data
                        random.seed(1024)
                        rids = random.sample(range(len(community_memberships)), n_sample)
                        sil_feats = feats[rids]
                        sil_labels = community_memberships[rids]
                    else:
                        sil_feats = feats
                        sil_labels = community_memberships
                    sil_score = metrics.silhouette_score(sil_feats, sil_labels)
                    print(f'[n_neighbors={n_neighbors}, shape_normalize={shape_normalize}]: sil={sil_score:.4f}')
                    if sil_score > max_sil_score:
                        max_sil_score = sil_score
                        max_n_neighbors = n_neighbors
                        max_shape_normalize = shape_normalize

                if n_neighbors > coords.shape[0]:
                    break
            
            # re-estimate using the best parameter
            print(f'***Best parameter are: [n_neighbors={max_n_neighbors}, shape_normalize={max_shape_normalize}], with silhouette_score={max_sil_score}***')
        else:
            max_n_neighbors = 200
            max_shape_normalize = False


        # top3 indices
        orig_fstd = feats.std(axis=0).mean()
        
        coords_t, community_memberships, partition = self.do_leiden(
                coords, feats, reg_mask, max_n_neighbors, max_shape_normalize, n_jobs
        )
        print('coords_t:',coords_t)
        print('community_memberships type:',len(np.unique(community_memberships)))
        #print('partition:',partition)
        print('feats:',feats)
        
        
        # average gini after estimation:
        avg_fstd = []
        for rid in np.unique(community_memberships):
            cur_m = community_memberships == rid
            feat_i = feats[cur_m]
            avg_fstd.append(feat_i.std(axis=0).mean())
        avg_fstd = np.mean(avg_fstd)
        print(f'--> [Feature STD] original and after: {orig_fstd:.3f} {avg_fstd:.3f}')

#         #denoising at the neuron level
#         #remove nodes with nearest nodes are not the same class
        topk = 6
        A1 = kneighbors_graph(coords_t, n_neighbors=topk-1, include_self=False, mode='distance', metric='euclidean')
        yc, xc = A1.nonzero()
        

        community_sizes = np.array([len(community) for community in partition])
        comms, counts = np.unique(community_sizes, return_counts=True)
        if self.debug:
            print(f'[Number of communities] = {len(partition)}')
            print(f'[Community statistics: mean/std/max/min]: {community_sizes.mean():.1f}, {community_sizes.std():.1f}, {community_sizes.max()}, {community_sizes.min()}')
            print(comms, counts)

        node_to_community = {node: community for node, community in enumerate(community_memberships)}
        # Initialize a dictionary to hold lists of nodes for each community
        communities = defaultdict(list) # community to node

        # Populate the dictionary with node indices grouped by their community
        for node_index, community_index in enumerate(community_memberships):
            communities[community_index].append(node_index)
        
        return communities, np.array(community_memberships)

    def insufficient_data(self, reg_mask, out_image_file, save_mask):
        mask_u16 = reg_mask.astype(np.uint16)
        if save_mask:
            save_image(out_image_file, mask_u16, useCompression=True)
        return mask_u16

    def parcellate_region(self, regid, save_mask=True):
        print(f'---> Processing for region={regid}')
        t0 = time.time()
        min_pts_per_parc = 12**3 # (0.25*x)^6 um^3
        if type(regid) is list:
            rprefix = 9999
        else:
            rprefix = regid

        out_image_file = os.path.join(self.out_mask_dir, f'parc_region{rprefix}.nrrd')

        # Compute the sparse nearest neighbors graph
        # Adjust n_neighbors based on your dataset and memory constraints
        coords_all = self.df[['soma_x', 'soma_y', 'soma_z']] 
        feats_all = self.df[self.fnames].to_numpy()

        # using CP to debug
        # CP: 672; MOB:507, CA1: 382, AOB:151
        if self.r314_mask:
            if type(regid) is list:
                cp_mask = self.df['region_id'].isin(regid)
            else:
                cp_mask = self.df['region_id'] == regid
        else:
            if type(regid) is list:
                cp_mask = self.df['region_id'].isin(regid)
            else:
                cp_mask = self.df['region_id'] == regid

        # Artificial cases for debugging
        #nz_tmp = cp_mask.values.nonzero()[0]
        #cp_mask = self.df.index.isin([self.df.iloc[nz_tmp[0]].name, self.df.iloc[nz_tmp[1]].name])

        # assign the current region into Voronoi cells
        if self.flipLR:
            __tmp_mask__ = self.lmask
        else:
            __tmp_mask__ = self.mask
        if type(regid) is list:
            reg_mask = __tmp_mask__ == -1
            for rid in regid:
                reg_mask = reg_mask | (__tmp_mask__ == rid)
        else:
            reg_mask = __tmp_mask__ == regid
        
        # [CHECK]: number of neurons within each region
        if cp_mask.sum() < self.min_neurons_per_region:
            print(f'[Warning] Insufficient samples ({cp_mask.sum()}) are found in regid={regid}')
            return self.insufficient_data(reg_mask, out_image_file, save_mask)
        # [CHECK]: region dimensionality
        if reg_mask.sum() == 0:
            print(f'[Error] The mask and the region is inconsistent')
            return

        coords = coords_all[cp_mask]
        print("selected data length:", len(coords))
        
        coords_int = np.floor(coords).astype('int')
        feats = feats_all[cp_mask]
#        n_rows = feats.shape[0]
#        if n_rows % 5 != 0:
#            new_n_rows = n_rows - (n_rows % 5) 
#            feats = feats[:new_n_rows] 

        nzcoords = reg_mask.nonzero()
        nzcoords_t = np.array(nzcoords).transpose()
        # We should only processing the mask region, so pre-extracting it.
        zmin, ymin, xmin = nzcoords_t.min(axis=0)
        zmax, ymax, xmax = nzcoords_t.max(axis=0)
        reg_sub_mask = reg_mask[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]

        # [CHECK]: region dimensionality:
        # if a region is very thin in one direction, it should be ignored as 
        # the data points could be erroneously registered
        ero = erosion(reg_sub_mask, morphology_cube(self.min_max_width))
        if ero.sum() <= 0:
            print(f'[Warning] The mask of region={regid} is two smaller, we could not rely on it, just skip')
            return self.insufficient_data(reg_mask, out_image_file, save_mask)

        communities, comms = self.community_detection(coords, feats, reg_sub_mask, n_jobs=1)
        if len(communities) == 1:
            cur_mask = reg_mask.astype(np.uint16)
            save_image(out_image_file, cur_mask, useCompression=True)
            print('No subdomain found!')
            return True
        #print('communities, comms:', communities, comms)
        if self.debug:
            dfp = self.df[cp_mask].copy()
            dfp['parc'] = comms
            dfp['soma_x'] -= zmin
            dfp['soma_y'] -= ymin
            dfp['soma_z'] -= xmin
            self.visualize_on_ccf(dfp, reg_sub_mask, prefix=f'temp_r{rprefix}')
    
        min_pts_per_comm = np.sqrt(coords.shape[0])

        # estimate the weighted center of each community
        mcoords = []    # weighted center of filtered communities
        mnodes = []     # filtered nodes
        mcomms = []     # community index for each node
        for icomm, inodes in communities.items():
            #if len(inodes) < min_pts_per_comm:
            #    continue
            cur_coords = coords.iloc[inodes]
            mnodes.extend(inodes)
            mcomms.extend([icomm]*len(inodes))
            mcoord = cur_coords.mean(axis=0).values
            mcoords.append(mcoord)
        mcoords = np.array(mcoords)


        if mcoords.shape[0] == 0:
            print(f'[Warning] Insufficient data to detect sub-regions for regid={regid}!')
            return self.insufficient_data(reg_mask, out_image_file, save_mask)

        parc_method = 'NearestNeighbor'
        if parc_method == 'Voronoi':
            dms, dmi = min_distances_between_two_sets(nzcoords_t, mcoords, topk=1, reciprocal=False, return_index=True, tree_type='BallTree')
            if self.debug:
                cmask = random_colorize(nzcoords_t, dmi[:,0], reg_mask.shape, dmi.max())
            # The following 2 lines of codes are added without verification!
            cur_mask = reg_mask.astype(uint16)
            cur_mask[nzcoords] = dmi[:,0]
        elif (parc_method == 'NearestNeighbor') or (parc_method == 'KNN'):
            if parc_method == 'NearestNeighbor':
                interp = NearestNDInterpolator(coords.iloc[mnodes][['soma_x', 'soma_y', 'soma_z']].values, mcomms)
                predv = interp(*nzcoords)
            elif parc_method == 'KNN':
                knn = KNeighborsClassifier(n_neighbors=3)
                knn.fit(coords.iloc[mnodes][['soma_x', 'soma_y', 'soma_z']].values, mcomms)
                predv = knn.predict(nzcoords_t)
            else:
                raise NotImplementedError
                
            cur_mask = reg_mask.astype(np.uint16)
            cur_mask[nzcoords] = predv
            
            sub_mask = cur_mask[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
            if self.debug:
                print(f'[Interpolation]: {time.time() - t0:.2f} seconds')

            if self.debug:
                print(f'[Initial No. of CCs]: {len(np.unique(cc3d.connected_components(sub_mask, connectivity=26)))-1}')

            # median filtering
            # Python version on CPU. GPU implementation using PyTorch should be much faster, 
            # we can refer to https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598 
            # for more specific implementation.
            sub_fg_mask = sub_mask > 0

            # 2 rounds of parcellation smoothing
            cc_mask, cc_ids, cc_cnts = None, None, None
            for i_interp in range(2):
                sub_mask = majority_filter(sub_mask, morphology_ball(5), mask=sub_fg_mask)
                sub_mask[~sub_fg_mask] = 0  # force to zero

                if sub_fg_mask.sum() > min_pts_per_parc:
                    sub_mask, cc_mask, cc_ids, cc_cnts = self.remove_small_parcs(sub_mask, sub_fg_mask, cc_mask, cc_ids, cc_cnts, min_pts_per_parc)
                if self.debug:
                    print(f'[No. of CCs for iter={i_interp}]: {cc_mask.max()}')
                    print(cc_ids, cc_cnts)

            # Ensure no small parcellations
            niter = 0
            while (sub_fg_mask.sum() > min_pts_per_parc) and ((cc_cnts<min_pts_per_parc).sum() > 0):
                if niter < 2:
                    sub_mask, cc_mask, cc_ids, cc_cnts = self.remove_small_parcs(sub_mask, sub_fg_mask, cc_mask, cc_ids, cc_cnts, min_pts_per_parc)
                else:
                    #sub_mask, cc_mask, cc_ids, cc_cnts = self.remove_small_parcs(sub_mask, sub_fg_mask, cc_mask, cc_ids, cc_cnts, min_pts_per_parc, use_cc_ids=True)
                    break

                print(regid, niter, cc_ids.shape, cc_cnts)
                niter += 1
            

            if self.debug:
                print(f'[No. of CCs finally]: {cc_mask.max()}')
                print(cc_ids, cc_cnts)

            # re-ordering the connected components or parcellations
            reorder_mask_using_cc(sub_mask, cc_mask, sub_fg_mask, min_pts_per_parc)
            
            cur_mask[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1] = sub_mask

            if self.debug: 
                cmask = random_colorize(nzcoords_t, cur_mask[nzcoords], self.mask.shape, int(predv.max()))

        if self.debug:
            dfp = self.df[cp_mask].copy()
            print(dfp.index)
            cur_comms = cur_mask[coords_int.values[:,0], coords_int.values[:,1], coords_int.values[:,2]]
            dfp['parc'] = cur_comms
            dfp['soma_x'] -= zmin
            dfp['soma_y'] -= ymin
            dfp['soma_z'] -= xmin
            # processing the out-boundary points caused by mirroring and flooring
            nz_parc = dfp['parc'] == 0


        if self.debug:
            cmask = cmask[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
            self.save_colorized_images(cmask, reg_sub_mask, sub_mask, out_image_file)
        
        if save_mask:
            save_image(out_image_file, cur_mask, useCompression=True)

        print(f'<--- [{len(np.unique(sub_mask))-1}] sub-regions for region index={regid} found in {time.time() - t0:.2f} seconds')
        return cur_mask


    def parcellate_brain(self, n_jobs=8):
        if self.r314_mask:
            regions = REGION314
        else:
            regions = TEST_REGION

        zyxs = self.df[['soma_x', 'soma_y', 'soma_z']]
        zyxs_int = np.floor(zyxs).astype(np.int32)
        zs, ys, xs = zyxs_int.transpose().values
        regids = self.mask[zs, ys, xs]
        rids, cnts = np.unique(regids, return_counts=True)
        # run parcellation each region separately in parallel
        save_mask = True
        args_list = []
        for rid in regions:
            # make sure we estimate only necessary regions
            #if rid in regions:
            if not os.path.exists(os.path.join(self.out_mask_dir, f'parc_region{rid}.nrrd')):
                args_list.append((rid, save_mask))
        print(f'No. of regions to calculate: {len(args_list)}')
        
        
        # multiprocessing
        pt = Pool(n_jobs)
        pt.starmap(self.parcellate_region, args_list)
        pt.close()
        pt.join()

    def merge_parcs(self, parc_file):
        mask = self.mask.copy()
        mask.fill(0)
        cur_id = 0
        cnt = 0
        t0 = time.time()
        parcs2ccf = {}
        for pfile in glob.glob(os.path.join(self.out_mask_dir, '*nrrd')):
            prefix = os.path.splitext(os.path.split(pfile)[-1])[0]
            regid = prefix[11:]
            # load the parcellation file for current region
            cur_mask = load_image(pfile)
            nzm = cur_mask != 0
            mask[nzm] = cur_mask[nzm] + cur_id
            # update the correspondence from parcellation to ccf atlas
            for pid in range(1, cur_mask.max()+1):
                parcs2ccf[pid+cur_id] = int(regid)

            cur_id += cur_mask.max()
            print(regid, cur_mask.max(), mask.max())

            cnt += 1
            if cnt % 10 == 0:
                print(f'===> Processed {cnt} regions in {time.time() - t0:.2f} seconds')

        save_image(parc_file, mask, useCompression=True)
        # save the correspondence file
        with open(f'{parc_file}.pkl', 'wb') as fp:
            pickle.dump(parcs2ccf, fp)

    def plot_parc_sections(self, parc_file, plot_regid=False):
        parc = load_image(parc_file)
        # get the subregion
        reg_mask = parc > 0
        nzcoords = reg_mask.nonzero()
        nzcoords_t = np.array(nzcoords).transpose()
        # We should only processing the mask region, so pre-extracting it.
        zmin, ymin, xmin = nzcoords_t.min(axis=0)
        zmax, ymax, xmax = nzcoords_t.max(axis=0)
        reg_sub_mask = reg_mask[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
        reg_sub_parc = parc[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
        #print(parc.max())
        cmask = random_colorize(nzcoords_t, parc[nzcoords], self.mask.shape, parc.max())
        #print(cmask.nonzero())
        sub_parc = cmask[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
        self.save_colorized_images(sub_parc, reg_sub_mask, reg_sub_parc, parc_file, thickX2=10, plot_regid=plot_regid)
        #print(cmask.shape)  
        #print(np.unique(cmask))
        print("finished")
    
if __name__ == '__main__':
    mefile = './data/mefeatures_130K.csv'
    scale = 1.
    feat_type = 'full'  # mRMR, PCA, full
    debug = True
    #regid = [998,997,49,30]
    #regid = [997]
    regid = 1001
    r314_mask = False
    
    if r314_mask:
        parc_dir = f'output_{feat_type.lower()}'
        parc_file = f'intermediate_data/parc_{feat_type.lower()}.nrrd'
    else:
        parc_dir = f'output_{feat_type.lower()}'
        parc_file = f'intermediate_data/parc_{feat_type.lower()}.nrrd'
    #parc_dir = 'Tmp'
    
    bp = BrainParcellation(mefile, scale=scale, feat_type=feat_type, r314_mask=r314_mask, debug=debug, out_mask_dir=parc_dir)
    #bp.parcellate_region(regid=regid)
    bp.parcellate_brain()
    bp.merge_parcs(parc_file=parc_file)   #generate parc_full.nrrd AND parc_full.nrrd.pkl
    
    #bp.plot_parc_sections('output_full/parc_region25.nrrd', plot_regid=False)