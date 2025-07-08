##########################################################
#Author:          Yufeng Liu
#Create time:     2024-08-01
#Description:               
##########################################################
import os
import time
import numpy as np
import pandas as pd

from anatomy.anatomy_config import MASK_CCF25_FILE, TEST_REGION
from anatomy.anatomy_core import parse_ana_tree
from file_io import load_image

class Projection:

    def __init__(self, use_two_hemispheres=True, resample_scale=8, atlas_file=None):
        # make sure the swc are uniformly sampled, otherwise the estimation
        # should be changed
        self.resample_scale = resample_scale
        # Get the atlas
        if atlas_file is None:
            atlas = load_image(MASK_CCF25_FILE)
            self.ccf_atlas = True
        else:
            atlas = load_image(atlas_file)
            self.ccf_atlas = False

        if use_two_hemispheres:
            # get the new atlas with differentiation of left-right hemisphere
            zdim, ydim, xdim = atlas.shape
            atlas_lr = np.zeros(atlas.shape, dtype=np.int64)
            atlas_lr[:zdim//2] = atlas[:zdim//2]
            atlas_lr[zdim//2:] = -atlas[zdim//2:].astype(np.int64)
            self.atlas_lr = atlas_lr
        else:
            self.atlas_lr = atlas

    def calc_proj_matrix(self, axon_files, proj_csv='temp.csv'):
        zdim, ydim, xdim = self.atlas_lr.shape
        
        # vector
        regids = np.unique(self.atlas_lr[self.atlas_lr != 0])
        rdict = dict(zip(regids, range(len(regids))))
        print(rdict)

        fnames = [os.path.split(fname)[-1][:-4] for fname in axon_files]
        #print(fnames)
        projs = pd.DataFrame(np.zeros((len(axon_files), len(regids))), index=fnames, columns=regids)
        print(projs)
        t0 = time.time()
        for iaxon, axon_file in enumerate(axon_files):
            ncoords = pd.read_csv(axon_file, sep=' ', usecols=(2,3,4,6), comment='#', header=None).values
            # flipping
            smask = ncoords[:,-1] == -1
            #print(smask)
            if smask.sum() == 0:
                print(axon_file)
            # convert to CCF-25um
            ncoords[:,:-1] = ncoords[:,:-1] / 2000. + ([-85.8770447/2, -63.0468216/2, 3.3968091/2])   #changed
            soma_coord = ncoords[smask][0,:-1]
            ncoords = ncoords[~smask][:,:-1]
            if soma_coord[2] > zdim/2:
                ncoords[:,2] = zdim - ncoords[:,2]
            # make sure no out-of-mask points
            ncoords = np.round(ncoords).astype(int)
            ncoords[:,0] = np.clip(ncoords[:,0], 0, zdim-1)
            ncoords[:,1] = np.clip(ncoords[:,1], 0, ydim-1)
            ncoords[:,2] = np.clip(ncoords[:,2], 0, xdim-1)
            # get the projected regions
            proj = self.atlas_lr[ncoords[:,0], ncoords[:,1], ncoords[:,2]]
            # to project matrix
            rids, rcnts = np.unique(proj, return_counts=True)
            # Occasionally, there are some nodes located outside of the atlas, due to
            # the registration error
            nzm = rids != 0
            rids = rids[nzm]
            rids = [rid for rid in rids if rid is not None and rid != '']
            #print(rids)
            rcnts = rcnts[nzm]
            #print(rcnts)
            rindices = np.array([int(rdict[rid]) for rid in rids if rid != 0])
            #print("test")
            try:
                projs.iloc[iaxon, rindices] = rcnts
            except:
                continue
 
            if (iaxon + 1) % 10 == 0:
                print(f'--> finished {iaxon+1} in {time.time()-t0:.2f} seconds')

        projs *= self.resample_scale # to um scale

        # zeroing non-salient regions
        if self.ccf_atlas:
            salient_mask = np.array([True if np.fabs(int(col)) in TEST_REGION else False for col in projs.columns])
            #keep_mask = (projs.sum() > 0) & salient_mask
            keep_mask = salient_mask
            # filter the neurons not in target regions
            projs = projs.loc[:, keep_mask]
        
        projs.to_csv(proj_csv)

        return projs

def preprocess_projections(projs, min_proj=1000., log=True, remove_non_proj_neuron=False,
                           keep_only_salient_regions=True, is_ccf_atlas=True):
    # convert the column dtype to integar
    projs.columns = projs.columns.astype(int)

    if min_proj > 0:
        projs = projs.copy()
        projs[projs < min_proj] = 0

    if keep_only_salient_regions and is_ccf_atlas:
        from anatomy.anatomy_config import TEST_REGION

        # NOTE that this only works for projection mapping in CCF regions now!
        salient_col_m = projs.columns.isin(TEST_REGION) | projs.columns.isin([-i for i in TEST_REGION])
        projs = projs.loc[:, projs.columns[salient_col_m]]

    # remove regions without real projection
    projs = projs.loc[:, projs.columns[projs.sum(axis=0) != 0]]
    if remove_non_proj_neuron:
        projs = projs.iloc[np.nonzero(projs.sum(axis=1))[0]]

    # convert to log space
    projs = np.log(projs+1)
    return projs


