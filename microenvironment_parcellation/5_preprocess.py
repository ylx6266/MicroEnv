#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : preprocess.py
#   Author       : Yufeng Liu
#   Date         : 2023-01-17
#   Description  : Aggregate the informations
#
#================================================================

import os
import time
import numpy as np
import pandas as pd
from sklearn import decomposition


def aggregate_information(swc_dir, gf_file, reg_file, soma_file, outfile):
    DEBUG = False

    df_feat = pd.read_csv(gf_file, index_col=0)  # d22
    df_reg = pd.read_csv(reg_file, index_col=0, skiprows=1, names=['Name', 'soma_region'])
    df_soma = pd.read_csv(soma_file, index_col=0, sep=' ', names=['Name', 'soma_x', 'soma_y', 'soma_z'])

    # Processing df_reg
    df_reg_new_indices = [int(iname) for iname in df_reg.index]
    df_reg.set_index(pd.Index(df_reg_new_indices), inplace=True)
    df_reg
    # Merge the brain and region information
    df_feat = df_feat[df_feat.index.isin(df_reg.index)]
    df = df_reg.merge(df_feat, left_index=True, right_index=True)
    # Processing df_soma
    df_soma_new_indices = [int(iname) for iname in df_soma.index]
    df_soma.set_index(pd.Index(df_soma_new_indices), inplace=True)
    # Merge
    df = df.merge(df_soma, left_index=True, right_index=True)

    # Get additional features using PCA
    new_cols = []
    t0 = time.time()
    for irow, row in df.iterrows():
        # Estimate the isotropy
        swcfile = os.path.join(swc_dir, f'{irow}.swc')
        coords = np.genfromtxt(swcfile, usecols=(2, 3, 4))
        pca = decomposition.PCA()
        pca.fit(coords)
        new_cols.append((*pca.components_[0], *pca.explained_variance_ratio_))

        if len(new_cols) % 100 == 0:
            print(f'[{len(new_cols)}]: {time.time() - t0:.2f} seconds')
            if DEBUG: break

    tmp_index = df.index[:len(new_cols)] if DEBUG else df.index
    new_cols = pd.DataFrame(new_cols, 
        columns=['pc11', 'pc12', 'pc13', 'pca_vr1', 'pca_vr2', 'pca_vr3'],
        index=tmp_index
    ).astype({
        'pc11': np.float32,
        'pc12': np.float32,
        'pc13': np.float32,
        'pca_vr1': np.float32,
        'pca_vr2': np.float32,
        'pca_vr3': np.float32
    })

    tmp_df = df[:len(new_cols)] if DEBUG else df
    df = df.merge(new_cols, left_index=True, right_index=True)

    df.to_csv(outfile, float_format='%g')


if __name__ == '__main__':
    swc_dir = '/home/ylx/fly/data_rename'
    gf_file = 'data/fly_resampled.csv'
    reg_file = 'data/fly_region.csv'
    soma_file = 'data/soma_coordinates.txt'
    outfile = 'data/morph_features_d28.csv'
    aggregate_information(swc_dir, gf_file, reg_file, soma_file, outfile)