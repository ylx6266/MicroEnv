##########################################################
#Author:          Yufeng Liu
#Create time:     2024-05-17
#Description:               
##########################################################
import os
import sys
import glob
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
#import multiprocessing

from global_features import calc_global_features_from_folder


def get_features_for_brains(swc_dir):
    all_list = glob.glob(os.path.join(swc_dir, '*.swc'))
    all_file_name = [os.path.splitext(os.path.basename(swc_file))[0] for swc_file in all_list]
    finished_list = glob.glob(os.path.join('data/tmp', '*.csv'))
    finished_file_name = [os.path.splitext(os.path.basename(swc_file))[0] for swc_file in finished_list]
    unfinished_file_name = list(set(all_file_name) - set(finished_file_name))
    args_list = [os.path.join(swc_dir, f'{name}.swc') for name in unfinished_file_name]
    print(len(args_list))

    # multiprocessing
    with Pool(processes=15) as pool:
        results = pool.map(calc_global_features_from_folder, args_list)


def merge_all_brains(csv_dir):
    for i, csv_file in enumerate(sorted(glob.glob(os.path.join(csv_dir, '*.csv')))):
        print(i, os.path.split(csv_file)[-1])
        dfi = pd.read_csv(csv_file, index_col=None)
        if i == 0:
            df = dfi
        else:
            df = pd.concat([df, dfi], ignore_index=True)
        
    new_column_names = ['Name', 'Nodes', 'SomaSurface', 'Stems', 'Bifurcations', 'Branches', 'Tips', 
    'OverallWidth', 'OverallHeight', 'OverallDepth', 'AverageDiameter', 'Length', 
    'Surface', 'Volume', 'MaxEuclideanDistance', 'MaxPathDistance', 'MaxBranchOrder', 
    'AverageContraction', 'AverageFragmentation', 'AverageParent-daughterRatio', 
    'AverageBifurcationAngleLocal', 'AverageBifurcationAngleRemote', 'HausdorffDimension']

    df.columns = new_column_names
    #df.rename({'Unnamed: 0': 'Name'}, axis=1, inplace=True)
    df.to_csv('fly_resampled.csv', index=False)
        

if __name__ == '__main__':
    
    #get_features_for_brains('/home/ylx/fly/data_resample')
    
    merge_all_brains('data/tmp')


