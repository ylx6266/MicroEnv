import os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

def read_swc(file_path):
    coords = []
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith('#'):  
                parts = line.split()
                if len(parts) >= 5: 
                    x, y, z = float(parts[2]) / 1000, float(parts[3]) / 1000, float(parts[4]) / 1000  
                    coords.append([x, y, z])
    coords = np.unique(np.array(coords), axis=0)  
    return coords

def plot_points(coords, ax=None, color='b', label=None):
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=color, label=label, alpha=0.6)
    
    ax.set_xlabel('X (μm)', fontsize=12)
    ax.set_ylabel('Y (μm)', fontsize=12)
    ax.set_zlabel('Z (μm)', fontsize=12)
    
    return ax

def compute_nearest_neighbor_overlap(pre_coords, post_coords, k=5, threshold=3):
    all_coords = np.vstack((pre_coords, post_coords))  
    tree = cKDTree(all_coords)  
    
    overlap_count = 0
    pre_total = len(pre_coords)
    post_total = len(post_coords)
    
    for i, coord in enumerate(pre_coords):
        dist, indices = tree.query(coord, k + 1)
        overlap_points = np.sum(indices[1:] >= pre_total)
        if overlap_points >= threshold:
            overlap_count += 1
    
    for i, coord in enumerate(post_coords):
        dist, indices = tree.query(coord, k + 1)
        overlap_points = np.sum(indices[1:] < pre_total)
        if overlap_points >= threshold:
            overlap_count += 1

    nn_overlap = overlap_count / (pre_total + post_total)  
    return nn_overlap

def analyze_intersection(pre_neurites_path, post_neurites_path, output_csv):
    pre_files = set(f for f in os.listdir(pre_neurites_path) if f.endswith('.swc'))
    post_files = set(f for f in os.listdir(post_neurites_path) if f.endswith('.swc'))
    common_files = pre_files & post_files  
    
    # Open CSV file for writing results
    with open(output_csv, mode='w', newline='') as csvfile:
        fieldnames = ['SWC_File', 'NN_Overlap']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        for swc_file in tqdm(sorted(common_files), desc="Processing Files"):  
            pre_file_path = os.path.join(pre_neurites_path, swc_file)
            post_file_path = os.path.join(post_neurites_path, swc_file)
            
            pre_coords = read_swc(pre_file_path)   
            post_coords = read_swc(post_file_path)   

            if pre_coords.shape[0] < 4 or post_coords.shape[0] < 4:
                print(f"File: {swc_file} - Not enough points for analysis.")
                continue

            nn_overlap = compute_nearest_neighbor_overlap(pre_coords, post_coords)
            print(f"File: {swc_file} - NN_overlap: {nn_overlap:.4f}")

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax = plot_points(pre_coords, ax=ax, color='r', label='Pre-synapse')
            ax = plot_points(post_coords, ax=ax, color='b', label='Post-synapse')
            
            ax.set_xlabel("X (μm)")
            ax.set_ylabel("Y (μm)")
            ax.set_zlabel("Z (μm)")
            
            file_name_without_extension = os.path.splitext(swc_file)[0]
            ax.set_title(f"ID:{file_name_without_extension}\n(Overlap = {nn_overlap:.3f})", fontsize=16)

            ax.legend()
            plt.show()

            file_name_without_extension = os.path.splitext(swc_file)[0]

            writer.writerow({'SWC_File': file_name_without_extension, 'NN_Overlap': nn_overlap})


pre_neurites_path = r'H:\energency_fly\data\pre_neurites'
post_neurites_path = r'H:\energency_fly\data\post_neurites'
output_csv = r'H:\energency_fly\data\nn_overlap_results.csv'

analyze_intersection(pre_neurites_path, post_neurites_path, output_csv)
