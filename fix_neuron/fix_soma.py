import os
import pandas as pd
from collections import defaultdict
import time
from multiprocessing import Pool
from tqdm import tqdm

def read_swc(file_path):
    df = pd.read_csv(file_path, sep='\s+', comment='#', header=None)
    df.columns = ['ID', 'Type', 'X', 'Y', 'Z', 'Radius', 'Parent']
    if not all(col in df.columns for col in ['ID', 'Type', 'X', 'Y', 'Z', 'Radius', 'Parent']):
        raise ValueError("Invalid SWC file")
    return df

def build_parent_child_map(df):
    parent_child_map = defaultdict(list)
    for _, row in df.iterrows():
        parent_child_map[row['Parent']].append(row['ID'])
    return parent_child_map

def get_branch_length_optimized(df, node_id, parent_child_map, length_cache=None):
    if length_cache is None:
        length_cache = {}

    if node_id in length_cache:
        return length_cache[node_id]
    
    MAX_DEPTH = 3000
    stack = [(node_id, 0)]  
    visited_set = set()
    total_length = 0

    while stack:
        current_node_id, depth = stack.pop()

        if depth >= MAX_DEPTH:
            length_cache[node_id] = MAX_DEPTH  
            return MAX_DEPTH

        if current_node_id in visited_set:
            continue
        visited_set.add(current_node_id)
        total_length += 1

        for child_id in parent_child_map.get(current_node_id, []):
            stack.append((child_id, depth + 1))

    length_cache[node_id] = total_length  
    return total_length

def count_main_branches_optimized(df):
    soma_info = df[df['Type'] == 1]
    if soma_info.empty:
        return None, []  
    soma_id = soma_info.iloc[0]['ID']
    branches = df[df['Parent'] == soma_id]
    
    if len(branches) == 1:
        return soma_id, [0]  

    branch_lengths = []
    parent_child_map = build_parent_child_map(df)
    length_cache = {}  

    for branch_id in branches['ID']:
        length = get_branch_length_optimized(df, branch_id, parent_child_map, length_cache)
        branch_lengths.append(length)

    return soma_id, branch_lengths

def remove_isolated_fragments(df):
    components = find_connected_components(df)
    largest_component = max(components, key=len)
    return df[df['ID'].isin(largest_component)]

def find_connected_components(df):
    parent_child_map = build_parent_child_map(df)
    visited = set()
    components = []

    def dfs_iterative(start_node):
        stack = [start_node]
        component = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for child_id in parent_child_map.get(node, []):
                stack.append(child_id)
            for parent_id, children in parent_child_map.items():
                if node in children:
                    stack.append(parent_id)
        return component

    for node_id in df['ID']:
        if node_id not in visited:
            component = dfs_iterative(node_id)
            components.append(component)

    return components

def process_swc_file_optimized(args):
    file_path, output_folder = args
    try:
        start_time = time.time()

        swc_data = read_swc(file_path)
        parent_child_map = build_parent_child_map(swc_data)

        soma_id, branch_lengths = count_main_branches_optimized(swc_data)

        if soma_id is None:
            return None  

        if branch_lengths:
            if len(branch_lengths) > 1:
                max_branch_length = max(branch_lengths)
                min_length = max_branch_length * 0.15 

                branches_from_soma = swc_data[swc_data['Parent'] == soma_id]
                valid_nodes = set(swc_data['ID'])

                for branch_id, branch_length in zip(branches_from_soma['ID'], branch_lengths):
                    if branch_length < min_length:
                        valid_nodes.remove(branch_id)

                swc_data = swc_data[swc_data['ID'].isin(valid_nodes)]
                swc_data = remove_isolated_fragments(swc_data)

            new_file_path = os.path.join(output_folder, f'fixed_{os.path.basename(file_path)}')
            swc_data.to_csv(new_file_path, sep=' ', index=False, header=False, float_format='%.3f')

        end_time = time.time()
        return new_file_path
    except Exception as e:
        return None

def process_swc_folder(folder_path, output_folder, max_processes=4):
    results = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.swc')]

    with Pool(processes=max_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_swc_file_optimized, [(os.path.join(folder_path, filename), output_folder) for filename in files]), total=len(files)))

    return [result for result in results if result]

input_folder_path = "data"
output_folder_path = "data_fix"

max_processes = 15

os.makedirs(output_folder_path, exist_ok=True)

processed_files = process_swc_folder(input_folder_path, output_folder_path, max_processes=max_processes)
