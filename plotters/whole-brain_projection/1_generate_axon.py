import os
from tqdm import tqdm  

def extract_cell_body(swc_file):
    with open(swc_file, 'r') as f:
        lines = f.readlines()
    
    data_lines = [line for line in lines if not line.startswith('#')]
    
    if data_lines:
        cell_body_line = data_lines[0] 
    else:
        cell_body_line = ""
    
    return cell_body_line

def add_cell_body_to_target(target_swc_file, cell_body_line, save_path):
    with open(target_swc_file, 'r') as f:
        lines = f.readlines()
    
    data_lines = [line for line in lines if not line.startswith('#')]
    
    data_lines.insert(0, cell_body_line)
    
    final_lines = [line for line in lines if line.startswith('#')] + data_lines
    
    with open(save_path, 'w') as f:
        f.writelines(final_lines)

def process_swc_files(path1, path2, output_dir):

    files1 = {f for f in os.listdir(path1) if f.endswith('.swc')}
    files2 = {f for f in os.listdir(path2) if f.endswith('.swc')}
    
    common_files = files1.intersection(files2)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for file in tqdm(common_files, desc="Processing SWC files", unit="file"):
        path1_swc = os.path.join(path1, file)
        path2_swc = os.path.join(path2, file)
        
        cell_body_line = extract_cell_body(path1_swc)
        
        output_swc_path = os.path.join(output_dir, file)
        
        add_cell_body_to_target(path2_swc, cell_body_line, output_swc_path)
        

path1 = '/mnt/g/energency_fly/data/sk_lod1_783_healed' 
path2 = '/mnt/g/energency_fly/data/pre_neurites' 
output_dir = '/mnt/g/energency_fly/data/axon_file' 

process_swc_files(path1, path2, output_dir)
