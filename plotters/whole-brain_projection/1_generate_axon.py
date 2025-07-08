import os
from tqdm import tqdm  # 导入 tqdm

def extract_cell_body(swc_file):
    """
    提取SWC文件中的胞体行。假设胞体行通常是SWC文件中的第一个数据节点
    """
    with open(swc_file, 'r') as f:
        lines = f.readlines()
    
    # 过滤掉注释行，假设注释行以 '#' 开头
    data_lines = [line for line in lines if not line.startswith('#')]
    
    # 假设胞体行是文件中的第一行数据
    if data_lines:
        cell_body_line = data_lines[0]  # 根据实际情况可能需要调整
    else:
        cell_body_line = ""
    
    return cell_body_line

def add_cell_body_to_target(target_swc_file, cell_body_line, save_path):
    """
    将胞体行添加到目标SWC文件的第一行，并保存到新路径
    """
    with open(target_swc_file, 'r') as f:
        lines = f.readlines()
    
    # 过滤掉注释行
    data_lines = [line for line in lines if not line.startswith('#')]
    
    # 在第一行插入胞体行
    data_lines.insert(0, cell_body_line)
    
    # 合并注释行与数据行
    final_lines = [line for line in lines if line.startswith('#')] + data_lines
    
    # 保存文件到新路径
    with open(save_path, 'w') as f:
        f.writelines(final_lines)

def process_swc_files(path1, path2, output_dir):
    """
    处理路径1和路径2中的SWC文件，找到交集并添加胞体行，保存到新的路径
    """
    # 获取路径1和路径2中的所有SWC文件
    files1 = {f for f in os.listdir(path1) if f.endswith('.swc')}
    files2 = {f for f in os.listdir(path2) if f.endswith('.swc')}
    
    # 获取交集文件
    common_files = files1.intersection(files2)
    
    # 确保输出路径存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用 tqdm 包装交集文件列表，添加进度条
    for file in tqdm(common_files, desc="Processing SWC files", unit="file"):
        path1_swc = os.path.join(path1, file)
        path2_swc = os.path.join(path2, file)
        
        # 提取路径1中SWC文件的胞体行
        cell_body_line = extract_cell_body(path1_swc)
        
        # 设置保存路径
        output_swc_path = os.path.join(output_dir, file)
        
        # 将胞体行添加到路径2中相应的SWC文件的第一行，并保存到新路径
        add_cell_body_to_target(path2_swc, cell_body_line, output_swc_path)
        
    print(f"处理完毕，所有文件已保存至 {output_dir}")

# 示例路径
path1 = '/mnt/g/energency_fly/data/sk_lod1_783_healed'  # 路径1
path2 = '/mnt/g/energency_fly/data/pre_neurites'  # 路径2
output_dir = '/mnt/g/energency_fly/data/axon_file'  # 输出目录

process_swc_files(path1, path2, output_dir)
