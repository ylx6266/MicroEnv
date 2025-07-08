import os
import pandas as pd
from tqdm import tqdm

axon_dir = '/home/ylx/fly/axon_file'
csv_file = '/home/ylx/fly/code/BrainParcellation-main/BrainParcellation-main/microenviron/data/mefeatures_100K.csv'
output_dir = '/home/ylx/fly/axon_file/axon_sink_soma'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df = pd.read_csv(csv_file, dtype={1:str}).dropna()

csv_names = set(df['Name'].unique())

swc_files = [filename for filename in os.listdir(axon_dir) if filename.endswith('.swc')]

for filename in tqdm(swc_files, desc="Processing SWC files", unit="file"):
    file_base_name = os.path.splitext(filename)[0]  
    
    if file_base_name in csv_names:
        file_path = os.path.join(axon_dir, filename)
        
        row = df[df['Name'] == file_base_name]
        
        if not row.empty:
            soma_x, soma_y, soma_z = row[['soma_x', 'soma_y', 'soma_z']].values[0] * 1000  
            
            with open(file_path, 'r') as swc_file:
                lines = swc_file.readlines()

            lines[0] = lines[0].split()
            lines[0][2] = str(soma_x)
            lines[0][3] = str(soma_y)
            lines[0][4] = str(soma_z)
            lines[0] = ' '.join(lines[0]) + '\n'

            output_file_path = os.path.join(output_dir, filename)

            with open(output_file_path, 'w') as swc_file:
                swc_file.writelines(lines)
            
            # print(f"Updated SWC file saved to: {output_file_path}")
        else:
            print(f"CSV data for {file_base_name} is missing soma coordinates.")
#    else:
#        print(f"File {filename} not found in CSV Name column.")
