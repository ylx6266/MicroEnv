import pandas as pd

file_path = 'data/fly_region.csv'

df = pd.read_csv(file_path)

first_column = df.iloc[:, 0]

output_path = './data/final_filtered_swc.txt'
first_column.to_csv(output_path, index=False, header=False)

print(f'Saved first column data to {output_path}')

