import pandas as pd

file_path = '/home/ylx/fly/dataframe_data/meta_data.csv'

df = pd.read_csv(file_path)
selected_columns = df[['Name', 'Soma_region']]

filtered_columns = selected_columns.dropna()

output_path = './data/fly_region.csv'
filtered_columns.to_csv(output_path, index=False)

print(f'Saved filtered data to {output_path}')





















