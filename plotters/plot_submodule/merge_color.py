import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

meta_test = pd.read_csv('region_colors.csv')
proj_ccf_me_test_copy = pd.read_csv('meta_test.csv', dtype={0: str})

merged_df = pd.merge(meta_test, proj_ccf_me_test_copy, on='region_id_me', how='left')

#merged_df['region_name_ccf'] = merged_df['region_name_ccf'].str.split('_').str[0]

merged_df.to_csv('merged_color.csv', index=False)
merged_df = pd.read_csv('merged_color.csv')


unique_region_names = merged_df['region_name_ccf'].unique()

color_cycle = cycle(plt.cm.tab20.colors)  
region_to_color = {region: next(color_cycle) for region in unique_region_names}

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

region_to_hex_color = {region: rgb_to_hex(color) for region, color in region_to_color.items()}

merged_df['Color'] = merged_df['region_name_ccf'].map(region_to_hex_color)
print(merged_df[['region_name_ccf', 'Color']].drop_duplicates())


merged_df.to_csv('merged_color.csv', index=False)