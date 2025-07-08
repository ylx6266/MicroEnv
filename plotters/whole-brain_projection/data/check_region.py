import pandas as pd

meta_test = pd.read_csv('meta_test.csv', dtype={0: str}).dropna()

mefeatures_100K = pd.read_csv('/home/ylx/fly/code/BrainParcellation-main/BrainParcellation-main/microenviron/data/mefeatures_100K.csv', usecols=['Name', 'soma_region'], dtype={'Name': str})

merged = pd.merge(meta_test, mefeatures_100K, on='Name', how='left', indicator=True)
merged['is_region_matching'] = merged['region_name_ccf'] == merged['soma_region']

deleted_rows = merged[~merged['is_region_matching']]

deleted_count = deleted_rows.shape[0]

meta_test_cleaned = merged[merged['is_region_matching']].drop(columns=['is_region_matching', '_merge', 'soma_region'])

meta_test_cleaned.to_csv('meta_test_cleaned.csv', index=False)