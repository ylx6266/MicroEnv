import pandas as pd

meta_test = pd.read_csv('meta_test.csv', dtype={0:str})
proj_ccf_me_test_copy = pd.read_csv('classification.csv', dtype={0:str})

merged_df = pd.merge(meta_test, proj_ccf_me_test_copy, on='Name', how='left')

merged_df.to_csv('merged_region_classification.csv', index=False)

#print(merged_df)
