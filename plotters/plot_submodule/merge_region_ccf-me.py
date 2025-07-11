import pandas as pd

meta_test = pd.read_csv('meta_test.csv', dtype={0:str})
proj_ccf_me_test_copy = pd.read_csv('proj_ccf-me_test_copy.csv', dtype={0:str})

merged_df = pd.merge(meta_test, proj_ccf_me_test_copy, on='Name', how='left').dropna()

merged_df.to_csv('merged_region_ccf-me.csv', index=False)

#print(merged_df)
