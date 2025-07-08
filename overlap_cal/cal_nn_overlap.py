import pandas as pd

proj_df = pd.read_csv("dataframe_data/proj_ccf-me_test_copy.csv", dtype={0: str})
meta_df = pd.read_csv("dataframe_data/meta_data_mannual_check.csv", dtype={0: str})
nn_df = pd.read_csv("dataframe_data/nn_overlap_results.csv", dtype={0: str})

meta_subset = meta_df[['Name', 'Soma_region']]

merged_df = proj_df.merge(meta_subset, on='Name', how='left')

final_merged_df = merged_df.merge(nn_df, on='Name', how='left').dropna()

final_merged_df = final_merged_df.loc[:, ~final_merged_df.columns.str.match(r'^\d+$')]

print(final_merged_df.head())

final_merged_df.to_csv("cleaned_nn_overlap.csv", index=False)
