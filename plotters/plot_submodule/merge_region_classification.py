import pandas as pd

# 读取 CSV 文件
meta_test = pd.read_csv('meta_test.csv', dtype={0:str})
proj_ccf_me_test_copy = pd.read_csv('classification.csv', dtype={0:str})

# 根据 'Name' 列合并两个 DataFrame
merged_df = pd.merge(meta_test, proj_ccf_me_test_copy, on='Name', how='left')

# 如果需要保存合并后的结果，可以使用下面的代码：
merged_df.to_csv('merged_region_classification.csv', index=False)

# 显示合并后的数据框
#print(merged_df)
