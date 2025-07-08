import pandas as pd

#select_region = ['AOTU_L', 'AOTU_R', 'LAL_L', 'LAL_R', 'LA_L', 'LA_R', 'LO_L', 'LO_R',
#    'LOP_L', 'LOP_R', 'ME_L', 'ME_R', 'OCG']
#select_region = ['FB','EB','PB','NO','AMMC_R','AMMC_L','FLA_R','FLA_L','CAN_R','CAN_L','PRW','SAD','GNG','AL_R','AL_L','LH_R','LH_L','MB_CA_R','MB_CA_L','MB_PED_R',
# 'MB_PED_L','MB_VL_R','MB_VL_L','MB_ML_R','MB_ML_L','BU_R','BU_L','GA_R','GA_L','LAL_R','LAL_L','SLP_R','SLP_L','SIP_R','SIP_L','SMP_R','SMP_L',
#'CRE_R','CRE_L','SCL_R','SCL_L','ICL_R','ICL_L','IB_R','IB_L','ATL_R','ATL_L','VES_R','VES_L','EPA_R','EPA_L','GOR_R','GOR_L','SPS_R','SPS_L',
#'IPS_R','IPS_L','AOTU_R','AOTU_L','AVLP_R','AVLP_L','PVLP_R','PVLP_L','PLP_R','PLP_L','WED_R','WED_L','ME_R','ME_L','AME_R','AME_L','LO_R','LO_L',
#'LOP_R','LOP_L','LA_R','LA_L','OCG'                                             
#]
#select_region = ['WED_L', 'WED_R']
select_region = ['LA_L', 'LA_R', 'OCG', 'AL_L', 'ME_L', 'ME_R']

df1 = pd.read_csv('proj_ccf-me_test_copy.csv', dtype={0: str})

df2 = pd.read_csv('/mnt/g/fly/dataframe_data/meta_data.csv', usecols=['Name', 'Soma_region', 'type'], dtype={'Name': str})

df1 = pd.merge(df1, df2, left_on=df1.columns[0], right_on='Name', how='left')

df1.to_csv('merged_me_ccf.csv', index=False)  
df_unipolar = df1[(df1['type'] == 'Unipolar') & (df1['Soma_region'].isin(select_region))]
df_bipolar = df1[(df1['type'] == 'Bipolar') & (df1['Soma_region'].isin(select_region))]
df_multipolar = df1[(df1['type'] == 'Multipolar') & (df1['Soma_region'].isin(select_region))]

df_unipolar = df_unipolar.drop(columns=['type', 'Soma_region'])
df_bipolar = df_bipolar.drop(columns=['type', 'Soma_region'])
df_multipolar = df_multipolar.drop(columns=['type', 'Soma_region'])

df_unipolar.to_csv('filter_data/me_proj_unipolar.csv', index=False)  
df_bipolar.to_csv('filter_data/me_proj_bipolar.csv', index=False)
df_multipolar.to_csv('filter_data/me_proj_multipolar.csv', index=False)
