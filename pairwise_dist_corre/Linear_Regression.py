import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

file_path1 = "/mnt/h/fly/dataframe_data/classification.csv"
file_path2 = "/mnt/h/fly/dataframe_data/mefeatures_100K.csv"
file_path3 = "/mnt/h/fly/dataframe_data/proj_ccf-me_test_copy.csv"
file_path4 = "/mnt/h/fly/dataframe_data/proj_test_copy.csv"
file_path5 = "/mnt/h/fly/dataframe_data/meta_data.csv"

Optic_Lobe = ['AME_L','AME_R','LA_L','LA_R','LO_L','LO_R','LOP_L','LOP_R','ME_L','ME_R','OCG']
Lateral_Complex = ['BU_L','BU_R','GA_L','GA_R']
Lateral_Horn = ['LH_L','LH_R']
Periesophageal_Neuropils = ['CAN_L','AMMC_L','FLA_L','CAN_R','AMMC_R','FLA_R','SAD','PRW'] 
Mushroom_body = ['MB_CA_L','MB_CA_R','MB_VL_R','MB_VL_L','MB_ML_L','MB_ML_R','MB_PED_L','MB_PED_R']
Inferior_Neuropils = ['ICL_L','ICL_R','IB_L','IB_R','ATL_L','ATL_R','CRE_L','CRE_R','SCL_L','SCL_R']
Ventromedial_Neuropils = ['VES_L','VES_R','GOR_L','GOR_R','SPS_L','SPS_R','IPS_L','IPS_R','EPA_L','EPA_R']
Antennal_Lobe = ['AL_L','AL_R']
Superior_Neuropils = ['SLP_L','SLP_R','SIP_L','SIP_R','SMP_L','SMP_R']
Ventrolateral_Neuropils = ['AVLP_L','AVLP_R','PVLP_L','PVLP_R','WED_L','WED_R','PLP_L','PLP_R','AOTU_L','AOTU_R']
Central_complex = ['NO','PB','EB','FB']
Gnathal_Ganglia = ['GNG']

CNS = ['AME_L','AME_R','LO_L','LO_R','LOP_L','LOP_R','BU_L','BU_R','GA_L','GA_R','LH_L','LH_R','CAN_L','AMMC_L','FLA_L','CAN_R','AMMC_R','FLA_R',
       'MB_CA_L','MB_CA_R','MB_VL_R','MB_VL_L','MB_ML_L','MB_ML_R','MB_PED_L','MB_PED_R','ICL_L','ICL_R','IB_L','IB_R','ATL_L','ATL_R','CRE_L','CRE_R','SCL_L','SCL_R',
       'VES_L','VES_R','GOR_L','GOR_R','SPS_L','SPS_R','IPS_L','IPS_R','EPA_L','EPA_R','AL_L','AL_R','SLP_L','SLP_R','SIP_L','SIP_R','SMP_L','SMP_R',
       'AVLP_L','AVLP_R','PVLP_L','PVLP_R','WED_L','WED_R','PLP_L','PLP_R','AOTU_L','AOTU_R','NO','PB','EB','FB','SAD','PRW','GNG','OCG']
PNS = ['ME_L','ME_R','LA_L','LA_R','OCG']

#region = 'Lateral_Complex'

meta_data_df = pd.read_csv(file_path5)
meta_data_df = meta_data_df[meta_data_df["Soma_region"].isin(["OCG"])]
classification_df = pd.read_csv(file_path1)
me_feature_df = pd.read_csv(file_path2)
me_feature_df = pd.merge(me_feature_df, meta_data_df, on='Name', how='left').dropna()
#print(me_feature_df)
proj_ccf_me_df = pd.read_csv(file_path3)
proj_ccf_df = pd.read_csv(file_path4)


me_columns = [col for col in me_feature_df.columns if col.endswith('_me')]
single_columns = [item.replace('_me', '') for item in me_columns]
ccf_me_columns = proj_ccf_me_df.select_dtypes(include=['number']).columns.tolist()
ccf_me_columns = [col for col in ccf_me_columns if col != 'Name']
ccf_columns = proj_ccf_df.select_dtypes(include=['number']).columns.tolist()
ccf_columns = [col for col in ccf_columns if col != 'Name']

merge_me_ccf_me = pd.merge(proj_ccf_me_df, me_feature_df, on='Name', how='left').dropna()
merge_me_ccf = pd.merge(proj_ccf_df, me_feature_df, on='Name', how='left').dropna()

X = merge_me_ccf_me[me_columns + single_columns]
y = merge_me_ccf_me[ccf_me_columns]

y_transformed = y.apply(lambda row: row.apply(lambda x: np.log1p(x)), axis=1)


scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_standardized, y_transformed, test_size=0.3, random_state=42)

X1_train = X_train[:, :len(me_columns)]
X1_test = X_test[:, :len(me_columns)]
X2_train = X_train[:, len(me_columns):]
X2_test = X_test[:, len(me_columns):]

#########################################################################
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

def train_and_evaluate_linear(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    return mse, r2, cv_scores.mean()

# Train and evaluate for X1 (me_columns)
mse_X1, r2_X1, _ = train_and_evaluate_linear(X1_train, X1_test, y_train, y_test)

# Train and evaluate for X2 (single_columns)
mse_X2, r2_X2, _ = train_and_evaluate_linear(X2_train, X2_test, y_train, y_test)

print("Linear Regression performance comparison:")
print(f"X1 (me_columns) - MSE: {mse_X1:.4f}, R2: {r2_X1:.4f}")
print(f"X2 (single_columns) - MSE: {mse_X2:.4f}, R2: {r2_X2:.4f}")