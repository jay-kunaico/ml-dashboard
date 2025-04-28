import pandas as pd

cc_fraud = pd.read_csv('cc_fraud.csv')
new_data = pd.read_csv('card_transdata.csv')
# print('data1 type:', type(cc_fraud))
# print('data2 type:', type(new_data))

if 'TransactionID' not in new_data.columns:
    new_data['TransactionID'] = range(1, len(new_data)+1)


new_data = new_data.rename(columns={'fraud': 'IsFraud'})
# cc_fraud = cc_fraud.reset_index(drop=True)
# new_data = new_data.reset_index(drop=True)
# print( cc_fraud.columns)
# print(new_data.columns)
# out = pd.merge(cc_fraud, new_data, on='IsFraud')
# print('OUT',out.head())



for col in cc_fraud.columns:
    if col not in new_data.columns:
        new_data[col] = None
        
for col in new_data.columns:
    if col not in cc_fraud.columns:
        cc_fraud[col] = None
        

combined_data = pd.merge(cc_fraud, new_data, on='TransactionID', how='outer')
print('combined data type:', type(combined_data))
print('shape of combined_data:', combined_data.shape)
print(combined_data.head())
print(combined_data.columns)
if isinstance(combined_data, pd.DataFrame):
    # combined_data = combined_data.columns.astype(str)
    # combined_data = combined_data.loc[:, ~combined_data.columns.str.contains('^Unnamed')]
    # combined_data = combined_data.dropna(axis=1, how='all')
    combined_data.to_csv('credit_card_fraud.csv')
else:
    print('Dataframe is not a DataFrame')