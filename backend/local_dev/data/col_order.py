import pandas as pd

df = pd.read_csv('combined_fraud.csv')

column_order = [
    "TransactionID", "TransactionDate", "MerchantID", "Location",
    "Amount", "TransactionType", "distance_from_home", "distance_from_last_transaction",
    "ratio_to_median_purchase_price", "repeat_retailer", "used_chip", "used_pin_number",
    "online_order", "fraud", "IsFraud"
]

df = df[column_order]
print(df.head())

df.to_csv('combined_fraud.csv', index=False)