import pandas as pd

# Load your data
data = pd.read_csv('transaction_dataset.csv')

# Display the first few rows of the dataset
print(data.head())
print(data.columns)

# Remove duplicates
data.drop_duplicates(inplace=True)

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Example feature engineering
data['transaction_amount'] = data['amount']
data['transaction_hour'] = pd.to_datetime(data['timestamp']).dt.hour

# Save the processed data
data.to_csv('preprocessed_transactions.csv', index=False)
