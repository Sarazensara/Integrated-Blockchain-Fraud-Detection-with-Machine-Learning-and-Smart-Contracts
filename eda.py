import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
data = pd.read_csv('preprocessed_transactions.csv')

# Distribution of transaction amounts
plt.figure(figsize=(10, 6))
sns.histplot(data['transaction_amount'], bins=50)
plt.title('Distribution of Transaction Amounts')
plt.show()

# Checking correlations
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()
