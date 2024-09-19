import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
data = pd.read_csv('preprocessed_transactions.csv')

# Check data types
print(data.dtypes)

# Convert relevant columns to numeric
data['total Ether sent'] = pd.to_numeric(data['total Ether sent'], errors='coerce')
data['total ether received'] = pd.to_numeric(data['total ether received'], errors='coerce')
# Repeat for other relevant columns as needed

# Drop non-numeric or irrelevant columns
columns_to_drop = [
    'Unnamed: 0', 
    'Index', 
    'Address', 
    'ERC20 most sent token type', 
    'ERC20_most_rec_token_type'
]
data = data.drop(columns=columns_to_drop)

# Check for any remaining non-numeric columns
print(data.dtypes)

# Set up the matplotlib figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# First subplot: Distribution of Total Ether Sent
sns.histplot(data['total Ether sent'], bins=50, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Total Ether Sent')
axes[0, 0].set_xlabel('Total Ether Sent')
axes[0, 0].set_ylabel('Frequency')

# Second subplot: Class Distribution
sns.countplot(x='FLAG', data=data, ax=axes[0, 1])
axes[0, 1].set_title('Class Distribution (Fraud vs Non-Fraud)')
axes[0, 1].set_xlabel('Class (0 = Non-Fraud, 1 = Fraud)')
axes[0, 1].set_ylabel('Count')

# Third subplot: Box Plot for Outliers
sns.boxplot(x='FLAG', y='total Ether sent', data=data, ax=axes[1, 0])
axes[1, 0].set_title('Total Ether Sent by Class')
axes[1, 0].set_xlabel('Class (0 = Non-Fraud, 1 = Fraud)')
axes[1, 0].set_ylabel('Total Ether Sent')

# Fourth subplot: Correlation with Target Variable
correlation_with_target = data.corr()['FLAG'].sort_values(ascending=False)
sns.barplot(x=correlation_with_target.index, y=correlation_with_target.values, ax=axes[1, 1])
axes[1, 1].set_title('Correlation with FLAG')
axes[1, 1].set_xticklabels(correlation_with_target.index, rotation=45)
axes[1, 1].set_ylabel('Correlation Coefficient')

# Adjust layout
plt.tight_layout()
plt.show()