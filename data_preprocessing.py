import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load your data
data = pd.read_csv('transaction_dataset.csv')

# Fill NaN values as needed
data.ffill(inplace=True)

# Select relevant features for correlation analysis
selected_features = [
    'total Ether sent',
    'total ether received',
    'total transactions (including tnx to create contract',
    'total ether balance',
    'Sent tnx',
    'Received Tnx',
    'FLAG'  # Assuming you want to analyze against the target variable
]

# Filter the data to include only the selected features
correlation_data = data[selected_features]

# Calculate the correlation matrix
correlation_matrix = correlation_data.corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Set up the matplotlib figure with smaller size
plt.figure(figsize=(10, 8))

# Draw the heatmap with the mask and correct aspect ratio
heatmap = sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                       square=True, linewidths=.5, cbar_kws={"shrink": .8},
                       annot_kws={"size": 10},
                       xticklabels=correlation_matrix.columns.tolist(),
                       yticklabels=correlation_matrix.columns.tolist())

# Title and axis labels
plt.title('Correlation Matrix of Selected Transaction Features', fontsize=16)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# Add color bar label
cbar = heatmap.collections[0].colorbar
cbar.set_label('Correlation Coefficient', fontsize=12)

plt.tight_layout()  # Adjust layout to prevent clipping of labels

# Optionally save the plot
plt.savefig('correlation_matrix.png', dpi=300)
plt.show()