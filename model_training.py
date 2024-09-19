import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load preprocessed data
data = pd.read_csv('preprocessed_transactions.csv')

# Assume 'fraudulent' is the column indicating fraud (0 or 1)
X = data.drop(['fraudulent'], axis=1)
y = data['fraudulent']

# Encode categorical variables if necessary
X = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
