import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Loads preprocessed data
data = pd.read_csv('preprocessed_transactions.csv')
print("Loading data from preprocessed_transactions.csv")
print(f"Data loaded successfully. Data shape: {data.shape}")

# Assumes 'FLAG' is the column indicating fraud (0 or 1)
X = data.drop(['FLAG'], axis=1)
y = data['FLAG']

print("Features and target prepared.")

# Encodes categorical variables if necessary
X = pd.get_dummies(X)
print("Categorical variables encoded.")

# Splits data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# Trains a model with GridSearchCV for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best model
model = grid_search.best_estimator_
print("Best model found.")

# Predict and evaluate
y_pred = model.predict(X_test)
print("Model predictions completed.")

# Classification report
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Feature Importance
feature_importances = model.feature_importances_
indices = np.argsort(feature_importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(12, 8))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

print("Model training and evaluation completed.")