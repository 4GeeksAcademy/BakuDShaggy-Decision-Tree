from utils import db_connect
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Database connection (as per starter code)
engine = db_connect()

# 1. Load and clean data
url = "https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv"
diabetes = pd.read_csv(url)

# Handle impossible zero values
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_columns:
    median_val = diabetes[diabetes[col] != 0][col].median()
    diabetes[col] = diabetes[col].replace(0, median_val)

# 2. Prepare data
X = diabetes.drop('Outcome', axis=1)
y = diabetes['Outcome']

# Split data (80% train, 20% test) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# 3. Build and optimize decision tree
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2', None]
}

model = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
model.fit(X_train, y_train)

# Get best model
best_model = model.best_estimator_
best_params = model.best_params_

# 4. Evaluate model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# 5. Save model and cleaned data
joblib.dump(best_model, 'decision_tree_model.pkl')
diabetes.to_csv('diabetes_clean.csv', index=False)

# 6. Generate evaluation report (printed to console)
print("="*60)
print("Diabetes Prediction Model Report")
print("="*60)
print(f"\nBest Parameters: {best_params}")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(class_report)

# Feature importance
importances = best_model.feature_importances_
feature_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nFeature Importances:")
print(feature_imp)

# Confusion matrix visualization
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')  # Save for later use
print("\nConfusion matrix saved as 'confusion_matrix.png'")

# Feature importance plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp, palette='viridis')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved as 'feature_importance.png'")

print("\nModel and cleaned data saved successfully!")
print("Files created: decision_tree_model.pkl, diabetes_clean.csv")