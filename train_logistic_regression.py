"""
ICC 2026 World Cup Prediction - Logistic Regression Model
Based on T20I data from 2010-2024
Made by: Chanitha Abeygunawardena
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("="*60)
print("ICC 2026 WORLD CUP PREDICTION - LOGISTIC REGRESSION MODEL")
print("="*60)

# Load training data
print("\n1. Loading training data...")
training_data = pd.read_csv('datasets/training_data.csv')
print(f"Training data shape: {training_data.shape}")
print(f"\nFeatures:\n{training_data.columns.tolist()}")

# Prepare features and target
print("\n2. Preparing features and target...")
feature_cols = [
    'team1_matches', 'team1_wins', 'team1_win_rate', 'team1_recent_form',
    'team2_matches', 'team2_wins', 'team2_win_rate', 'team2_recent_form',
    'h2h_total', 'h2h_team1_wins', 'h2h_win_rate'
]

X = training_data[feature_cols]
y = training_data['team1_wins_match']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Split data
print("\n3. Splitting data (80-20 split)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Feature scaling
print("\n4. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
print("\n5. Training Logistic Regression model...")
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs',
    C=1.0
)

lr_model.fit(X_train_scaled, y_train)
print("Model trained successfully")

# Make predictions
print("\n6. Making predictions...")
y_train_pred = lr_model.predict(X_train_scaled)
y_test_pred = lr_model.predict(X_test_scaled)

# Evaluate model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\n{'='*60}")
print("MODEL PERFORMANCE")
print(f"{'='*60}")
print(f"Training Accuracy: {train_accuracy:.2%}")
print(f"Test Accuracy: {test_accuracy:.2%}")

print(f"\n{'='*60}")
print("CLASSIFICATION REPORT (Test Set)")
print(f"{'='*60}")
print(classification_report(y_test, y_test_pred, 
                          target_names=['Team 2 Wins', 'Team 1 Wins']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Feature coefficients (importance)
print(f"\n{'='*60}")
print("FEATURE COEFFICIENTS")
print(f"{'='*60}")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': lr_model.coef_[0],
    'abs_coefficient': np.abs(lr_model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

for idx, row in feature_importance.iterrows():
    print(f"{row['feature']:25s}: {row['coefficient']:7.4f} (|{row['abs_coefficient']:.4f}|)")

# Visualizations
print("\n7. Creating visualizations...")

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Feature Coefficients
ax1 = plt.subplot(2, 2, 1)
sns.barplot(data=feature_importance, x='coefficient', y='feature', palette='coolwarm', ax=ax1)
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax1.set_title('Feature Coefficients - Logistic Regression', fontsize=14, fontweight='bold')
ax1.set_xlabel('Coefficient Value', fontsize=12)
ax1.set_ylabel('Feature', fontsize=12)

# 2. Confusion Matrix Heatmap
ax2 = plt.subplot(2, 2, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Team 2 Wins', 'Team 1 Wins'],
            yticklabels=['Team 2 Wins', 'Team 1 Wins'], ax=ax2)
ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax2.set_ylabel('Actual', fontsize=12)
ax2.set_xlabel('Predicted', fontsize=12)

# 3. Model Accuracy Comparison
ax3 = plt.subplot(2, 2, 3)
accuracies = pd.DataFrame({
    'Dataset': ['Training', 'Test'],
    'Accuracy': [train_accuracy, test_accuracy]
})
sns.barplot(data=accuracies, x='Dataset', y='Accuracy', palette='coolwarm', ax=ax3)
ax3.set_ylim([0, 1])
ax3.set_title('Model Accuracy: Training vs Test', fontsize=14, fontweight='bold')
ax3.set_ylabel('Accuracy', fontsize=12)
for i, v in enumerate(accuracies['Accuracy']):
    ax3.text(i, v + 0.02, f'{v:.2%}', ha='center', fontsize=12, fontweight='bold')

# 4. Prediction Distribution
ax4 = plt.subplot(2, 2, 4)
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred
})
pred_counts = predictions_df.groupby(['Actual', 'Predicted']).size().reset_index(name='Count')
pred_counts['Label'] = pred_counts.apply(
    lambda x: f"Actual: {'Team 1' if x['Actual'] == 1 else 'Team 2'}\nPred: {'Team 1' if x['Predicted'] == 1 else 'Team 2'}", 
    axis=1
)
sns.barplot(data=pred_counts, x='Label', y='Count', palette='Set2', ax=ax4)
ax4.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
ax4.set_xlabel('Outcome', fontsize=12)
ax4.set_ylabel('Count', fontsize=12)
ax4.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('visualizations/logistic_regression_analysis.png', dpi=300, bbox_inches='tight')
print("Saved visualization: visualizations/logistic_regression_analysis.png")

# Save model
import pickle
with open('models/logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
with open('models/scaler_lr.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Saved model: models/logistic_regression_model.pkl")
print("Saved scaler: models/scaler_lr.pkl")

print("\n" + "="*60)
print("LOGISTIC REGRESSION MODEL TRAINING COMPLETE")
print("="*60)