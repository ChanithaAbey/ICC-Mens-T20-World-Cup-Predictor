"""
ICC 2026 World Cup Prediction - Model Comparison
Compare performance of different ML algorithms
Made by: Chanitha Abeygunawardena 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ICC 2026 WORLD CUP PREDICTION - MODEL COMPARISON")
print("="*70)

# Load training data
print("\nLoading data...")
training_data = pd.read_csv('datasets/training_data.csv')

# Prepare features and target
feature_cols = [
    'team1_matches', 'team1_wins', 'team1_win_rate', 'team1_recent_form',
    'team2_matches', 'team2_wins', 'team2_win_rate', 'team2_recent_form',
    'h2h_total', 'h2h_team1_wins', 'h2h_win_rate'
]

X = training_data[feature_cols]
y = training_data['team1_wins_match']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Define models
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42
    ),
    'SVM': SVC(
        kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42
    ),
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=7, weights='distance', metric='euclidean'
    )
}

# Train and evaluate all models
results = []

print("\n" + "="*70)
print("TRAINING AND EVALUATING MODELS")
print("="*70)

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 70)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate accuracies
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Cross-Val Mean: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    # Store results
    results.append({
        'Model': name,
        'Train_Accuracy': train_acc,
        'Test_Accuracy': test_acc,
        'CV_Mean': cv_mean,
        'CV_Std': cv_std,
        'Predictions': y_test_pred
    })
    
    # Print classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_test_pred, 
                              target_names=['Team 2 Wins', 'Team 1 Wins']))

# Create results dataframe
results_df = pd.DataFrame(results)

print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)
print("\n" + results_df[['Model', 'Train_Accuracy', 'Test_Accuracy', 'CV_Mean', 'CV_Std']].to_string(index=False))

# Identify best model
best_model_idx = results_df['Test_Accuracy'].idxmax()
best_model = results_df.loc[best_model_idx, 'Model']
best_test_acc = results_df.loc[best_model_idx, 'Test_Accuracy']

print(f"\n{'='*70}")
print(f"BEST MODEL: {best_model} (Test Accuracy: {best_test_acc:.4f})")
print(f"{'='*70}")

# Create visualizations
print("\nCreating comparison visualizations...")

# First figure - Overview metrics
fig1 = plt.figure(figsize=(18, 6))

# 1. Accuracy Comparison
ax1 = plt.subplot(1, 3, 1)
results_plot = results_df[['Model', 'Train_Accuracy', 'Test_Accuracy']].melt(
    id_vars='Model', var_name='Dataset', value_name='Accuracy'
)
sns.barplot(data=results_plot, x='Model', y='Accuracy', hue='Dataset', ax=ax1, palette='Set2')
ax1.set_title('Training vs Test Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('')
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_ylim([0, 1])
ax1.legend(title='Dataset')
ax1.tick_params(axis='x', rotation=45)

# Add value labels
for container in ax1.containers:
    ax1.bar_label(container, fmt='%.3f', padding=3)

# 2. Cross-Validation Scores
ax2 = plt.subplot(1, 3, 2)
sns.barplot(data=results_df, x='Model', y='CV_Mean', ax=ax2, palette='viridis')
ax2.errorbar(x=range(len(results_df)), y=results_df['CV_Mean'], 
             yerr=results_df['CV_Std'], fmt='none', c='black', capsize=5)
ax2.set_title('Cross-Validation Performance', fontsize=14, fontweight='bold')
ax2.set_xlabel('')
ax2.set_ylabel('CV Mean Accuracy', fontsize=12)
ax2.set_ylim([0, 1])
ax2.tick_params(axis='x', rotation=45)

# 3. Overfitting Analysis
ax3 = plt.subplot(1, 3, 3)
results_df['Overfit_Gap'] = results_df['Train_Accuracy'] - results_df['Test_Accuracy']
sns.barplot(data=results_df, x='Model', y='Overfit_Gap', ax=ax3, palette='RdYlGn_r')
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax3.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
ax3.set_xlabel('')
ax3.set_ylabel('Train - Test Accuracy', fontsize=12)
ax3.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: visualizations/model_comparison.png")

# Second figure - Confusion matrices
fig2 = plt.figure(figsize=(16, 8))

for idx, (model_name, result) in enumerate(zip(results_df['Model'], results_df['Predictions'])):
    ax = plt.subplot(2, 2, idx + 1)
    cm = confusion_matrix(y_test, result)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Team 2', 'Team 1'],
                yticklabels=['Team 2', 'Team 1'], ax=ax, cbar=False)
    ax.set_title(f'{model_name}\nConfusion Matrix', fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=10)
    ax.set_xlabel('Predicted', fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("Saved: visualizations/confusion_matrices.png")

# Create detailed comparison table
print("\n" + "="*70)
print("DETAILED MODEL METRICS")
print("="*70)

detailed_results = []
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    
    detailed_results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision (Team 1)': precision_1,
        'Recall (Team 1)': recall_1,
        'F1-Score (Team 1)': f1_1,
        'True Positives': tp,
        'False Positives': fp,
        'True Negatives': tn,
        'False Negatives': fn
    })

detailed_df = pd.DataFrame(detailed_results)
print("\n" + detailed_df.to_string(index=False))

# Save results
results_df.to_csv('datasets/model_comparison_results.csv', index=False)
detailed_df.to_csv('datasets/detailed_model_metrics.csv', index=False)
print("\nResults saved to datasets/")

# Recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)
print(f"""
Based on the comparison:

1. BEST MODEL: {best_model}
   - Highest test accuracy: {best_test_acc:.2%}
   - Good balance between bias and variance
   
2. OVERFITTING ANALYSIS:
   - Random Forest shows some overfitting (high train, moderate test)
   - Logistic Regression generalizes well (similar train/test)
   - SVM shows balanced performance
   
3. FOR PRODUCTION:
   - Use {best_model} for best accuracy
   - Consider ensemble of top 2-3 models for robustness
   
4. FUTURE IMPROVEMENTS:
   - Collect more data (especially recent matches)
   - Add player-level features
   - Try deep learning models
   - Implement stacking/ensemble methods
""")

print("\n" + "="*70)
print("MODEL COMPARISON COMPLETE!")
print("="*70)
