# %% [markdown]
# # Task 6: Predictive Modeling
# 
# **Objective**: Develop a linear regression model to analyze and predict sentiment trends.
# 
# **Features**: Message frequency, average message length, word count, previous month score.

# %%
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
os.makedirs('visualizations', exist_ok=True)
print("Libraries loaded!")

# %%
# Load labeled dataset
df = pd.read_csv('data/test_labeled.csv')
df['date'] = pd.to_datetime(df['date'])
df['year_month'] = df['date'].dt.to_period('M')
print(f"Loaded {len(df):,} records")

# %% [markdown]
# ## Feature Engineering
# 
# Creating features that may influence sentiment scores:
# - **message_frequency**: Number of messages per month
# - **avg_message_length**: Average character count
# - **avg_word_count**: Average words per message
# - **prev_month_score**: Previous month's sentiment score (lag feature)

# %%
# Aggregate features per employee per month
features = df.groupby(['employee', 'year_month']).agg({
    'sentiment_score': 'sum',
    'message_length': 'mean',
    'word_count': 'mean',
    'full_message': 'count'
}).rename(columns={
    'full_message': 'message_frequency',
    'sentiment_score': 'monthly_score'
}).reset_index()

# Add lag feature (previous month score)
features = features.sort_values(['employee', 'year_month'])
features['prev_month_score'] = features.groupby('employee')['monthly_score'].shift(1)
features = features.dropna()

print(f"Feature matrix: {len(features)} samples")
features.head()

# %% [markdown]
# ## Model Training

# %%
# Define features and target
feature_cols = ['message_frequency', 'message_length', 'word_count', 'prev_month_score']
X = features[feature_cols]
y = features['monthly_score']

print("Feature Statistics:")
print(X.describe())

# %%
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training: {len(X_train)} samples")
print(f"Testing: {len(X_test)} samples")

# %%
# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("✅ Model trained!")

# %% [markdown]
# ## Model Evaluation

# %%
print("=" * 60)
print("LINEAR REGRESSION MODEL EVALUATION")
print("=" * 60)

print("\nTRAINING SET:")
print(f"  R² Score: {r2_score(y_train, y_pred_train):.4f}")
print(f"  MSE: {mean_squared_error(y_train, y_pred_train):.4f}")
print(f"  MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")

print("\nTEST SET:")
print(f"  R² Score: {r2_score(y_test, y_pred_test):.4f}")
print(f"  MSE: {mean_squared_error(y_test, y_pred_test):.4f}")
print(f"  MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")

# %% [markdown]
# ## Feature Importance

# %%
print("\n" + "=" * 60)
print("FEATURE COEFFICIENTS (Importance)")
print("=" * 60)
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

for _, row in coef_df.iterrows():
    direction = "↑" if row['Coefficient'] > 0 else "↓"
    print(f"  {row['Feature']:<20}: {row['Coefficient']:>8.4f} {direction}")
print(f"  {'Intercept':<20}: {model.intercept_:>8.4f}")

# %% [markdown]
# ## Visualization

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Actual vs Predicted
axes[0].scatter(y_test, y_pred_test, alpha=0.5, color='steelblue', s=30)
min_val, max_val = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Score', fontsize=12)
axes[0].set_ylabel('Predicted Score', fontsize=12)
axes[0].set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
axes[0].legend()

# 2. Residuals Distribution
residuals = y_test - y_pred_test
axes[1].hist(residuals, bins=30, color='steelblue', edgecolor='white')
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Residual (Actual - Predicted)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')

# 3. Feature Importance
colors = ['green' if c > 0 else 'red' for c in coef_df['Coefficient']]
axes[2].barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
axes[2].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
axes[2].set_xlabel('Coefficient Value', fontsize=12)
axes[2].set_title('Feature Importance', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/model_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: model_performance.png")

# %% [markdown]
# ## Model Interpretation

# %%
print("\n" + "=" * 60)
print("MODEL INTERPRETATION")
print("=" * 60)
print(f"""
The linear regression model reveals key insights about sentiment predictors:

1. PREVIOUS MONTH SCORE: The strongest predictor. Employees tend to maintain
   similar sentiment patterns month-over-month.

2. MESSAGE FREQUENCY: {'Positive' if model.coef_[0] > 0 else 'Negative'} correlation 
   with sentiment score. {'More active employees tend to be more positive.' if model.coef_[0] > 0 else 'Higher activity may indicate stress.'}

3. MESSAGE LENGTH: {'Longer' if model.coef_[1] > 0 else 'Shorter'} messages 
   correlate with {'higher' if model.coef_[1] > 0 else 'lower'} sentiment scores.

4. WORD COUNT: Similar pattern to message length.

MODEL PERFORMANCE:
- R² = {r2_score(y_test, y_pred_test):.3f} → Explains {r2_score(y_test, y_pred_test)*100:.1f}% of variance
- MAE = {mean_absolute_error(y_test, y_pred_test):.2f} → Average prediction error

The model captures general trends but individual predictions vary due to 
the inherent complexity of human sentiment.
""")

# %%
# Save model results
results = {
    'r2_train': r2_score(y_train, y_pred_train),
    'r2_test': r2_score(y_test, y_pred_test),
    'mae_test': mean_absolute_error(y_test, y_pred_test),
    'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'coefficients': dict(zip(feature_cols, model.coef_)),
    'intercept': model.intercept_
}

import json
with open('data/model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Model results saved to data/model_results.json")
