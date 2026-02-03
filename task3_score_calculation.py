# %% [markdown]
# # Task 3: Employee Score Calculation
# 
# **Objective**: Compute monthly sentiment scores for each employee.
# 
# **Scoring**: Positive = +1, Negative = -1, Neutral = 0

# %%
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
# ## Score Calculation Methodology
# 
# For each employee and each month:
# - Sum of sentiment scores (Positive=+1, Negative=-1, Neutral=0)
# - Scores reset at the beginning of each month

# %%
# Calculate monthly scores per employee
monthly_scores = df.groupby(['employee', 'year_month']).agg({
    'sentiment_score': 'sum',
    'sentiment': 'count'
}).rename(columns={'sentiment': 'message_count'})

monthly_scores = monthly_scores.reset_index()
monthly_scores.columns = ['employee', 'year_month', 'monthly_score', 'message_count']

# Add employee name
monthly_scores['employee_name'] = monthly_scores['employee'].str.split('@').str[0].str.replace('.', ' ', regex=False).str.title()

print(f"Total employee-month records: {len(monthly_scores):,}")
monthly_scores.head(10)

# %% [markdown]
# ## Score Statistics

# %%
print("Monthly Score Statistics:")
print(monthly_scores['monthly_score'].describe())

print(f"\nScore Range: {monthly_scores['monthly_score'].min()} to {monthly_scores['monthly_score'].max()}")

# %%
# Distribution of monthly scores
fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(monthly_scores['monthly_score'], bins=50, color='steelblue', edgecolor='white')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero (Neutral)')
ax.axvline(x=monthly_scores['monthly_score'].mean(), color='green', linestyle='--', 
           linewidth=2, label=f'Mean ({monthly_scores["monthly_score"].mean():.2f})')
ax.set_xlabel('Monthly Sentiment Score', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Monthly Employee Sentiment Scores', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('visualizations/monthly_score_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: monthly_score_distribution.png")

# %% [markdown]
# ## Employee Score Summary

# %%
# Overall scores per employee
overall_scores = monthly_scores.groupby('employee').agg({
    'monthly_score': ['sum', 'mean', 'count'],
    'message_count': 'sum'
}).reset_index()
overall_scores.columns = ['employee', 'total_score', 'avg_monthly_score', 'months_active', 'total_messages']

overall_scores['employee_name'] = overall_scores['employee'].str.split('@').str[0].str.replace('.', ' ', regex=False).str.title()
overall_scores = overall_scores.sort_values('total_score', ascending=False)

print("Top 10 Employees by Overall Score:")
print(overall_scores.head(10)[['employee_name', 'total_score', 'avg_monthly_score', 'total_messages']].to_string(index=False))

# %%
# Save monthly scores
monthly_scores.to_csv('data/monthly_employee_scores.csv', index=False)
overall_scores.to_csv('data/overall_employee_scores.csv', index=False)
print("\n✅ Scores saved to:")
print("   - data/monthly_employee_scores.csv")
print("   - data/overall_employee_scores.csv")

# %% [markdown]
# ## Visualization: Score Heatmap

# %%
# Pivot for heatmap (top 15 employees by activity)
top_employees = df['employee'].value_counts().head(15).index
pivot_data = monthly_scores[monthly_scores['employee'].isin(top_employees)].pivot_table(
    index='employee', columns='year_month', values='monthly_score', fill_value=0
)
pivot_data.index = [e.split('@')[0][:15] for e in pivot_data.index]

fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(pivot_data, cmap='RdYlGn', center=0, annot=False, ax=ax, cbar_kws={'label': 'Score'})
ax.set_title('Monthly Sentiment Scores Heatmap (Top 15 Employees)', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Employee')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('visualizations/score_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: score_heatmap.png")
