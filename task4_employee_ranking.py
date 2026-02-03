# %% [markdown]
# # Task 4: Employee Ranking
# 
# **Objective**: Identify top 3 positive and top 3 negative employees per month.
# 
# **Sorting**: First by score (descending), then alphabetically.

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
# Load monthly scores
monthly_scores = pd.read_csv('data/monthly_employee_scores.csv')
print(f"Loaded {len(monthly_scores):,} employee-month records")

# %% [markdown]
# ## Ranking Methodology
# 
# For each month:
# 1. Sort employees by score (descending)
# 2. For ties, sort alphabetically by email
# 3. Extract top 3 positive (highest scores)
# 4. Extract top 3 negative (lowest scores)

# %%
def get_rankings(df, n=3):
    """Get top positive and negative employees per month."""
    positive_rankings = []
    negative_rankings = []
    
    for month in df['year_month'].unique():
        month_data = df[df['year_month'] == month].copy()
        
        # Sort: score descending, then alphabetically
        sorted_desc = month_data.sort_values(['monthly_score', 'employee'], ascending=[False, True])
        sorted_asc = month_data.sort_values(['monthly_score', 'employee'], ascending=[True, True])
        
        # Top positive
        for rank, (_, row) in enumerate(sorted_desc.head(n).iterrows(), 1):
            positive_rankings.append({
                'month': month, 'rank': rank,
                'employee': row['employee'],
                'score': row['monthly_score'],
                'messages': row['message_count']
            })
        
        # Top negative
        for rank, (_, row) in enumerate(sorted_asc.head(n).iterrows(), 1):
            negative_rankings.append({
                'month': month, 'rank': rank,
                'employee': row['employee'],
                'score': row['monthly_score'],
                'messages': row['message_count']
            })
    
    return pd.DataFrame(positive_rankings), pd.DataFrame(negative_rankings)

top_positive, top_negative = get_rankings(monthly_scores)

# %% [markdown]
# ## Top 3 Positive Employees by Month

# %%
print("=" * 70)
print("TOP 3 POSITIVE EMPLOYEES BY MONTH")
print("=" * 70)
display_pos = top_positive.copy()
display_pos['employee'] = display_pos['employee'].str.split('@').str[0]
print(display_pos.to_string(index=False))

# %% [markdown]
# ## Top 3 Negative Employees by Month

# %%
print("=" * 70)
print("TOP 3 NEGATIVE EMPLOYEES BY MONTH")
print("=" * 70)
display_neg = top_negative.copy()
display_neg['employee'] = display_neg['employee'].str.split('@').str[0]
print(display_neg.to_string(index=False))

# %% [markdown]
# ## Visualization: Rankings

# %%
# Recent 6 months visualization
latest_months = sorted(monthly_scores['year_month'].unique())[-6:]
latest_data = monthly_scores[monthly_scores['year_month'].isin(latest_months)]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, month in enumerate(latest_months):
    month_data = latest_data[latest_data['year_month'] == month].sort_values('monthly_score', ascending=False).head(10)
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in month_data['monthly_score']]
    
    axes[idx].barh(range(len(month_data)), month_data['monthly_score'], color=colors)
    axes[idx].set_yticks(range(len(month_data)))
    axes[idx].set_yticklabels([e.split('@')[0][:12] for e in month_data['employee']])
    axes[idx].set_title(f'{month}', fontsize=12, fontweight='bold')
    axes[idx].axvline(x=0, color='black', linestyle='-', linewidth=0.5)

plt.suptitle('Employee Rankings by Month (Recent 6 Months)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/employee_rankings.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: employee_rankings.png")

# %% [markdown]
# ## Overall Rankings (Across All Months)

# %%
# Calculate overall rankings
overall = monthly_scores.groupby('employee').agg({'monthly_score': 'sum', 'message_count': 'sum'}).reset_index()
overall = overall.sort_values(['monthly_score', 'employee'], ascending=[False, True])

print("\n" + "=" * 60)
print("OVERALL TOP 3 POSITIVE EMPLOYEES")
print("=" * 60)
for i, row in overall.head(3).iterrows():
    print(f"{overall.head(3).index.get_loc(i)+1}. {row['employee']} (Score: {row['monthly_score']})")

overall_neg = overall.sort_values(['monthly_score', 'employee'], ascending=[True, True])
print("\n" + "=" * 60)
print("OVERALL TOP 3 NEGATIVE EMPLOYEES")
print("=" * 60)
for i, row in overall_neg.head(3).iterrows():
    print(f"{overall_neg.head(3).index.get_loc(i)+1}. {row['employee']} (Score: {row['monthly_score']})")

# %%
# Save rankings
top_positive.to_csv('data/top_positive_employees.csv', index=False)
top_negative.to_csv('data/top_negative_employees.csv', index=False)
print("\n✅ Rankings saved to:")
print("   - data/top_positive_employees.csv")
print("   - data/top_negative_employees.csv")
