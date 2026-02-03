# %% [markdown]
# # Task 5: Flight Risk Identification
# 
# **Objective**: Identify employees at risk of leaving based on negative sentiment patterns.
# 
# **Criteria**: 4+ negative emails within any **30-day rolling window** (irrespective of calendar months).

# %%
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
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
print(f"Loaded {len(df):,} records")
print(f"Negative messages: {(df['sentiment'] == 'Negative').sum():,}")

# %% [markdown]
# ## Flight Risk Detection Algorithm
# 
# For each employee:
# 1. Get all negative emails sorted by date
# 2. For each negative email, count negatives in preceding 30 days
# 3. If count ≥ 4, flag employee as flight risk

# %%
def identify_flight_risks(df, window_days=30, min_negative=4):
    """Identify employees with 4+ negative emails in any 30-day window."""
    flight_risks = set()
    flight_details = []
    
    negative_df = df[df['sentiment'] == 'Negative'].copy()
    negative_df = negative_df.sort_values(['employee', 'date'])
    
    for employee in negative_df['employee'].unique():
        emp_neg = negative_df[negative_df['employee'] == employee].sort_values('date')
        
        if len(emp_neg) < min_negative:
            continue
        
        dates = emp_neg['date'].tolist()
        
        for end_date in dates:
            start_date = end_date - timedelta(days=window_days)
            count = sum(1 for d in dates if start_date <= d <= end_date)
            
            if count >= min_negative:
                flight_risks.add(employee)
                flight_details.append({
                    'employee': employee,
                    'window_end': end_date,
                    'negative_count': count
                })
                break
    
    return list(flight_risks), flight_details

# %%
# Run detection
flight_risks, flight_details = identify_flight_risks(df)

print("=" * 60)
print("FLIGHT RISK IDENTIFICATION RESULTS")
print("=" * 60)
print(f"\nTotal employees analyzed: {df['employee'].nunique()}")
print(f"Employees flagged as flight risk: {len(flight_risks)}")
print(f"Percentage at risk: {len(flight_risks)/df['employee'].nunique()*100:.1f}%")

# %% [markdown]
# ## Flight Risk Employee List

# %%
print("\nFLIGHT RISK EMPLOYEES:")
print("-" * 40)
for i, emp in enumerate(sorted(flight_risks), 1):
    print(f"{i:3}. {emp}")

# %%
# Create details DataFrame
flight_df = pd.DataFrame(flight_details)
if len(flight_df) > 0:
    flight_df['employee_name'] = flight_df['employee'].str.split('@').str[0].str.replace('.', ' ', regex=False).str.title()
    print("\nFlight Risk Details:")
    print(flight_df[['employee_name', 'window_end', 'negative_count']].head(20).to_string(index=False))

# %% [markdown]
# ## Visualization

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart: Flight Risk vs Normal
labels = ['Flight Risk', 'Normal']
sizes = [len(flight_risks), df['employee'].nunique() - len(flight_risks)]
colors = ['#e74c3c', '#2ecc71']
axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, 
            explode=[0.05, 0], shadow=True, startangle=90)
axes[0].set_title('Flight Risk Distribution', fontsize=14, fontweight='bold')

# Top risks by negative count
if len(flight_df) > 0:
    top_risks = flight_df.groupby('employee')['negative_count'].max().sort_values(ascending=False).head(10)
    axes[1].barh(range(len(top_risks)), top_risks.values, color='#e74c3c')
    axes[1].set_yticks(range(len(top_risks)))
    axes[1].set_yticklabels([e.split('@')[0][:18] for e in top_risks.index])
    axes[1].set_xlabel('Negative Emails in 30-Day Window')
    axes[1].set_title('Top 10 Highest Risk Employees', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/flight_risk_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: flight_risk_analysis.png")

# %% [markdown]
# ## Additional Risk Analysis

# %%
# Analyze negative email patterns for flight risks
if len(flight_risks) > 0:
    risk_df = df[df['employee'].isin(flight_risks)]
    normal_df = df[~df['employee'].isin(flight_risks)]
    
    print("\nComparison: Flight Risk vs Normal Employees")
    print("-" * 50)
    print(f"{'Metric':<30} {'Flight Risk':>12} {'Normal':>12}")
    print("-" * 50)
    print(f"{'Avg messages per employee':<30} {len(risk_df)/len(flight_risks):>12.1f} {len(normal_df)/(df['employee'].nunique()-len(flight_risks)):>12.1f}")
    print(f"{'% Negative messages':<30} {(risk_df['sentiment']=='Negative').mean()*100:>11.1f}% {(normal_df['sentiment']=='Negative').mean()*100:>11.1f}%")
    print(f"{'% Positive messages':<30} {(risk_df['sentiment']=='Positive').mean()*100:>11.1f}% {(normal_df['sentiment']=='Positive').mean()*100:>11.1f}%")

# %%
# Save flight risk data
flight_risk_list = pd.DataFrame({'employee': sorted(flight_risks)})
flight_risk_list.to_csv('data/flight_risk_employees.csv', index=False)

if len(flight_df) > 0:
    flight_df.to_csv('data/flight_risk_details.csv', index=False)

print("\n✅ Flight risk data saved to:")
print("   - data/flight_risk_employees.csv")
print("   - data/flight_risk_details.csv")
