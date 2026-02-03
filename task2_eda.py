# %% [markdown]
# # Task 2: Exploratory Data Analysis (EDA)
# 
# **Objective**: Understand the structure, distribution, and trends in the dataset.
# 
# **Prerequisites**: Run Task 1 first to generate the labeled dataset.

# %%
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
os.makedirs('visualizations', exist_ok=True)

print("Libraries loaded!")

# %%
# Load labeled dataset
df = pd.read_csv('data/test_labeled.csv')
df['date'] = pd.to_datetime(df['date'])
df['year_month'] = df['date'].dt.to_period('M')
print(f"Loaded {len(df):,} records")

# %% [markdown]
# ## 2.1 Data Overview

# %%
print("=" * 60)
print("DATA OVERVIEW")
print("=" * 60)
print(f"Total Records: {len(df):,}")
print(f"Unique Employees: {df['employee'].nunique()}")
print(f"Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")

# %% [markdown]
# ## 2.2 Sentiment Distribution

# %%
colors = {'Positive': '#2ecc71', 'Neutral': '#f1c40f', 'Negative': '#e74c3c'}
sentiment_counts = df['sentiment'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
axes[0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=[colors[s] for s in sentiment_counts.index], explode=[0.02]*3,
            shadow=True, startangle=90)
axes[0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')

# Bar chart
bars = axes[1].bar(sentiment_counts.index, sentiment_counts.values,
                   color=[colors[s] for s in sentiment_counts.index])
axes[1].set_xlabel('Sentiment', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Sentiment Counts', fontsize=14, fontweight='bold')
for bar, val in zip(bars, sentiment_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 100, f'{val:,}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('visualizations/sentiment_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: sentiment_distribution.png")

# %% [markdown]
# ## 2.3 Sentiment Trends Over Time

# %%
monthly_sentiment = df.groupby(['year_month', 'sentiment']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(14, 6))
monthly_sentiment.plot(kind='bar', stacked=True, ax=ax, 
                       color=[colors.get(c, '#3498db') for c in monthly_sentiment.columns], width=0.8)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Number of Messages', fontsize=12)
ax.set_title('Monthly Sentiment Trends', fontsize=14, fontweight='bold')
ax.legend(title='Sentiment')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('visualizations/sentiment_trends_monthly.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: sentiment_trends_monthly.png")

# %%
# Sentiment ratio trends
monthly_total = df.groupby('year_month').size()
monthly_pos = df[df['sentiment'] == 'Positive'].groupby('year_month').size()
monthly_neg = df[df['sentiment'] == 'Negative'].groupby('year_month').size()

pos_ratio = (monthly_pos / monthly_total * 100).fillna(0)
neg_ratio = (monthly_neg / monthly_total * 100).fillna(0)

fig, ax = plt.subplots(figsize=(14, 5))
x = range(len(pos_ratio))
ax.plot(x, pos_ratio.values, 'g-o', label='Positive %', linewidth=2)
ax.plot(x, neg_ratio.values, 'r-s', label='Negative %', linewidth=2)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Percentage', fontsize=12)
ax.set_title('Sentiment Ratio Over Time', fontsize=14, fontweight='bold')
ax.legend()
ax.set_xticks(x[::3])
ax.set_xticklabels([str(p) for p in list(pos_ratio.index)[::3]], rotation=45, ha='right')
plt.tight_layout()
plt.savefig('visualizations/sentiment_ratio_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: sentiment_ratio_trends.png")

# %% [markdown]
# ## 2.4 Employee Activity Analysis

# %%
top_senders = df['employee_name'].value_counts().head(10)

fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(top_senders.index[::-1], top_senders.values[::-1], color=sns.color_palette("viridis", 10))
ax.set_xlabel('Number of Messages', fontsize=12)
ax.set_title('Top 10 Most Active Employees', fontsize=14, fontweight='bold')
for i, v in enumerate(top_senders.values[::-1]):
    ax.text(v + 10, i, str(v), va='center')
plt.tight_layout()
plt.savefig('visualizations/top_active_employees.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: top_active_employees.png")

# %% [markdown]
# ## 2.5 Message Length Analysis

# %%
df['message_length'] = df['full_message'].str.len()
df['word_count'] = df['full_message'].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df.boxplot(column='message_length', by='sentiment', ax=axes[0])
axes[0].set_title('Message Length by Sentiment', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Sentiment')
axes[0].set_ylabel('Character Count')
plt.suptitle('')

for sent in ['Positive', 'Negative', 'Neutral']:
    subset = df[df['sentiment'] == sent]['word_count']
    axes[1].hist(subset, bins=50, alpha=0.5, label=sent, color=colors[sent])
axes[1].set_xlabel('Word Count')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Word Count Distribution', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].set_xlim(0, 500)

plt.tight_layout()
plt.savefig('visualizations/message_length_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: message_length_analysis.png")

# Save updated dataset with length features
df.to_csv('data/test_labeled.csv', index=False)

# %% [markdown]
# ## Key EDA Observations

# %%
print("=" * 60)
print("KEY EDA OBSERVATIONS")
print("=" * 60)
print(f"""
1. DATASET: {len(df):,} employee messages

2. SENTIMENT BREAKDOWN:
   - Positive: {(df['sentiment'] == 'Positive').sum():,} ({(df['sentiment'] == 'Positive').mean()*100:.1f}%)
   - Neutral: {(df['sentiment'] == 'Neutral').sum():,} ({(df['sentiment'] == 'Neutral').mean()*100:.1f}%)
   - Negative: {(df['sentiment'] == 'Negative').sum():,} ({(df['sentiment'] == 'Negative').mean()*100:.1f}%)

3. EMPLOYEE ACTIVITY:
   - {df['employee'].nunique()} unique employees
   - Top sender: {top_senders.index[0]} ({top_senders.values[0]} messages)

4. MESSAGE CHARACTERISTICS:
   - Avg length: {df['message_length'].mean():.0f} chars
   - Avg words: {df['word_count'].mean():.0f}
""")
