# # Employee Sentiment Analysis
# 
# This script performs comprehensive sentiment analysis on employee email communications.
# 
# ## Table of Contents
# 1. Setup & Data Loading
# 2. Task 1: Sentiment Labeling
# 3. Task 2: Exploratory Data Analysis (EDA)
# 4. Task 3: Employee Score Calculation
# 5. Task 4: Employee Ranking
# 6. Task 5: Flight Risk Identification
# 7. Task 6: Predictive Modeling


# 1. Setup & Data Loading


# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("Libraries imported successfully!")


# Load the dataset
df = pd.read_csv('data/test.csv')
print(f"Dataset loaded: {len(df):,} records")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")


# Preview the data
df.head()


# ### Data Preprocessing


# Check for missing values
print("Missing Values Summary:")
print(df.isnull().sum())
print(f"\nTotal missing: {df.isnull().sum().sum()}")


# Clean and preprocess the data
# Rename 'from' column to 'employee' for clarity
df = df.rename(columns={'from': 'employee'})

# Handle missing values - fill with empty string for text columns
df['Subject'] = df['Subject'].fillna('')
df['body'] = df['body'].fillna('')

# Combine Subject and body for full message context
df['full_message'] = df['Subject'] + ' ' + df['body']

# Parse dates
df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')

# Extract additional date features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['year_month'] = df['date'].dt.to_period('M')

# Clean employee emails (extract name if needed)
df['employee_name'] = df['employee'].str.split('@').str[0].str.replace('.', ' ', regex=False).str.title()

print("Data preprocessing complete!")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Unique employees: {df['employee'].nunique()}")

# %% [markdown]
# ## 2. Task 1: Sentiment Labeling
# 
# **Objective**: Label each message as Positive, Negative, or Neutral using a transformer-based sentiment model.
# 
# **Approach**: Using HuggingFace Transformers with DistilBERT fine-tuned on SST-2 for sentiment classification.

# %%
# Import transformer libraries
from transformers import pipeline
import torch

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU (CUDA)' if device == 0 else 'CPU'}")

# %%
# Initialize sentiment analysis pipeline
print("Loading sentiment analysis model...")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
    truncation=True,
    max_length=512
)
print("Model loaded successfully!")

# %%
# Function to analyze sentiment with proper handling
def analyze_sentiment(text):
    """
    Analyze sentiment of text using transformer model.
    Returns: (sentiment_label, sentiment_score)
    """
    if not text or len(text.strip()) == 0:
        return 'Neutral', 0.5
    
    try:
        # Truncate very long texts
        text = text[:512] if len(text) > 512 else text
        result = sentiment_analyzer(text)[0]
        
        label = result['label']
        score = result['score']
        
        # Map to our labels
        if label == 'POSITIVE':
            # High confidence positive
            if score > 0.7:
                return 'Positive', score
            else:
                return 'Neutral', score
        else:  # NEGATIVE
            if score > 0.7:
                return 'Negative', score
            else:
                return 'Neutral', score
                
    except Exception as e:
        return 'Neutral', 0.5

# %%
# Apply sentiment analysis to all messages
print("Analyzing sentiment for all messages...")
print("This may take several minutes...")

from tqdm import tqdm
tqdm.pandas()

# Process in batches for efficiency
batch_size = 32
sentiments = []
scores = []

for i in tqdm(range(0, len(df), batch_size)):
    batch = df['full_message'].iloc[i:i+batch_size].tolist()
    for text in batch:
        sentiment, score = analyze_sentiment(text)
        sentiments.append(sentiment)
        scores.append(score)

df['sentiment'] = sentiments
df['confidence_score'] = scores

print("\nSentiment analysis complete!")

# %%
# Map sentiment to numerical score for calculations
sentiment_score_map = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
df['sentiment_score'] = df['sentiment'].map(sentiment_score_map)

# Display sentiment distribution
print("\nSentiment Distribution:")
print(df['sentiment'].value_counts())
print(f"\nPercentages:")
print(df['sentiment'].value_counts(normalize=True).round(4) * 100)

# %%
# Save labeled dataset
df.to_csv('data/test_labeled.csv', index=False)
print("Labeled dataset saved to 'data/test_labeled.csv'")

# %% [markdown]
# ## 3. Task 2: Exploratory Data Analysis (EDA)
# 
# **Objective**: Understand the structure, distribution, and trends in the dataset.

# %%
# Create visualizations directory reference
import os
viz_path = 'visualizations'
os.makedirs(viz_path, exist_ok=True)

# %% [markdown]
# ### 2.1 Data Overview

# %%
# Data overview summary
print("=" * 60)
print("DATA OVERVIEW")
print("=" * 60)
print(f"Total Records: {len(df):,}")
print(f"Total Columns: {len(df.columns)}")
print(f"Unique Employees: {df['employee'].nunique()}")
print(f"Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"\nColumn Info:")
for col in df.columns:
    print(f"  - {col}: {df[col].dtype}")

# %% [markdown]
# ### 2.2 Sentiment Distribution

# %%
# Sentiment distribution visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
colors = {'Positive': '#2ecc71', 'Neutral': '#f1c40f', 'Negative': '#e74c3c'}
sentiment_counts = df['sentiment'].value_counts()
axes[0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=[colors[s] for s in sentiment_counts.index], explode=[0.02]*3,
            shadow=True, startangle=90)
axes[0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')

# Bar chart
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=axes[1],
            palette=[colors[s] for s in sentiment_counts.index])
axes[1].set_xlabel('Sentiment', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Sentiment Counts', fontsize=14, fontweight='bold')
for i, v in enumerate(sentiment_counts.values):
    axes[1].text(i, v + 100, f'{v:,}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig(f'{viz_path}/sentiment_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: sentiment_distribution.png")

# %% [markdown]
# ### 2.3 Sentiment Trends Over Time

# %%
# Monthly sentiment trends
monthly_sentiment = df.groupby(['year_month', 'sentiment']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(14, 6))
monthly_sentiment.plot(kind='bar', stacked=True, ax=ax, 
                       color=[colors.get(c, '#3498db') for c in monthly_sentiment.columns],
                       width=0.8)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Number of Messages', fontsize=12)
ax.set_title('Monthly Sentiment Trends', fontsize=14, fontweight='bold')
ax.legend(title='Sentiment', loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{viz_path}/sentiment_trends_monthly.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: sentiment_trends_monthly.png")

# %%
# Sentiment ratio over time
monthly_total = df.groupby('year_month').size()
monthly_positive = df[df['sentiment'] == 'Positive'].groupby('year_month').size()
monthly_negative = df[df['sentiment'] == 'Negative'].groupby('year_month').size()

positive_ratio = (monthly_positive / monthly_total * 100).fillna(0)
negative_ratio = (monthly_negative / monthly_total * 100).fillna(0)

fig, ax = plt.subplots(figsize=(14, 5))
x = range(len(positive_ratio))
ax.plot(x, positive_ratio.values, 'g-o', label='Positive %', linewidth=2, markersize=6)
ax.plot(x, negative_ratio.values, 'r-s', label='Negative %', linewidth=2, markersize=6)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Percentage', fontsize=12)
ax.set_title('Sentiment Ratio Over Time', fontsize=14, fontweight='bold')
ax.legend()
ax.set_xticks(x)
ax.set_xticklabels([str(p) for p in positive_ratio.index], rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{viz_path}/sentiment_ratio_trends.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: sentiment_ratio_trends.png")

# %% [markdown]
# ### 2.4 Employee Activity Analysis

# %%
# Top 10 most active employees
top_senders = df['employee_name'].value_counts().head(10)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(top_senders.index[::-1], top_senders.values[::-1], color=sns.color_palette("viridis", 10))
ax.set_xlabel('Number of Messages', fontsize=12)
ax.set_ylabel('Employee', fontsize=12)
ax.set_title('Top 10 Most Active Employees', fontsize=14, fontweight='bold')
for i, v in enumerate(top_senders.values[::-1]):
    ax.text(v + 10, i, str(v), va='center')
plt.tight_layout()
plt.savefig(f'{viz_path}/top_active_employees.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: top_active_employees.png")

# %% [markdown]
# ### 2.5 Message Length Analysis

# %%
# Add message length features
df['message_length'] = df['full_message'].str.len()
df['word_count'] = df['full_message'].str.split().str.len()

# Message length by sentiment
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot of message length by sentiment
df.boxplot(column='message_length', by='sentiment', ax=axes[0])
axes[0].set_title('Message Length by Sentiment', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Sentiment')
axes[0].set_ylabel('Character Count')
plt.suptitle('')

# Word count distribution
for sent in ['Positive', 'Negative', 'Neutral']:
    subset = df[df['sentiment'] == sent]['word_count']
    axes[1].hist(subset, bins=50, alpha=0.5, label=sent, color=colors[sent])
axes[1].set_xlabel('Word Count')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Word Count Distribution by Sentiment', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].set_xlim(0, 500)

plt.tight_layout()
plt.savefig(f'{viz_path}/message_length_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: message_length_analysis.png")

# %% [markdown]
# ### 2.6 Key EDA Observations

# %%
print("=" * 60)
print("KEY EDA OBSERVATIONS")
print("=" * 60)
print(f"""
1. DATASET SIZE: {len(df):,} employee messages analyzed

2. SENTIMENT DISTRIBUTION:
   - Positive: {(df['sentiment'] == 'Positive').sum():,} ({(df['sentiment'] == 'Positive').mean()*100:.1f}%)
   - Neutral: {(df['sentiment'] == 'Neutral').sum():,} ({(df['sentiment'] == 'Neutral').mean()*100:.1f}%)
   - Negative: {(df['sentiment'] == 'Negative').sum():,} ({(df['sentiment'] == 'Negative').mean()*100:.1f}%)

3. EMPLOYEE ACTIVITY:
   - {df['employee'].nunique()} unique employees
   - Top sender: {top_senders.index[0]} ({top_senders.values[0]} messages)

4. MESSAGE CHARACTERISTICS:
   - Average length: {df['message_length'].mean():.0f} characters
   - Average word count: {df['word_count'].mean():.0f} words
""")

# %% [markdown]
# ## 4. Task 3: Employee Score Calculation
# 
# **Objective**: Compute monthly sentiment scores for each employee.
# 
# **Scoring**:
# - Positive: +1
# - Negative: -1
# - Neutral: 0

# %%
# Calculate monthly scores per employee
monthly_scores = df.groupby(['employee', 'year_month']).agg({
    'sentiment_score': 'sum',
    'sentiment': 'count'
}).rename(columns={'sentiment': 'message_count'})

monthly_scores = monthly_scores.reset_index()
monthly_scores.columns = ['employee', 'year_month', 'monthly_score', 'message_count']

print("Monthly Employee Scores Sample:")
monthly_scores.head(10)

# %%
# Summary statistics of monthly scores
print("\nMonthly Score Statistics:")
print(monthly_scores['monthly_score'].describe())

# Distribution of monthly scores
fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(monthly_scores['monthly_score'], bins=50, color='steelblue', edgecolor='white')
ax.axvline(x=0, color='red', linestyle='--', label='Zero (Neutral)')
ax.set_xlabel('Monthly Sentiment Score', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Monthly Employee Sentiment Scores', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{viz_path}/monthly_score_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: monthly_score_distribution.png")

# %% [markdown]
# ## 5. Task 4: Employee Ranking
# 
# **Objective**: Identify top 3 positive and negative employees per month.
# 
# **Sorting**: First by score (descending), then alphabetically.

# %%
def get_top_employees_by_month(monthly_scores_df, n=3):
    """Get top positive and negative employees for each month."""
    results = {'positive': [], 'negative': []}
    
    for month in monthly_scores_df['year_month'].unique():
        month_data = monthly_scores_df[monthly_scores_df['year_month'] == month].copy()
        
        # Sort by score descending, then by employee name alphabetically
        month_data_sorted = month_data.sort_values(
            ['monthly_score', 'employee'], 
            ascending=[False, True]
        )
        
        # Top positive (highest scores)
        top_positive = month_data_sorted.head(n)
        for _, row in top_positive.iterrows():
            results['positive'].append({
                'month': str(month),
                'employee': row['employee'],
                'score': row['monthly_score'],
                'messages': row['message_count']
            })
        
        # Top negative (lowest scores)
        month_data_sorted_neg = month_data.sort_values(
            ['monthly_score', 'employee'], 
            ascending=[True, True]
        )
        top_negative = month_data_sorted_neg.head(n)
        for _, row in top_negative.iterrows():
            results['negative'].append({
                'month': str(month),
                'employee': row['employee'],
                'score': row['monthly_score'],
                'messages': row['message_count']
            })
    
    return results

rankings = get_top_employees_by_month(monthly_scores)

# %%
# Display top positive employees
print("=" * 60)
print("TOP 3 POSITIVE EMPLOYEES BY MONTH")
print("=" * 60)
top_positive_df = pd.DataFrame(rankings['positive'])
print(top_positive_df.to_string(index=False))

# %%
# Display top negative employees  
print("\n" + "=" * 60)
print("TOP 3 NEGATIVE EMPLOYEES BY MONTH")
print("=" * 60)
top_negative_df = pd.DataFrame(rankings['negative'])
print(top_negative_df.to_string(index=False))

# %%
# Visualize employee rankings for latest months
latest_months = sorted(monthly_scores['year_month'].unique())[-6:]  # Last 6 months
latest_data = monthly_scores[monthly_scores['year_month'].isin(latest_months)]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, month in enumerate(latest_months):
    month_data = latest_data[latest_data['year_month'] == month].sort_values('monthly_score', ascending=False).head(10)
    colors_bar = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in month_data['monthly_score']]
    axes[idx].barh(range(len(month_data)), month_data['monthly_score'], color=colors_bar)
    axes[idx].set_yticks(range(len(month_data)))
    axes[idx].set_yticklabels([e.split('@')[0][:15] for e in month_data['employee']])
    axes[idx].set_title(f'{month}', fontsize=12, fontweight='bold')
    axes[idx].axvline(x=0, color='black', linestyle='-', linewidth=0.5)

plt.suptitle('Top Employee Scores by Month (Recent 6 Months)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{viz_path}/employee_rankings.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: employee_rankings.png")

# %%
# Save rankings to CSV
top_positive_df.to_csv('data/top_positive_employees.csv', index=False)
top_negative_df.to_csv('data/top_negative_employees.csv', index=False)
print("Rankings saved to CSV files.")

# %% [markdown]
# ## 6. Task 5: Flight Risk Identification
# 
# **Objective**: Identify employees who sent 4+ negative emails within any 30-day rolling window.
# 
# **Implementation**: True rolling 30-day window (not calendar month-based).

# %%
def identify_flight_risks(df, window_days=30, min_negative=4):
    """
    Identify employees at flight risk based on 30-day rolling negative email count.
    
    An employee is flagged if they sent 4+ negative emails within any 30-day period.
    """
    flight_risk_employees = set()
    flight_risk_details = []
    
    # Filter to negative messages only
    negative_df = df[df['sentiment'] == 'Negative'].copy()
    negative_df = negative_df.sort_values(['employee', 'date'])
    
    for employee in negative_df['employee'].unique():
        emp_negative = negative_df[negative_df['employee'] == employee].copy()
        
        if len(emp_negative) < min_negative:
            continue
        
        emp_negative = emp_negative.sort_values('date')
        dates = emp_negative['date'].tolist()
        
        # Check each date as potential end of 30-day window
        for i, end_date in enumerate(dates):
            start_date = end_date - timedelta(days=window_days)
            
            # Count negative emails in this window
            count = sum(1 for d in dates if start_date <= d <= end_date)
            
            if count >= min_negative:
                flight_risk_employees.add(employee)
                flight_risk_details.append({
                    'employee': employee,
                    'window_end': end_date,
                    'negative_count_in_window': count
                })
                break  # Found a qualifying window, move to next employee
    
    return list(flight_risk_employees), flight_risk_details

# %%
# Identify flight risks
flight_risks, flight_risk_details = identify_flight_risks(df)

print("=" * 60)
print("FLIGHT RISK EMPLOYEES")
print("=" * 60)
print(f"\nTotal employees flagged as flight risk: {len(flight_risks)}")
print("\nFlight Risk Employee List:")
for i, emp in enumerate(sorted(flight_risks), 1):
    print(f"  {i}. {emp}")

# %%
# Create flight risk details DataFrame
flight_risk_df = pd.DataFrame(flight_risk_details)
if len(flight_risk_df) > 0:
    print("\nFlight Risk Details (sample):")
    print(flight_risk_df.head(20))
    
    # Save flight risk list
    flight_risk_df.to_csv('data/flight_risk_employees.csv', index=False)
    print("\nFlight risk details saved to 'data/flight_risk_employees.csv'")

# %%
# Visualize flight risk analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Flight risk count
labels = ['Flight Risk', 'Normal']
sizes = [len(flight_risks), df['employee'].nunique() - len(flight_risks)]
colors_pie = ['#e74c3c', '#2ecc71']
axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie, 
            explode=[0.05, 0], shadow=True, startangle=90)
axes[0].set_title('Flight Risk Employee Distribution', fontsize=14, fontweight='bold')

# Top flight risks by negative count
if len(flight_risk_df) > 0:
    top_risks = flight_risk_df.groupby('employee')['negative_count_in_window'].max().sort_values(ascending=False).head(10)
    axes[1].barh(range(len(top_risks)), top_risks.values, color='#e74c3c')
    axes[1].set_yticks(range(len(top_risks)))
    axes[1].set_yticklabels([e.split('@')[0][:20] for e in top_risks.index])
    axes[1].set_xlabel('Max Negative Emails in 30-Day Window')
    axes[1].set_title('Top Flight Risk Employees', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{viz_path}/flight_risk_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: flight_risk_analysis.png")

# %% [markdown]
# ## 7. Task 6: Predictive Modeling
# 
# **Objective**: Develop a linear regression model to analyze sentiment trends.
# 
# **Features**:
# - Message frequency per month
# - Average message length
# - Average word count

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# %%
# Prepare features for modeling
# Aggregate features per employee per month
model_features = df.groupby(['employee', 'year_month']).agg({
    'sentiment_score': 'sum',           # Target: monthly sentiment score
    'message_length': 'mean',           # Feature: avg message length
    'word_count': 'mean',               # Feature: avg word count
    'full_message': 'count'             # Feature: message frequency
}).rename(columns={'full_message': 'message_frequency', 
                   'sentiment_score': 'monthly_score'}).reset_index()

# Add additional features
# Previous month score (lag feature)
model_features = model_features.sort_values(['employee', 'year_month'])
model_features['prev_month_score'] = model_features.groupby('employee')['monthly_score'].shift(1)
model_features = model_features.dropna()

print("Feature Matrix Sample:")
model_features.head(10)

# %%
# Define features and target
feature_cols = ['message_frequency', 'message_length', 'word_count', 'prev_month_score']
X = model_features[feature_cols]
y = model_features['monthly_score']

print(f"Total samples: {len(X)}")
print(f"Features: {feature_cols}")

# %%
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# %%
# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# %%
# Model evaluation
print("=" * 60)
print("LINEAR REGRESSION MODEL EVALUATION")
print("=" * 60)

print("\nTraining Set Metrics:")
print(f"  R² Score: {r2_score(y_train, y_pred_train):.4f}")
print(f"  MSE: {mean_squared_error(y_train, y_pred_train):.4f}")
print(f"  MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")

print("\nTest Set Metrics:")
print(f"  R² Score: {r2_score(y_test, y_pred_test):.4f}")
print(f"  MSE: {mean_squared_error(y_test, y_pred_test):.4f}")
print(f"  MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")

# %%
# Feature importance (coefficients)
print("\nFeature Coefficients:")
for feat, coef in zip(feature_cols, model.coef_):
    print(f"  {feat}: {coef:.4f}")
print(f"  Intercept: {model.intercept_:.4f}")

# %%
# Visualization of model results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Actual vs Predicted
axes[0].scatter(y_test, y_pred_test, alpha=0.5, color='steelblue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Score')
axes[0].set_ylabel('Predicted Score')
axes[0].set_title('Actual vs Predicted Sentiment Scores', fontsize=12, fontweight='bold')

# Residuals
residuals = y_test - y_pred_test
axes[1].hist(residuals, bins=30, color='steelblue', edgecolor='white')
axes[1].axvline(x=0, color='red', linestyle='--')
axes[1].set_xlabel('Residual')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')

# Feature importance
coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': model.coef_})
coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=True)
colors_coef = ['green' if c > 0 else 'red' for c in coef_df['Coefficient']]
axes[2].barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_coef)
axes[2].set_xlabel('Coefficient')
axes[2].set_title('Feature Importance', fontsize=12, fontweight='bold')
axes[2].axvline(x=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig(f'{viz_path}/model_performance.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: model_performance.png")

# %% [markdown]
# ### Model Interpretation
# 
# The linear regression model reveals:
# 1. **Previous month score** has the strongest positive correlation with current month score
# 2. **Message frequency** and **message length** contribute to predicting sentiment trends
# 3. The model captures general sentiment patterns but individual predictions may vary

# %% [markdown]
# ## 8. Summary and Key Findings

# %%
# Generate summary
print("=" * 70)
print("EMPLOYEE SENTIMENT ANALYSIS - FINAL SUMMARY")
print("=" * 70)

# Get overall top employees
overall_positive = monthly_scores.groupby('employee')['monthly_score'].sum().sort_values(ascending=False)
overall_negative = monthly_scores.groupby('employee')['monthly_score'].sum().sort_values(ascending=True)

print(f"""
DATASET OVERVIEW:
-----------------
• Total Messages Analyzed: {len(df):,}
• Unique Employees: {df['employee'].nunique()}
• Date Range: {df['date'].min().date()} to {df['date'].max().date()}

SENTIMENT DISTRIBUTION:
-----------------------
• Positive: {(df['sentiment'] == 'Positive').sum():,} ({(df['sentiment'] == 'Positive').mean()*100:.1f}%)
• Neutral: {(df['sentiment'] == 'Neutral').sum():,} ({(df['sentiment'] == 'Neutral').mean()*100:.1f}%)
• Negative: {(df['sentiment'] == 'Negative').sum():,} ({(df['sentiment'] == 'Negative').mean()*100:.1f}%)

TOP 3 MOST POSITIVE EMPLOYEES (Overall):
----------------------------------------
1. {overall_positive.index[0]} (Score: {overall_positive.values[0]})
2. {overall_positive.index[1]} (Score: {overall_positive.values[1]})
3. {overall_positive.index[2]} (Score: {overall_positive.values[2]})

TOP 3 MOST NEGATIVE EMPLOYEES (Overall):
----------------------------------------
1. {overall_negative.index[0]} (Score: {overall_negative.values[0]})
2. {overall_negative.index[1]} (Score: {overall_negative.values[1]})
3. {overall_negative.index[2]} (Score: {overall_negative.values[2]})

FLIGHT RISK EMPLOYEES:
----------------------
• Total Flagged: {len(flight_risks)}
• Criteria: 4+ negative emails within any 30-day period

MODEL PERFORMANCE:
------------------
• R² Score (Test): {r2_score(y_test, y_pred_test):.4f}
• MAE (Test): {mean_absolute_error(y_test, y_pred_test):.4f}
""")

# %%
# Save summary data for README
summary_data = {
    'total_messages': len(df),
    'unique_employees': df['employee'].nunique(),
    'positive_count': (df['sentiment'] == 'Positive').sum(),
    'neutral_count': (df['sentiment'] == 'Neutral').sum(),
    'negative_count': (df['sentiment'] == 'Negative').sum(),
    'top_positive': overall_positive.head(3).to_dict(),
    'top_negative': overall_negative.head(3).to_dict(),
    'flight_risks': flight_risks,
    'model_r2': r2_score(y_test, y_pred_test)
}

import json
with open('data/summary_data.json', 'w') as f:
    json.dump(summary_data, f, indent=2, default=str)

print("\nSummary data saved to 'data/summary_data.json'")
print("\n✅ Analysis Complete!")
