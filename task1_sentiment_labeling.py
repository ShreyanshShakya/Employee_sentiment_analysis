# %% [markdown]
# # Task 1: Sentiment Labeling
# 
# **Objective**: Label each employee message as Positive, Negative, or Neutral.
# 
# **Approach**: Using TextBlob library for sentiment analysis based on polarity scores.
# 
# ## Methodology
# TextBlob provides polarity scores ranging from -1 (negative) to +1 (positive).
# - Polarity > 0.1: Positive
# - Polarity < -0.1: Negative  
# - Otherwise: Neutral

# %%
# Import required libraries
import pandas as pd
import numpy as np
from datetime import datetime
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")
print("Using TextBlob for sentiment analysis")

# %%
# Load the dataset
df = pd.read_csv('data/test.csv')
print(f"Dataset loaded: {len(df):,} records")
print(f"\nColumns: {df.columns.tolist()}")

# %% [markdown]
# ## Data Preprocessing
# 
# **Observation**: Before sentiment analysis, we need to clean and prepare the data.
# This includes handling missing values and combining subject with body for full context.

# %%
# Check for missing values
print("Missing Values Summary:")
print(df.isnull().sum())
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# %%
# Preprocess the data
# Rename 'from' column to 'employee' for clarity
df = df.rename(columns={'from': 'employee'})

# Handle missing values - fill with empty string for text columns
df['Subject'] = df['Subject'].fillna('')
df['body'] = df['body'].fillna('')

# Combine Subject and body for full message context
df['full_message'] = df['Subject'] + ' ' + df['body']

# Parse dates
df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['year_month'] = df['date'].dt.to_period('M')

# Clean employee names for readability
df['employee_name'] = df['employee'].str.split('@').str[0].str.replace('.', ' ', regex=False).str.title()

print("Preprocessing complete!")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Unique employees: {df['employee'].nunique()}")

# %% [markdown]
# ## Sentiment Analysis with TextBlob
# 
# **Approach**: TextBlob analyzes text and returns a polarity score between -1 and +1.
# 
# **Classification Thresholds**:
# - Polarity > 0.1 → Positive (+1)
# - Polarity < -0.1 → Negative (-1)
# - -0.1 ≤ Polarity ≤ 0.1 → Neutral (0)

# %%
def analyze_sentiment_textblob(text):
    """
    Analyze sentiment using TextBlob.
    
    Returns tuple of (sentiment_label, polarity_score, sentiment_score)
    """
    if not text or len(str(text).strip()) == 0:
        return 'Neutral', 0.0, 0
    
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        
        # Classify based on polarity threshold
        if polarity > 0.1:
            return 'Positive', polarity, 1
        elif polarity < -0.1:
            return 'Negative', polarity, -1
        else:
            return 'Neutral', polarity, 0
            
    except Exception as e:
        return 'Neutral', 0.0, 0

# %%
# Apply sentiment analysis to all messages
print("Analyzing sentiment for all messages using TextBlob...")
print("This process is fast with TextBlob!")

# Apply the function
results = df['full_message'].apply(analyze_sentiment_textblob)

# Extract results into separate columns
df['sentiment'] = results.apply(lambda x: x[0])
df['polarity_score'] = results.apply(lambda x: x[1])
df['sentiment_score'] = results.apply(lambda x: x[2])

print("\n✅ Sentiment analysis complete!")

# %% [markdown]
# ## Results Summary
# 
# **Observation**: Let's examine the distribution of sentiments across all messages.

# %%
# Display sentiment distribution
print("=" * 60)
print("SENTIMENT DISTRIBUTION")
print("=" * 60)
print(df['sentiment'].value_counts())
print(f"\nPercentages:")
print((df['sentiment'].value_counts(normalize=True) * 100).round(2))

# %%
# Display polarity statistics
print("\n" + "=" * 60)
print("POLARITY SCORE STATISTICS")
print("=" * 60)
print(df['polarity_score'].describe())

# %% [markdown]
# ## Key Observations
# 
# 1. **TextBlob Classification**: Messages are classified based on polarity thresholds
# 2. **Polarity Range**: Scores range from -1 (very negative) to +1 (very positive)
# 3. **Neutral Zone**: Messages with polarity between -0.1 and 0.1 are considered neutral

# %%
# Save labeled dataset
df.to_csv('data/test_labeled.csv', index=False)
print("\n✅ Labeled dataset saved to 'data/test_labeled.csv'")
print(f"\nColumns in saved dataset: {df.columns.tolist()}")

# %% [markdown]
# ## Summary
# 
# - **Method Used**: TextBlob sentiment analysis
# - **Classification**: Positive (polarity > 0.1), Negative (polarity < -0.1), Neutral (otherwise)
# - **Output**: Added `sentiment`, `polarity_score`, and `sentiment_score` columns
# - **Labeled dataset saved** for use in subsequent tasks (EDA, scoring, ranking, etc.)
