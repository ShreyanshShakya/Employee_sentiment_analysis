# Employee Sentiment Analysis

A comprehensive NLP project analyzing employee email communications to assess sentiment, calculate engagement scores, identify flight risks, and predict sentiment trends.

---

## 📋 Project Overview

This project analyzes **2,191 employee email messages** from **10 employees** spanning **January 2010 to December 2011** to:

1. **Sentiment Labeling** - Classify messages as Positive, Negative, or Neutral using **TextBlob**
2. **Exploratory Data Analysis (EDA)** - Visualize sentiment distributions, trends, and patterns
3. **Employee Scoring** - Calculate monthly sentiment scores per employee
4. **Employee Ranking** - Identify top positive/negative employees each month
5. **Flight Risk Detection** - Flag employees with 4+ negative emails in 30-day windows
6. **Predictive Modeling** - Build linear regression model to predict sentiment

---

## 🗂️ Project Structure

```
employee_sentiment_analysis/
├── data/
│   ├── test.csv                      # Original dataset (2,191 records)
│   ├── test_labeled.csv              # Dataset with sentiment labels
│   ├── monthly_employee_scores.csv   # Monthly scores per employee
│   ├── overall_employee_scores.csv   # Overall employee scores
│   ├── top_positive_employees.csv    # Top positive rankings by month
│   ├── top_negative_employees.csv    # Top negative rankings by month
│   ├── flight_risk_employees.csv     # Flight risk employee list
│   ├── flight_risk_details.csv       # Detailed flight risk data
│   └── model_results.json            # ML model metrics
│
├── visualizations/                   # 10 PNG visualization files
│   ├── sentiment_distribution.png
│   ├── sentiment_trends_monthly.png
│   ├── sentiment_ratio_trends.png
│   ├── top_active_employees.png
│   ├── message_length_analysis.png
│   ├── monthly_score_distribution.png
│   ├── score_heatmap.png
│   ├── employee_rankings.png
│   ├── flight_risk_analysis.png
│   └── model_performance.png
│
├── task1_sentiment_labeling.ipynb    # Task 1: Sentiment classification
├── task2_eda.ipynb                   # Task 2: Exploratory data analysis
├── task3_score_calculation.ipynb     # Task 3: Monthly score calculation
├── task4_employee_ranking.ipynb      # Task 4: Employee rankings
├── task5_flight_risk.ipynb           # Task 5: Flight risk identification
├── task6_predictive_modeling.ipynb   # Task 6: Linear regression model
│
├── Final_Report.docx                 # Comprehensive project report
├── generate_report.py                # Script to regenerate DOCX report
└── README.md                         # This file
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn textblob python-docx
```

### Run Order
Execute notebooks sequentially as each depends on previous outputs:
```
Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6
```

---

## 📊 Key Findings

### Sentiment Distribution (TextBlob Analysis)

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| **Positive** | 975 | 44.5% |
| **Neutral** | 1,020 | 46.6% |
| **Negative** | 196 | 8.9% |

### Top 3 Most Positive Employees (Highest Overall Score)

| Rank | Employee | Total Score |
|------|----------|-------------|
| 1 | lydia.delgado@enron.com | 101 |
| 2 | john.arnold@enron.com | 97 |
| 3 | sally.beck@enron.com | 89 |

### Top 3 Most Negative Employees (Lowest Overall Score)

| Rank | Employee                | Total Score |
|------|-------------------------|-------------|
| 1    | rhonda.denton@enron.com | 50          |
| 2    | kayne.coulter@enron.com | 61          |
| 3    | don.baughman@enron.com  | 72          |

### 🚨 Flight Risk Employees (6 Identified)

Employees who sent 4+ negative emails within any 30-day rolling window:

1. bobette.riner@ipgdirect.com
2. johnny.palmer@enron.com
3. lydia.delgado@enron.com
4. patti.thompson@enron.com
5. rhonda.denton@enron.com
6. sally.beck@enron.com

---

## 🔍 Methodology

### Task 1: Sentiment Labeling (TextBlob)

**Library Used**: TextBlob (lexicon-based sentiment analysis)

**Classification Criteria**:
- Polarity > 0.1 → **Positive** (+1)
- Polarity < -0.1 → **Negative** (-1)
- -0.1 ≤ Polarity ≤ 0.1 → **Neutral** (0)

### Task 2: Exploratory Data Analysis

- Dataset overview and missing value analysis
- Sentiment distribution visualization (pie/bar charts)
- Monthly sentiment trends analysis
- Top 10 most active employees identification
- Message length and word count analysis by sentiment

### Task 3: Employee Score Calculation

**Scoring Method**:
- Each **Positive** message: +1 point
- Each **Negative** message: -1 point
- Each **Neutral** message: 0 points

Scores aggregated monthly per employee.

### Task 4: Employee Ranking

- Top 3 positive employees identified per month
- Top 3 negative employees identified per month
- Sorted by score (descending), then alphabetically for ties

### Task 5: Flight Risk Identification

**Criteria**: 4+ negative emails within any **30-day rolling window**

**Result**: 60% of employees (6/10) flagged as potential flight risks

### Task 6: Predictive Modeling

**Algorithm**: Linear Regression (scikit-learn)

**Features**:
- Message frequency (count per month)
- Average message length (characters)
- Average word count
- Previous month's sentiment score

**Model Performance**:

| Metric   | Value |
|----------|-------|
| R² Score | 0.41 |
| MAE      | 1.56 |
| RMSE     | 2.17 |

**Key Finding**: Message frequency is the strongest predictor—more active employees tend to have higher sentiment scores.

---

## 📈 Visualizations

| Visualization | Description |
|---------------|-------------|
| `sentiment_distribution.png` | Pie and bar chart showing sentiment breakdown |
| `sentiment_trends_monthly.png` | Stacked bar chart of monthly sentiment counts |
| `sentiment_ratio_trends.png` | Line chart of positive/negative percentage over time |
| `top_active_employees.png` | Horizontal bar chart of top 10 most active employees |
| `message_length_analysis.png` | Box plot and histograms of message length by sentiment |
| `monthly_score_distribution.png` | Histogram of employee monthly scores |
| `score_heatmap.png` | Heatmap of monthly scores by employee |
| `employee_rankings.png` | Monthly ranking visualization for top employees |
| `flight_risk_analysis.png` | Flight risk distribution and comparison charts |
| `model_performance.png` | Actual vs predicted, residuals, and feature importance |

---

## 💡 Key Observations

1. **Balanced Sentiment**: 44.5% positive and 46.6% neutral messages indicate professional workplace communication

2. **Low Negativity**: Only 8.9% of messages classified as negative, suggesting generally constructive email culture

3. **Flight Risk Detection**: Despite low negativity, 60% of employees showed clustered negative patterns in 30-day windows

4. **Predictive Insight**: Message frequency strongly predicts sentiment (R²=0.41)—more active employees tend to be more positive

5. **Most Active Employee**: Lydia Delgado with 284 messages over the period

---

## 📝 Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- textblob

