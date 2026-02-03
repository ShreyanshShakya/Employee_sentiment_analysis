# Employee Sentiment Analysis

A comprehensive NLP project analyzing employee email communications to assess sentiment and engagement patterns.

## 📋 Project Overview

This project analyzes 2,191 employee messages from 10 employees to:
- Automatically classify sentiment (Positive/Negative/Neutral) using **TextBlob**
- Track employee engagement through monthly sentiment scores
- Identify flight-risk employees based on negative communication patterns
- Predict sentiment trends using linear regression

## 🗂️ Project Structure

```
employee_sentiment_analysis/
├── data/
│   ├── test.csv                      # Original dataset
│   ├── test_labeled.csv              # Dataset with sentiment labels
│   ├── monthly_employee_scores.csv   # Monthly sentiment scores
│   ├── top_positive_employees.csv    # Top positive rankings
│   ├── top_negative_employees.csv    # Top negative rankings
│   ├── flight_risk_employees.csv     # Flight risk list
│   └── model_results.json            # ML model metrics
├── visualizations/                   # All charts and graphs (10 files)
├── task1_sentiment_labeling.ipynb    # Task 1: Sentiment classification
├── task2_eda.ipynb                   # Task 2: Exploratory analysis
├── task3_score_calculation.ipynb     # Task 3: Monthly scores
├── task4_employee_ranking.ipynb      # Task 4: Employee rankings
├── task5_flight_risk.ipynb           # Task 5: Flight risk detection
├── task6_predictive_modeling.ipynb   # Task 6: ML modeling
├── Final_Report.docx                 # Comprehensive report
└── README.md                         # This file
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn textblob
```

### Run Order
Execute notebooks in sequence (Task 1 → Task 6), as each builds on previous outputs.

---

## 📊 Key Findings

### Sentiment Distribution (Using TextBlob)
| Sentiment | Count | Percentage |
|-----------|-------|------------|
| Positive  | 976   | 44.5%      |
| Neutral   | 1,021 | 46.6%      |
| Negative  | 196   | 8.9%       |

### Top 3 Most Positive Employees (Highest Overall Score)
| Rank | Employee | Total Score |
|------|----------|-------------|
| 1 | **lydia.delgado@enron.com** | 101         |
| 2 | john.arnold@enron.com | 97          |
| 3 | sally.beck@enron.com  | 89          |

### Top 3 Most Negative Employees (Highest Overall Score)
| Rank | Employee | Total Score |
|------|----------|-------------|
| 1 | **rhonda.denton@enron.com** | 50          |
| 2 | kayne.coulter@enron.com  | 61          |
| 3 | don.baughman@enron.com | 72          |

### 🚨 Flight Risk Employees (6 identified)
Employees who sent 4+ negative emails within any 30-day period:

1. **bobette.riner@ipgdirect.com**
2. **johnny.palmer@enron.com**
3. **lydia.delgado@enron.com**
4. **patti.thompson@enron.com**
5. **rhonda.denton@enron.com**
6. **sally.beck@enron.com**

---

## 🔍 Methodology

### Sentiment Labeling (Task 1) - TextBlob
- **Library**: TextBlob (rule-based sentiment analysis)
- **Classification**:
  - Polarity > 0.1 → Positive (+1)
  - Polarity < -0.1 → Negative (-1)
  - Otherwise → Neutral (0)

### Employee Scoring (Task 3)
- **Positive message**: +1 point
- **Negative message**: -1 point
- **Neutral message**: 0 points
- Scores aggregated monthly per employee

### Flight Risk Criteria (Task 5)
- **Window**: Rolling 30-day period (not calendar month)
- **Threshold**: 4 or more negative emails within any 30-day window

### Predictive Model (Task 6)
- **Algorithm**: Linear Regression (sklearn)
- **Features**: Message frequency, average length, word count, previous month score
- **R² Score**: 0.41 (explains 41% of variance)
- **MAE**: 1.56 points average prediction error

---

## 📈 Generated Visualizations

| File | Description |
|------|-------------|
| `sentiment_distribution.png` | Pie/bar chart of sentiment breakdown |
| `sentiment_trends_monthly.png` | Stacked bar chart of monthly trends |
| `sentiment_ratio_trends.png` | Line chart of positive/negative % over time |
| `top_active_employees.png` | Top 10 most active employees bar chart |
| `message_length_analysis.png` | Box plot and histogram by sentiment |
| `monthly_score_distribution.png` | Histogram of employee monthly scores |
| `score_heatmap.png` | Heatmap of monthly scores by employee |
| `employee_rankings.png` | Monthly rankings visualization |
| `flight_risk_analysis.png` | Flight risk distribution charts |
| `model_performance.png` | Actual vs predicted, residuals, feature importance |

---

## 💡 Key Insights

1. **Balanced Sentiment**: With TextBlob, 44.5% positive and 46.6% neutral messages indicate generally professional communication
2. **Flight Risk Detection**: 60% of employees (6 out of 10) showed patterns of clustered negative communications
3. **Predictive Power**: Model R²=0.41 shows message frequency is the strongest predictor of sentiment
4. **Active = Positive**: Message frequency has positive correlation with sentiment scores

---

## 📝 Requirements

- Python 3.8+
- TextBlob for sentiment analysis
- scikit-learn for predictive modeling
- Pandas, NumPy, Matplotlib, Seaborn for data processing


