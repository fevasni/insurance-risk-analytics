"""# Insurance Risk Analytics: Exploratory Data Analysis (EDA)

This project focuses on analyzing historical insurance data to identify patterns, geographic trends, and financial risks. The goal is to provide data-driven insights into premium pricing and claim frequency.

## 🚀 Project Overview
This repository contains a comprehensive Exploratory Data Analysis (EDA) for an insurance portfolio over an 18-month period.

### Key Objectives:
* Identify geographic trends in premiums and claims across provinces.
* Analyze the relationship between `TotalPremium` and `TotalClaims`.
* Calculate and compare **Loss Ratios** across different demographics (Gender, Vehicle Type).
* Detect outliers and anomalies in financial reporting.

---

## 🛠️ Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/fevasni/insurance-risk-analytics.git](https://github.com/fevasni/insurance-risk-analytics.git)
   cd insurance-risk-analytics
   
   📊 EDA Workflow
The analysis is structured into the following phases:

1. Data Summarization & Quality
Descriptive Statistics: Summary for TotalPremium, TotalClaims, and CustomValueEstimate.

Data Typing: Reviewing dtypes to ensure dates and categorical features are correctly handled.

Handling Missing Values: Documentation of missing value checks and handling strategy.

2. Univariate & Multivariate Analysis
Distribution Check: Histograms for numerical columns and bar charts for categorical features.

Correlation: Scatter plots and correlation matrices analyzing the link between TotalPremium and TotalClaims as a function of ZipCode.

Geographic Trends: Comparisons of cover types, premium, and auto makes across various provinces.

3. Outlier Detection
Box plots are utilized on key financial features to identify skewness in the data.

💡 Guiding Questions Answered
The analysis provides detailed answers to the following:

What is the overall Loss Ratio? How does it vary by Province, VehicleType, and Gender?

What are the distributions of key financial variables? Are there outliers in TotalClaims or CustomValueEstimate that could skew analysis?

Are there temporal trends? Did claim frequency or severity change over the 18-month period?

Which vehicle makes/models are associated with the highest and lowest claim amounts?

📈 Visualizations
The project produces three creative and well-designed plots capturing key insights, including:

Geographic Risk Analysis: Visualizing insurance metrics across provinces.

Financial Correlation: Exploring relationships between premiums and claims.

Temporal Claims Severity: Tracking changes over time.

🧑‍💻 Development Workflow
Branching: Work is performed on the task-1 branch.

Commit Frequency: Work is committed at least three times a day with descriptive messages.

Submission: To merge changes:

Bash
git checkout main
git pull origin main
git merge task-1
git push origin main
"""

with open("README.md", "w", encoding="utf-8") as f:
f.write(readme_content)
