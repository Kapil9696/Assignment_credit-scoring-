
# Credit Scoring Assignment

This repository contains the solution for the credit scoring assignment, which aims to predict customer defaulting behavior using the `credit.csv` dataset.

## Project Overview

The primary objectives of this project are:

*   Perform Exploratory Data Analysis (EDA) to understand the data's characteristics.
*   Develop and train a predictive model to classify customers based on their likelihood of defaulting.
*   Evaluate the model's performance using appropriate metrics.
*   Estimate the potential business impact of deploying the model in a real-world setting.

## Data

The dataset used in this project is `credit.csv`, located in the  directory. This dataset contains information about customers applying for loans, including demographic data, loan details, credit history, and their actual defaulting behavior (the target variable).
## Notebooks

The core analysis and modeling process is implemented in the `notebooks/credit_scoring.ipynb` Jupyter Notebook. This notebook covers the following steps:

1.  **Data Loading and Preprocessing:** Loading the `credit.csv` data and handling any missing values or data transformations.
2.  **Exploratory Data Analysis (EDA):** Performing various EDA techniques to gain insights into the data, including descriptive statistics, visualizations, and correlation analysis.
3.  **Feature Engineering (If applicable):** Creating new features from existing ones to potentially improve model performance.
4.  **Model Training:** Training a chosen machine learning model (e.g., Logistic Regression, Random Forest) to predict defaulting behavior.
5.  **Model Evaluation:** Evaluating the trained model's performance using metrics such as accuracy, precision, recall, specificity, and AUC.
6.  **Feature Importance Analysis:** Determining the relative importance of each feature in the model's predictions.
7.  **Business Impact Estimation:** Estimating the potential financial impact of deploying the model.

## Key Findings and Results

*   **Exploratory Data Analysis (EDA):**
    *   Strong negative correlation observed between `credit_history` and `default`.
    *   Positive correlation between `amount` and `months_loan_duration`.
    *   (Add other key EDA findings here)

*   **Model Performance:**
    *   Accuracy: 84.00%
    *   Sensitivity/Recall: 87%
    *   Specificity: 81.42%
    *   Precision: 81.04%
    *   AUC: 0.94

*   **Feature Importance:**

| Feature                     | Importance |
|------------------------------|------------|
| `checking_balance`           | 11.98      |
| `amount`                     | 10.83      |
| `age`                        | 10.35      |
| `residence_history`          | 9.83       |
| `months_loan_duration`       | 8.31       |
| `purpose`                    | 7.28       |
| `installment_rate`           | 6.91       |
| `credit_history`             | 6.76       |
| `employment_length_category` | 6.37       |
| `property`                   | 5.21       |
| `job`                        | 3.10       |
| `gender`                     | 3.00       |
| `savings_balance_category`   | 2.73       |
| `existing_credits`           | 2.67       |
| `housing`                    | 2.40       |
| `personal_status`            | 1.20       |
| `dependents`                 | 1.06       |

*   **Business Impact Estimation:


## Instructions for Running the Notebook

1.  Clone the repository: `git clone https://github.com/Kapil9696/Assignment_credit-scoring-.git`
2.  Navigate to the project directory: `cd Assignment_credit-scoring-`
3.  Create a virtual environment (recommended):
    *   Linux/macOS: `python3 -m venv venv` and `source venv/bin/activate`
    *   Windows: `python -m venv venv` and `venv\Scripts\activate`
5.  Run the Jupyter Notebook: `jupyter notebook notebooks/credit_scoring.ipynb`

