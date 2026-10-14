# Yuva Intern - Customer Churn Prediction

A machine learning project planned and implemented in Python as part of the **Yuva Intern Machine Learning Implementation Internship**.

## Project Overview

Customer churn prediction is a supervised machine learning problem in which historical customer information is used to estimate whether a customer is likely to leave a service.

The project uses the publicly available **IBM Telco Customer Churn** sample as the planned dataset. The workflow covers:

1. Problem definition and hypothesis formulation
2. Data collection and data-quality checks
3. Exploratory Data Analysis (EDA)
4. Data preprocessing and feature engineering
5. Baseline model development
6. Model comparison and evaluation
7. Risk assessment and future deployment planning

## Objectives

- Predict customer churn using customer demographics, services, contract information and billing attributes.
- Compare multiple classification algorithms.
- Evaluate the models using metrics beyond accuracy.
- Identify important factors associated with churn.
- Build a reproducible and well-documented ML workflow.

## Planned Models

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting (optional extension)

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Recall and F1-score are particularly important because a churn model should identify as many genuinely at-risk customers as reasonably possible while controlling false alerts.

## Repository Structure

```text
yuva-intern-ml-customer-churn/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── 01_customer_churn_eda_and_baseline.ipynb
│
├── src/
│   ├── train_model.py
│   └── README.md
│
└── reports/
    ├── Week_1_Project_Plan.docx
    └── README.md
```

## Dataset

The planned dataset is the IBM Telco Customer Churn sample.

Public references:

- IBM Customer Churn overview: https://www.ibm.com/think/topics/customer-churn
- IBM Telco Customer Churn documentation: https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=samples-telco-customer-churn

**Note:** The dataset itself is intentionally not committed to this repository. Place the CSV in `data/` after downloading it from an appropriate public source.

Expected filename:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## Run the Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/01_customer_churn_eda_and_baseline.ipynb
```

## Run the Training Script

After placing the dataset in `data/`:

```bash
python src/train_model.py
```

The script performs cleaning, preprocessing, train/test splitting, model training and evaluation.

## Week 1 Deliverable

The Week 1 deliverable is the project planning report:

`reports/Week_1_Project_Plan.docx`

It contains the problem statement, hypothesis, research background, methodology, timeline, evaluation criteria and risk assessment.

## Future Scope

- Hyperparameter optimization
- Probability calibration
- Cost-sensitive learning
- Model explainability
- Deployment as a REST API or Streamlit application
- Model monitoring and drift detection
- Validation on current organizational data
- A/B testing of retention strategies

## Author

**Mohit Yadav**

Yuva Intern - Machine Learning Implementation Internship
