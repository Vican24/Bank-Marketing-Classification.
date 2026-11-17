# Bank Marketing Subscription Prediction

## Project Overview

This project uses Machine Learning to predict whether a bank customer will subscribe to a term deposit based on their demographic information, financial background, and previous marketing campaign interactions.

The primary objective is to help banks identify potential customers who are more likely to subscribe, allowing marketing teams to make better-informed decisions and improve campaign effectiveness.

I implemented **Logistic Regression** as the classification algorithm, evaluated its baseline performance, and applied hyperparameter tuning using `GridSearchCV` to improve its ability to identify potential subscribers.

## Business Problem

Bank marketing campaigns can be expensive and time-consuming. Contacting customers who are unlikely to subscribe may result in wasted marketing resources.

The business problem is therefore:

**Can we build a classification model that identifies customers who are more likely to subscribe to a bank term deposit?**

Since identifying potential subscribers is the primary objective, particular attention was given to the model's recall score for the positive class (`yes`).

## Dataset

The project uses the Bank Marketing dataset.

The target variable is:

- `y`: Whether the customer subscribed to a term deposit (`yes` or `no`).

The dataset contains customer information and marketing campaign features.

## Tools and Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Joblib

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

- Examined the dataset structure and dimensions.
- Investigated numerical and categorical features.
- Checked for missing values and duplicate records.
- Examined the distribution of the target variable.
- Identified class imbalance.

### 2. Data Preprocessing

- Separated features from the target variable.
- Split the dataset into training and testing sets.
- Applied One-Hot Encoding to categorical features.
- Prepared the numerical and categorical features for model training.

### 3. Baseline Model

Implemented a baseline Logistic Regression model to establish initial performance.

**Baseline Evaluation:**

| Metric | Score |
|---|---:|
| Accuracy | 89.33% |
| Precision | 64.12% |
| Recall | 18.38% |
| F1-score | 28.57% |
| ROC-AUC | 0.7750 |

Although the baseline model achieved 89.33% accuracy, its recall for the positive class was only 18.38%.

This meant it identified only a small proportion of customers who actually subscribed.

### 4. Hyperparameter Tuning

Used `GridSearchCV` with stratified 5-fold cross-validation to search for better model parameters.

The parameters explored included:

- `C`
- `solver`
- `class_weight`

**Best Parameters:**

```python
{
    'C': 0.01,
    'class_weight': 'balanced',
    'solver': 'liblinear'
}
```

Using `class_weight='balanced'` helped the model pay more attention to the minority class.

### 5. Tuned Model Evaluation

The tuned model achieved the following results:

| Metric | Score |
|---|---:|
| Accuracy | 74.13% |
| Precision | 26.05% |
| Recall | 66.76% |
| F1-score | 37.48% |
| ROC-AUC | 0.7772 |

### 6. Model Comparison

| Metric | Baseline Model | Tuned Model |
|---|---:|---:|
| Accuracy | 89.33% | 74.13% |
| Precision | 64.12% | 26.05% |
| Recall | 18.38% | 66.76% |
| F1-score | 28.57% | 37.48% |
| ROC-AUC | 0.7750 | 0.7772 |

The tuned model significantly improved recall, increasing it from 18.38% to 66.76%.

It also identified 701 actual subscribers compared with 193 identified by the baseline model.

However, this improvement came with a reduction in precision and accuracy, alongside an increase in false positives.

## Key Findings

- The dataset was imbalanced, making accuracy alone an insufficient evaluation metric.
- The baseline model achieved high accuracy but performed poorly at identifying actual subscribers.
- Hyperparameter tuning with `GridSearchCV` improved the model's recall considerably.
- The tuned model identified more potential subscribers but also produced more false positives.
- ROC-AUC improved slightly from 0.7750 to 0.7772.
- The choice of evaluation metric should depend on the business objective.

## Conclusion

The tuned Logistic Regression model was selected as the preferred model for this project because the primary objective was to identify as many potential term-deposit subscribers as possible.

With a recall score of 66.76%, the tuned model identified a considerably larger proportion of actual subscribers than the baseline model.

However, its precision of 26.05% indicates that many customers predicted as potential subscribers did not actually subscribe.

In a real-world banking environment, the bank would need to consider its marketing budget and the cost of contacting customers who are unlikely to subscribe before deciding on the final classification threshold.

## Challenges and Lessons Learned

Throughout this project, I developed a better understanding of:

- Handling imbalanced classification datasets.
- Data preprocessing and One-Hot Encoding.
- Training and evaluating Logistic Regression models.
- Understanding confusion matrices.
- Precision, recall, F1-score, and ROC-AUC.
- Hyperparameter tuning using `GridSearchCV`.
- Understanding the trade-off between precision and recall.
- Interpreting Logistic Regression coefficients.

## Future Improvements

- Experiment with other classification algorithms such as Random Forest and XGBoost.
- Explore additional classification thresholds.
- Investigate feature selection and feature engineering.
- Evaluate the model using Precision-Recall curves.
- Develop a complete prediction pipeline for new customer data.

## Author

**AI & Machine Learning Enthusiast**

This project is part of my journey toward becoming an AI/ML Engineer.
