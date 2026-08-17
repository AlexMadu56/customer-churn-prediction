# Customer Churn Prediction with Machine Learning

## Overview

Customer churn is a major challenge for telecommunications companies, as retaining existing customers is often more cost-effective than acquiring new ones.

This project develops an end-to-end machine learning pipeline to predict customer churn using demographic, service, contract, tenure and billing information. The workflow covers data quality assessment, preprocessing, model comparison, cross-validation, hyperparameter optimization, threshold optimization and model interpretability using SHAP.

The main objective is not only to build a predictive model, but also to provide an interpretable framework that can support data-driven customer retention strategies.

---

## Objectives

The project aims to:

- Explore and understand the customer churn dataset.
- Assess data quality and identify potential preprocessing issues.
- Develop several binary classification models.
- Compare model performance using appropriate evaluation metrics.
- Apply stratified cross-validation to obtain robust performance estimates.
- Optimize the best-performing model through hyperparameter search.
- Evaluate the final model on an untouched test set.
- Optimize the classification threshold according to predictive performance.
- Interpret model predictions using SHAP.
- Translate model findings into potential business implications.

---

## Dataset

The project uses the **Telco Customer Churn** dataset from Kaggle:

**Source:** `blastchar/telco-customer-churn`

The dataset contains information about telecommunications customers, including:

- Demographic characteristics
- Customer tenure
- Phone and internet services
- Contract information
- Payment methods
- Monthly charges
- Total charges

The target variable is `Churn`:

- `0` → No churn
- `1` → Churn

The dataset contains **7,043 observations**.

The customer identifier (`customerID`) is removed before modelling because it is an identifier rather than a predictive characteristic.

---

## Methodology

### 1. Data Quality

The dataset is inspected for:

- Missing values
- Duplicated observations
- Data types
- Descriptive statistics
- Potential inconsistencies

This step is performed before model development to ensure that the data are suitable for subsequent analysis.

### 2. Train-Test Split

The data are divided into:

- **80% training set**
- **20% test set**

The split is stratified according to the target variable to preserve the proportion of churned and non-churned customers.

The test set remains untouched during model selection and hyperparameter optimization.

### 3. Preprocessing

A scikit-learn `ColumnTransformer` and `Pipeline` are used to ensure that preprocessing is integrated into the modelling workflow.

#### Numerical variables

- Median imputation
- Standardization

#### Categorical variables

- Most-frequent imputation
- One-hot encoding

This approach ensures that preprocessing parameters are learned exclusively from the corresponding training data, reducing the risk of data leakage.

---

## Models

Three classification algorithms are evaluated:

### Logistic Regression

Used as an interpretable linear baseline and provides a useful reference for evaluating more complex models.

### Random Forest

An ensemble of decision trees capable of modelling nonlinear relationships and interactions between variables.

### XGBoost

A gradient boosting algorithm based on decision trees. XGBoost is used as the main high-performance candidate because of its ability to capture nonlinear relationships and complex interactions.

---

## Model Evaluation

The primary evaluation metric is **ROC-AUC**, complemented by:

- PR-AUC
- Precision
- Recall
- F1-score
- Accuracy during model comparison

ROC-AUC evaluates the model's ability to discriminate between churned and non-churned customers independently of a specific classification threshold.

PR-AUC is particularly useful in churn prediction because it focuses on the quality of positive-class predictions.

Precision, recall and F1-score are evaluated after converting predicted probabilities into binary classifications.

---

## Cross-Validation

Model development uses **Stratified K-Fold cross-validation**.

The main cross-validation configuration uses:

- 5 folds
- Stratification
- Shuffling
- Fixed random seed

For model comparison, repeated stratified cross-validation is used with:

- 5 folds
- 3 repetitions
- 15 validation folds in total

The fixed random seed is:

`4983`

This improves reproducibility and ensures that the same experimental configuration can be reproduced.

---

## Hyperparameter Optimization

The XGBoost model is further optimized using `RandomizedSearchCV`.

The hyperparameter search explores:

- Number of estimators
- Learning rate
- Maximum tree depth
- Minimum child weight
- Subsample ratio
- Column subsampling
- Gamma
- L1 regularization
- L2 regularization

The optimization uses **ROC-AUC** as the scoring metric and is performed exclusively on the training data using 5-fold stratified cross-validation.

The search evaluates **50 randomly selected hyperparameter combinations**.

Best cross-validation ROC-AUC:

`0.85`

Best hyperparameters:

```text
- classifier__n_estimators: 200
- classifier__learning_rate: 0.05
- classifier__max_depth: 2
- classifier__min_child_weight: 1
- classifier__subsample: 0.7
- classifier__colsample_bytree: 0.6
- classifier__gamma: 0.1
- classifier__reg_alpha: 0.01
- classifier__reg_lambda: 1
```

---

## Final Model

The optimized **XGBoost** pipeline is selected as the final model.

The final pipeline combines:

1. Data preprocessing
2. Feature transformation
3. XGBoost classification

The model is subsequently refitted using the complete training set and evaluated on the previously untouched test set.

This separation between model development and final evaluation provides a more reliable estimate of out-of-sample performance.

---

## Classification Threshold Optimization

The default probability threshold of `0.50` is not necessarily optimal for customer churn prediction.

Because identifying potential churners is particularly important in a retention context, different probability thresholds are evaluated.

Thresholds between `0.10` and `0.90` are tested, and **F1-score** is used as the main criterion for selecting the optimal threshold.

The selected threshold is then applied to the untouched test-set probabilities.

Optimal threshold:

`[FILL IN OPTIMAL THRESHOLD]`

---

## Final Test Performance

The final XGBoost model achieves the following performance on the untouched test set:

| Metric | Test Performance |
|---|---:|
| ROC-AUC | **0.847570** |
| PR-AUC | **0.646249** |
| Precision | **0.558185** |
| Recall | **0.756684** |
| F1-score | **0.642452** |

### Interpretation

The final model achieves a **ROC-AUC of 0.8476**, indicating good discriminative ability between customers who churn and customers who remain.

The **PR-AUC of 0.6462** provides additional information about the model's performance when identifying the positive class.

After threshold optimization, the model achieves a **recall of 75.67%**, meaning that it identifies approximately three quarters of the customers who actually churn.

The corresponding precision is **55.82%**, while the resulting F1-score is **64.25%**, providing a balance between identifying churners and limiting false positive predictions.

For a retention-oriented application, the relatively high recall can be particularly valuable because failing to identify a customer who is likely to churn may represent a greater business opportunity cost than contacting a customer who ultimately remains.

---

## Model Interpretability with SHAP

Predictive performance alone does not explain why a model makes its predictions.

SHAP (**SHapley Additive exPlanations**) is used to interpret the optimized XGBoost model.

The SHAP analysis examines:

- Global feature importance
- The contribution of individual features
- The direction of feature effects
- Which customer characteristics are associated with higher or lower predicted churn probability

The SHAP summary plot provides a global view of how the model uses the available features when generating predictions.

### Main Churn Drivers

The most influential variables identified through the SHAP analysis should be reported here based directly on the generated SHAP summary plot.

1. `Customer Tenure`
2. `Contract Type`
3. `Internet Service`

These findings should be interpreted as **model associations rather than causal relationships**.

---

## Business Implications

The model can potentially support customer retention strategies by assigning each customer an estimated probability of churn.

A telecommunications company could use these predictions to:

- Prioritize high-risk customers.
- Identify customers who may benefit from retention campaigns.
- Allocate retention resources more efficiently.
- Investigate the characteristics associated with elevated churn risk.
- Adjust the classification threshold according to the relative cost of false positives and false negatives.

The optimal threshold should ultimately depend on the company's business objective. For example, a company willing to contact more customers in exchange for identifying more potential churners may prefer a lower threshold and higher recall.

---

## Limitations

Several limitations should be considered:

- The analysis is based on a single telecommunications dataset.
- The model may not generalize directly to other companies, markets or customer populations.
- The data are observational, so the identified relationships should not be interpreted as causal effects.
- The optimal classification threshold depends on the business costs associated with false positives and false negatives.
- External validation using an independent dataset was not performed.
- SHAP explanations describe the behaviour of the trained model and do not establish causal relationships.

---

## Future Work

Potential extensions include:

- Probability calibration.
- Cost-sensitive classification.
- Optimization based on business-specific churn costs.
- Bayesian or more advanced hyperparameter optimization.
- Comparison with additional gradient boosting algorithms such as CatBoost and LightGBM.
- External validation using an independent customer dataset.
- Model monitoring after deployment.
- Deployment through an API.
- Development of an interactive dashboard for customer-risk analysis.

---

## Technologies

The project is implemented in Python using:

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Jupyter Notebook
- KaggleHub

---

## Project Workflow

```text
Raw Data
   │
   ▼
Data Quality Assessment
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Feature / Target Definition
   │
   ▼
Train-Test Split
   │
   ▼
Preprocessing Pipeline
   │
   ▼
Baseline Model
   │
   ▼
Model Comparison
   │
   ├── Logistic Regression
   ├── Random Forest
   └── XGBoost
   │
   ▼
Hyperparameter Optimization
   │
   ▼
Final XGBoost Model
   │
   ▼
Untouched Test Set
   │
   ▼
Threshold Optimization
   │
   ▼
Final Evaluation
   │
   ▼
SHAP Interpretability
   │
   ▼
Business Insights
```

---

## Conclusion

This project presents an end-to-end machine learning workflow for customer churn prediction, combining predictive modelling, robust validation, hyperparameter optimization, threshold selection and model interpretability.

The optimized XGBoost model achieves a **ROC-AUC of 0.8476** and a **PR-AUC of 0.6462** on the untouched test set. After threshold optimization, it reaches a **recall of 75.67%** and an **F1-score of 64.25%**.

The results indicate that the model can effectively distinguish between customers with different churn risks while maintaining a relatively high ability to identify customers who actually churn.

Beyond predictive performance, the use of SHAP provides an interpretable layer that helps connect machine learning predictions with potential customer retention strategies.

Overall, the project demonstrates a complete and reproducible machine learning pipeline, from raw customer data to model evaluation and business-oriented interpretation.

