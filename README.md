
# Loan Default Risk Predictor

A machine learning-powered decision-support tool for estimating the likelihood that a borrower will default on a loan. The project is designed primarily for micro-lenders, POS agents, and small business owners who need a quick and data-driven assessment of borrower repayment risk.

The system uses an **XGBoost classifier** trained on historical loan application data and provides a predicted probability of default and a final classification based on a selected decision threshold.

An interactive **Streamlit MVP** allows users to enter borrower information and obtain a loan-risk assessment without interacting directly with the underlying machine learning code.

Access the deployed app here:
https://loan-default-risk-assessment-sme.streamlit.app/

---

## Project Overview

Small-scale lenders and POS operators often make lending decisions using limited information and informal judgment. This can make it difficult to consistently assess repayment risk and can expose lenders to avoidable loan losses.

This project explores how supervised machine learning can be used to support these decisions by learning patterns from historical loan applications and estimating the likelihood of default for a new borrower.

The project follows a complete machine learning workflow:

1. Data loading
2. Data wrangling and preprocessing
3. Exploratory data analysis
4. Train/validation/test separation
5. XGBoost model development
6. Hyperparameter optimization
7. Probability-threshold analysis
8. Final model evaluation on an untouched test set
9. Model serialization
10. Deployment as an interactive Streamlit application

---

## Objectives

The main objectives of the project are to:

- Develop a machine learning model for predicting loan default.
- Estimate the probability that a borrower will default.
- Select an appropriate classification threshold based on the validation data.
- Evaluate the model using appropriate classification metrics.
- Package the trained model and preprocessing pipeline for deployment.
- Develop an accessible interface for non-technical users.
- Provide a foundation for future explainability and batch-prediction functionality.

---

## Machine Learning Approach

### Model

The project uses **XGBoost (Extreme Gradient Boosting)** for binary classification.

The model predicts two classes:

| Class | Meaning |
|---|---|
| `0` | No Default |
| `1` | Default |

Rather than relying exclusively on the default 0.50 classification threshold, the model produces a probability of default and applies a threshold selected using the validation data.

### Decision Threshold

The selected classification threshold is:

```text
0.25
````

The decision rule is:

```text
Probability of Default >= 0.25
        → Default Risk

Probability of Default < 0.25
        → No Default Risk
```

The threshold was selected after analyzing the trade-off between precision, recall, and F1-score on the validation data.

This threshold is **not an XGBoost hyperparameter**. It is a decision parameter applied after the trained model generates a probability.

---

## Data Preprocessing

The preprocessing workflow was incorporated into the saved machine learning pipeline.

This approach ensures that the same preprocessing operations used during model development are automatically applied when the deployed application receives new borrower information.

The saved pipeline contains:

```text
Raw borrower information
        ↓
Preprocessing
        ↓
XGBoost model
        ↓
Probability of default
```

This prevents the deployed application from having to manually reproduce the preprocessing workflow.

---

## Model Development

The model development process included:

### 1. Data Wrangling

The data was inspected and prepared for machine learning, including handling the structure and data types required by the model.

### 2. Exploratory Data Analysis

Exploratory analysis was performed to understand:

* Feature distributions
* Target-class distribution
* Relationships between variables
* Potential data-quality issues
* Patterns associated with loan default

### 3. Model Training

An initial XGBoost classifier was trained as a baseline.

### 4. Hyperparameter Optimization

Multiple approaches were explored for improving the XGBoost model, including:

* Initial model training
* RandomizedSearchCV
* Optuna

The final model was selected after the hyperparameter-tuning process.

### 5. Threshold Optimization

The model's predicted probabilities were evaluated across different classification thresholds.

This was necessary because the default threshold of 0.50 produced relatively low recall on the validation data.

A threshold of 0.25 provided a better balance for the project's objective of identifying potential defaulters.

---

## Final Test Performance

After model development and threshold selection, the final model was evaluated on an **untouched test set**.

The test set was not used for hyperparameter tuning or threshold selection.

### Results

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 79.72.52% |
| Precision | 50.00% |
| Recall    | 66.67% |
| F1 Score  | 57.14% |
| ROC-AUC   | 85.31% |

### Confusion Matrix

|                       | Predicted No Default | Predicted Default |
| --------------------- | -------------------: | ----------------: |
| **Actual No Default** |              49 (TN) |           10 (FP) |
| **Actual Default**    |              5  (FN) |           10 (TP) |

The model correctly identified **25 of the 37 actual defaulters** in the test set, while 12 actual defaulters were missed.

The model also incorrectly classified 21 non-defaulters as potential defaulters.

These results demonstrate that the model has meaningful predictive signal while also highlighting the trade-off between detecting potential defaulters and avoiding false alarms.

---

## Why Recall Matters

For a loan-default prediction system, false negatives can be particularly costly.

A false negative occurs when:

```text
Actual borrower → Defaults
Model prediction → No Default
```

This represents a borrower who is considered sufficiently safe by the model but subsequently defaults.

The selected threshold therefore places considerable emphasis on identifying potential defaulters.

The final test recall of **68.42%** means that the model detected approximately 68 out of every 100 actual defaulters in the test set.

However, the model should be considered a **decision-support tool rather than an autonomous lending decision-maker**.

---

## MVP Application

The trained model has been integrated into an interactive **Streamlit application**.

The current MVP allows a user to:

1. Enter borrower information through a form.
2. Submit the borrower information.
3. Pass the information through the saved preprocessing + XGBoost pipeline.
4. Generate a probability of default.
5. Apply the selected 0.25 classification threshold.
6. Display the resulting loan-risk classification.

### Application Workflow

```text
Borrower Information
        ↓
Streamlit Interface
        ↓
Saved ML Pipeline
        ↓
Preprocessing
        ↓
XGBoost
        ↓
Probability of Default
        ↓
Threshold = 0.25
        ↓
Risk Classification
```

---

## Model Artifacts

The trained model and supporting configuration are stored as serialized Python objects.

```text

    └── model_artifacts/
        ├── loan-train.csv
        ├── Loan_Default_Risk_Predictor.ipynb
        ├── loan_default_risk_predictor_model.pkl
        ├── loan_feature_columns.pkl
        ├── loan_model_config.pkl
        └── loan_model_metadata.pkl

```

### `loan_default_model.pkl`

Contains the complete trained preprocessing + XGBoost pipeline.

### `model_config.pkl`

Contains the selected classification threshold.

### `model_metadata.pkl`

Contains information about the model, target variable, class definitions, and threshold.

### `feature_columns.pkl`

Contains the expected input features for the model.

---

## Project Structure

```text
└── Loan-Default-Risk-Analysis/
    ├── app.py
    ├── README.md
    ├── requirements.txt
    └── model_artifacts/
        ├── loan-train.csv
        ├── Loan_Default_Risk_Predictor.ipynb
        ├── loan_default_risk_predictor_model.pkl
        ├── loan_feature_columns.pkl
        ├── loan_model_config.pkl
        └── loan_model_metadata.pkl

```

---

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate into the project directory:

```bash
cd loan-default-predictor
```

Create and activate a virtual environment if desired:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Run the Streamlit application with:

```bash
streamlit run app.py
```

The application will open in a web browser.

If Streamlit is not recognized as a command, use:

```bash
python -m streamlit run app.py
```

---

## Example Use Case

A POS agent or small business owner can enter the information of a prospective borrower into the application.

The system processes the information and returns a result such as:

```text
Probability of Default: 37.4%

Model Classification:
Default Risk
```

The classification is determined using the selected threshold:

```text
37.4% >= 25%
        ↓
Default Risk
```

---

## Limitations

This project is an **MVP and research/portfolio prototype**, not a production-ready credit underwriting system.

Important limitations include:

* Model performance depends heavily on the quality and representativeness of the training data.
* The available dataset may not fully represent the Nigerian micro-lending and POS lending environment.
* The test dataset is relatively limited in size.
* A false positive can result in a potentially creditworthy borrower being rejected.
* A false negative can result in lending to a borrower who subsequently defaults.
* The model's predictions should therefore not be treated as guaranteed outcomes.
* The selected threshold reflects the current development objective and should be reconsidered when actual lending costs and business constraints are available.
* Additional validation using a larger, geographically relevant dataset is required before real-world deployment.

---

## Future Improvements

Planned improvements include:

* **SHAP-based explainability** for individual borrower predictions.
* Key-factor explanations showing why a borrower was classified as risky.
* Batch CSV prediction.
* Downloadable prediction reports.
* Improved risk-level categorization.
* Cost-sensitive threshold optimization based on actual lender economics.
* Model calibration to improve the interpretation of predicted probabilities.
* Evaluation using larger and more representative Nigerian lending datasets.
* Monitoring of model performance after deployment.
* Model retraining using new repayment outcomes.
* Improved UI/UX for POS agents and small-business users.
* Deployment on a scalable production infrastructure.

---

## Disclaimer

This application is intended for **educational, research, and decision-support purposes**.

A machine learning prediction is not a guarantee that a borrower will or will not repay a loan. Lending decisions should incorporate additional information, appropriate risk-management procedures, applicable regulations, and human judgment.

---

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Optuna**
* **Matplotlib**
* **Joblib**
* **Streamlit**
* **Google Colab / Jupyter Notebook**

---

## Author

**Ahmed Ramadan Bamidele**

This project was developed as a machine learning application exploring the use of predictive analytics for loan-default risk assessment and decision support for small-scale lenders.


Access the deployed app online here:
https://loan-default-risk-assessment-sme.streamlit.app/
```