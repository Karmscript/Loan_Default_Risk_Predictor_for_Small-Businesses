#Import neccessary libraries
import joblib
import pandas as pd
import streamlit as st

# Loading model artifacts

# Loan models
model = joblib.load(
    "model_artifacts/loan_default_risk_predictor_model.pkl"
)

# Loan metadata
metadata = joblib.load(
    "model_artifacts/loan_model_metadata.pkl"
)
print(metadata)

# Loan names of feature columns
feature_names = joblib.load(
    "model_artifacts/loan_feature_columns.pkl"
)
print(feature_names)

# Loan model config (threshold)
model_config = joblib.load(
    "model_artifacts/loan_model_config.pkl"
)
THRESHOLD = model_config["threshold"]

def predict_loan_risk(customer_data):

    # Convert the customer's information into a DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Get probability of default
    probability = model.predict_proba(customer_df)[0, 1]

    # Apply chosen threshold
    prediction = int(probability >= THRESHOLD)

    return probability, prediction

# # Testing
# customer_data = {
#     "Gender": "Male",
#     "Married": "Yes",
#     "Dependents": "1",
#     "Education": "Graduate",
#     "Self_Employed": "No",
#     "ApplicantIncome": 5000,
#     "CoapplicantIncome": 2000,
#     "LoanAmount": 150,
#     "Loan_Amount_Term": 240, 
#     "Credit_History": 1,
#     "Property_Area": "Urban"
# }
# (x, y)=predict_loan_risk(customer_data)
# print(x, y)
st.set_page_config(
    page_title="Predict Default Risk Associated With Customer Loan",
    page_icon="💰",
    layout="centered"
)

st.title("Loan Default Risk Predictor")

st.write(
    "Assess the likelihood that a customer will default on a loan"
    "based on historical application data"
)

#Create the borrower application form
Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

Married = st.selectbox(
    "Married",
    ["Yes", "No"]
)

Dependents = st.selectbox(
    "Dependents",
    ["0", "1", "2", "3+"]
)

Education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)
Self_Employed = st.selectbox(
    "Self Employed?",
    ["Yes", "No"]
)
ApplicantIncome = st.number_input(
    "Income",
    min_value=0.0
)
CoapplicantIncome = st.number_input(
    "Income of Coapplicant",
    min_value=0.0
)
LoanAmount = st.number_input(
    "Loan Amount",
    min_value=0.0
)
Loan_Amount_Term = st.number_input(
    "Loan_Amount_Term",
    min_value=0.0,
    max_value=400.0
)
Credit_History = st.radio(
    "Credit History:", 
    options=[0, 1], 
    horizontal=True
)
Property_Area = st.selectbox(
    "Property Area",
    ["Urban", "Rural", "Semiurban"]
)

# Connecting Borrower Application Form Data to Model
if st.button("Assess Loan Risk"):

    customer_data = {
    "Gender": Gender,
    "Married": Married,
    "Dependents": Dependents,
    "Education": Education,
    "Self_Employed": Self_Employed,
    "ApplicantIncome": ApplicantIncome,
    "CoapplicantIncome": CoapplicantIncome,
    "LoanAmount": LoanAmount,
    "Loan_Amount_Term": Loan_Amount_Term, 
    "Credit_History": Credit_History,
    "Property_Area": Property_Area
    }

    probability, prediction = predict_loan_risk(
        customer_data
    )
    # Assessment Results
    st.subheader("Assessment Result")

    st.metric(
    "Probability of Default",
    f"{probability * 100:.1f}%"
    )

    # Risk Assesment
    if prediction == 1:
        st.error("Default Risk Detected")
    else:
        st.success("Lower Default Risk")

