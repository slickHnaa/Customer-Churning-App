import gradio as gr
import pickle
import pandas as pd
import numpy as np
import joblib
from gradio.components import *


#define prediction function
def make_prediction(gender, Partner, Dependents, tenure, MultipleLines,
       InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
       TechSupport, Contract, PaperlessBilling, PaymentMethod,
       MonthlyCharges, TotalCharges):
    #make a dataframe from input data
    input_data = pd.DataFrame({'gender':[gender], 'Partner':[Partner], 'Dependents':[Dependents], 'tenure':[tenure], 'MultipleLines':[MultipleLines],
       'InternetService':[InternetService], 'OnlineSecurity':[OnlineSecurity], 'OnlineBackup':[OnlineBackup], 'DeviceProtection':[DeviceProtection],
       'TechSupport':[TechSupport], 'Contract':[Contract], 'PaperlessBilling':[PaperlessBilling], 'PaymentMethod':[PaymentMethod],
       'MonthlyCharges':[MonthlyCharges], 'TotalCharges':[TotalCharges]})
       
   #load already saved pipeline and make predictions
    with open("best_model.joblib", "rb") as f:
        model = joblib.load(f)
        predt = model.predict(input_data) 
    #return prediction 
    if predt == 'Yes':
        return 'Customer Will Churn'
    return 'Customer Will Not Churn'

#create the input components for gradio
gender_input = gr.Dropdown(choices =['Female', 'Male']) 
Partner_input = gr.Dropdown(choices =['Yes', 'No']) 
Dependents_input = gr.Dropdown(choices =['Yes', 'No'])
tenure_input = gr.Number()
MultipleLines_input = gr.Dropdown(choices =['No phone service', 'No', 'Yes'])
InternetService_input = gr.Dropdown(choices =['DSL', 'Fiber optic', 'No']) 
OnlineSecurity_input = gr.Dropdown(choices =['No', 'Yes', 'No internet service']) 
OnlineBackup_input = gr.Dropdown(choices =['Yes', 'No', 'No internet service']) 
DeviceProtection_input = gr.Dropdown(choices =['No', 'Yes', 'No internet service'])
TechSupport_input = gr.Dropdown(choices =['No', 'Yes', 'No internet service'])
Contract_input = gr.Dropdown(choices =['Month-to-month', 'One year', 'Two year'])
PaperlessBilling_input = gr.Dropdown(choices =['Yes', 'No']) 
PaymentMethod_input = gr.Dropdown(choices =['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'])    
MonthlyCharges_input = gr.Number()
TotalCharges_input = gr.Number()

output = gr.Textbox(label='Prediction')

app = gr.Interface(fn =make_prediction, inputs =[gender_input,
                                                 Partner_input,
                                                 Dependents_input,
                                                 tenure_input,
                                                 MultipleLines_input,
                                                 InternetService_input,
                                                 OnlineSecurity_input,
                                                 OnlineBackup_input,
                                                 DeviceProtection_input,
                                                 TechSupport_input,
                                                 Contract_input,
                                                 PaperlessBilling_input,
                                                 PaymentMethod_input,
                                                 MonthlyCharges_input,
                                                 TotalCharges_input], outputs = output)

app.launch(share = True, debug = True)
