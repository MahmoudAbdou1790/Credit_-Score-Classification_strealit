import streamlit as st
import pickle
import pandas as pd
import time
from PIL import Image
import streamlit.components.v1 as components
import os

import matplotlib.pyplot as plt
from xgboost import XGBClassifier


data=  pickle.load(open('data_for_deploy.pkl', 'rb'))
#loaded_model = XGBClassifier()

# Model Saving
#file_path = os.path.abspath('F:/data science/Final_Project_Criteria_Dataset/preprocessor.pkl')

# Load Preprocessor
 

#loaded_le = pickle.load(open('F:/data science/Final_Project_Criteria_Dataset/preprocessor.pkl', 'rb'))
#loaded_enc = pickle.load(open('F:/data science/Final_Project_Criteria_Dataset/model.pkl', 'rb'))

#df[cat] = ['Credit_Mix']





col1, col2, col3 = st.columns([1,8,1]) 

    #try:
    #image url
url = "https://storage.googleapis.com/kaggle-datasets-images/2289007/3846912/ad5e128929f5ac26133b67a6110de7c0/dataset-cover.jpg?"
    
    #Image, df

st.image(url, caption="Kaggle: Credit score classification")        
st.markdown('[Kaggle: Credit Score Classification  Dataset](https://www.kaggle.com/datasets/parisrohan/credit-score-classification/data)')
        
 #Define User input
def user_input_data():
    # Sidebar Configuration
    Age = st.number_input('Age', min_value=18, max_value=100)
    Month = st.number_input('Month', min_value=1, max_value=8 )
    Annual_Income = st.number_input('Annual_Income', min_value=0.00, max_value=300000.00)
    Monthly_Inhand_Salary       = st.sidebar.slider('Monthly_Inhand_Salary',min_value=2000, max_value=20000)
    Monthly_Balance       = st.sidebar.slider('Monthly_Inhand_Salary',min_value=500, max_value=20000)
    Total_EMI_per_month    = st.number_input('Total_EMI_per_month', min_value=0.00, max_value=5000.00)
    Num_Bank_Accounts      = st.number_input('Num_Bank_Accounts', min_value=0, max_value=20)
    Num_Credit_Inquiries       = st.sidebar.slider('Num_Credit_Inquiries',min_value=0, max_value=20000)
    Credit_History_Age       = st.number_input('Credit_History_Age', min_value=0, max_value=500)
    Num_of_Loan       = st.sidebar.slider('Num_of_Loan',min_value=0.00, max_value=12.0)
    Num_of_Delayed_Payment = st.sidebar.slider('Num_of_Delayed_Payment', 0, 25, 14, 1)
    Payment_Behaviour = st.sidebar.slider('Payment_Behaviour', min_value=1, max_value=6) 
    Credit_Utilization_Ratio = st.slider('Credit_Utilization_Ratio', min_value=0.00, max_value=100.00)
    Delay_from_due_date = st.number_input('Delay_from_due_date', min_value=0, max_value=20)
    Changed_Credit_Limit   = st.sidebar.slider('Changed_Credit_Limit', 0.5, 30.0, 9.40, 0.1)
    Amount_invested_monthly= st.number_input('Amount_invested_monthly', min_value=0, max_value=20)
    Num_Credit_Card        = st.number_input('Num_Credit_Card', min_value=0, max_value=12)
    Outstanding_Debt       = st.sidebar.slider('Outstanding_Debt', 0.0, 5000.0, 1426.0, 0.1)
    Interest_Rate          = st.sidebar.slider('Interest_Rate', 1, 34, 14, 1)   
    Age_Group          = st.sidebar.slider('Age_Group',  min_value=1, max_value=4)
    Delay_Group          = st.sidebar.slider('Delay_Group', min_value=1, max_value=4)
    #Credit_Mix             = st.sidebar.selectbox('Credit_Mix:', ['Standard', 'Bad', 'Good'])    
    Credit_Mix             = st.sidebar.slider('Credit_Mix:', min_value=0, max_value=2)
    Occupation             = st.number_input('Month', min_value=1, max_value=15 )
    Payment_of_Min_Amount = st.number_input('Payment_of_Min_Amount', min_value=0, max_value=2)
    
    
    
    html_temp = """
    <div style="background-color:tomato;padding:1.5px">
    <h1 style="color:white;text-align:center;">Single Customer </h1>
    </div><br>"""
    st.sidebar.markdown(html_temp,unsafe_allow_html=True)
    

    new_data = {'Age'                  :   Age,
                'Month'                 : Month,
                'Occupation'            : Occupation,
                'Credit_History_Age'    : Credit_History_Age,
                'Payment_of_Min_Amount' : Payment_of_Min_Amount,
                'Age_Group'             : Age_Group,
                'Delay_Group'           : Delay_Group,
                'Annual_Income'         : Annual_Income,
                'Monthly_Inhand_Salary' : Monthly_Inhand_Salary,
                'Monthly_Balance'       : Monthly_Balance,
                'Credit_Utilization_Ratio':Credit_Utilization_Ratio,
        
                'Num_Credit_Inquiries'  : Num_Credit_Inquiries,
                'Num_of_Loan'           : Num_of_Loan,
                'Amount_invested_monthly': Amount_invested_monthly,
                'Payment_Behaviour'     : Payment_Behaviour,
                'Total_EMI_per_month'   : Total_EMI_per_month,
                'Num_Bank_Accounts'     : Num_Bank_Accounts,
                'Num_of_Delayed_Payment': Num_of_Delayed_Payment, 
                'Delay_from_due_date'   : Delay_from_due_date,
                'Changed_Credit_Limit'  : Changed_Credit_Limit,
                'Num_Credit_Card'       : Num_Credit_Card,        
                'Outstanding_Debt'      : Outstanding_Debt,
                'Interest_Rate'         : Interest_Rate,       
                'Credit_Mix'            : Credit_Mix,
        
         
    }
    input_data = pd.DataFrame(new_data, index=[0])  
    
    return input_data
    
# Sidebar Configuration
# Add a sidebar to the web page. 
st.sidebar.header("User input parameter")

# get input datas
col1, col2 = st.columns([4, 6])
# st.sidebar.write('Developed by ...')
# st.sidebar.write('Contact at ...')


df = user_input_data() 
with col1:
    if st.checkbox('Show User Inputs:', value=True):
        st.dataframe(df.astype(str).T.rename(columns={0:'input_data'}).style.highlight_max(axis=0))

with col2:
    for i in range(2): 
        st.markdown('#')
    if st.button('Make Prediction'):   
        sound = st.empty()
        # assign for music sound
        video_html = """
            <iframe width="0" height="0" 
            src="https://www.youtube-nocookie.com/embed/t3217H8JppI?rel=0&amp;autoplay=1&mute=0&start=2860&amp;end=2866&controls=0&showinfo=0" 
            allow="autoplay;"></iframe>
            """
        sound.markdown(video_html, unsafe_allow_html=True)
        
        
        # Use the loaded model for predictions
        #df[cat]    = loaded_enc.transform(df[cat]) 
        #prediction = loaded_model.predict(df)
        #prediction = loaded_le.inverse_transform(prediction)[0]
        
        preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
        df_preprocessed = preprocessor.transform(df)   
        model = pickle.load(open('model.pkl', 'rb'))
        Credit_Score = model.predict(df_preprocessed) # in log scale
        #Credit_Score1 =loaded_le.inverse_transform(Credit_Score)[0]

        time.sleep(3.7)  # wait for 2 seconds to finish the playing of the audio
        sound.empty()  # optionally delete the element afterwards   
        
        st.success(f'Credit score probability is:&emsp;{Credit_Score}')

