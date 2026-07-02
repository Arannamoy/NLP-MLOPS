import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import os

# ### Load the trained model, scaler pickle,onehot
model=load_model('./model.h5')
with open('label_encoder_gender.pkl','rb') as fl:
    label_encoder_gender=pickle.load(fl)

with open('open_hot_encoder_geo.pkl','rb') as fl:
    open_hot_encoder_geo=pickle.load(fl)

with open("scaler.pkl","rb") as fl:
    scaler=pickle.load(fl)


st.title("Customer Churn Prediction")
geography=st.selectbox('Geography',open_hot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])


input_data= {
    'CreditScore':credit_score,
    'Geography':geography,
    'Gender':gender,
    'Age':age,
    'Tenure':tenure,
    'Balance':balance,
    'NumOfProducts':num_of_products,
    'HasCrCard':has_cr_card,
    'IsActiveMember':is_active_member,
    'EstimatedSalary':estimated_salary
}

geo_encoded=open_hot_encoder_geo.transform([[input_data['Geography']]])
geo_encoded_df=pd.DataFrame(geo_encoded,columns=open_hot_encoder_geo.get_feature_names_out(['Geography']))
input_df=pd.DataFrame([input_data])
input_df['Gender']=label_encoder_gender.transform(input_df['Gender'])
input_df=pd.concat([input_df.drop('Geography',axis=1),geo_encoded_df],axis=1)
input_scaled=scaler.transform(input_df)
prediction_prob=model.predict(input_scaled)
if(prediction_prob[0][0]>0.5):
    st.text(f"{prediction_prob[0][0]}\n The customer is likely to churn")
    print('The customer is likely to churn')
else:
    st.text(f"{prediction_prob[0][0]}\n The customer is not likely to churn")
    print('The customer is not likely to churn.')