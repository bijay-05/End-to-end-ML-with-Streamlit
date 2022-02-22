import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

std_X = StandardScaler()
pickled_model = pickle.load(open('svm_model.pkl', 'rb'))
st.title('Diabetes Detection System')
st.markdown (
    """
    <style>
    [data-testid='stSidebar'][aria-expanded='true'] > div:first-child{
       width:350px
    }
    [data-testid='stSidebar'][aria-expanded='true'] > div:first-child{
       width:350px
       margin-left: -350px
    }
    </style>


    """,
    unsafe_allow_html=True
)
pregnancies = st.number_input('Pregnancies')
glucose = st.number_input('Glucose')
bloodpressure = st.number_input('BloodPressure')
skinthickness = st.number_input('SkinThickness')
insulin = st.number_input('Insulin')
BMI = st.number_input('BMI')
diabetespf = st.number_input('Diabetes Pedigree Function')
age = st.number_input('Age')

submitted = st.button('Submit')
if submitted:

  df = pd.DataFrame({
      'Pregnancies': [pregnancies],
       'Glucose': [glucose],
        'BloodPressure': [bloodpressure],
         'SkinThickness': [skinthickness],
          'Insulin': [insulin],
           'BMI': [BMI],
            'DiabetesPedigreeFunction': [diabetespf],
             'Age': [age]})
  
  x = pd.DataFrame(std_X.fit_transform(df))
  prediction = pickled_model.predict(x)
  if prediction > 0.5:
    st.write('You are diabetic')
  else:
    st.write('You are non-diabetic')






