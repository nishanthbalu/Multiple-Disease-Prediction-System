import streamlit as st
import pandas as pd
import numpy as np
import pickle



#load the saved model
model_dict = pickle.load(open("model_save/diabetes_knn_model.pkl", "rb"))
model_dict1 = pickle.load(open("model_save/heart_knn_model.pkl", "rb"))
scaler = model_dict["scaler"]
scaler_heart = model_dict1["scaler_heart"]
model_diabetes = model_dict["classifier_knn"]
model_heart = model_dict1["classifier_heart"]


# LINK TO THE CSS FILE
with open('static/style.css') as f:
 st.markdown(f"<style>{f.read()}</style>", 
unsafe_allow_html = True)

st.markdown("<hr/>", unsafe_allow_html = True)


def diabetes_input():

    col1, col2 = st.columns(2)
    with col1:
        a = float(st.number_input("No of Pregnancies",step= 0.1))
    with col2:
        b = float(st.number_input("Glucose Level In Blood",step= 0.1))
    with col1:
        c = float(st.number_input("Blood Pressure",step= 0.1))
    with col2:
        d = float(st.number_input("Skin Thickness",step= 0.1))
    with col1:
        e = float(st.number_input("Insulin Level In Blood",step= 0.1))
    with col2:
        f = float(st.number_input("Body Mass Index (BMI)",step= 0.1))
    with col1:
        g = float(st.number_input("Diabetes Percentage",step= 0.1))
    with col2:
        h = float(st.number_input("Age",step= 0.1))

    if st.button("Check Test Result"):
        data =  [[a,b,c,d,e,f,g,h]]
        input_data = data
        input_data_scaled = scaler.transform(input_data)
        prediction = model_diabetes.predict(input_data_scaled)[0]


        if prediction == 0:
            st.caption("Not Diabetic Person")

        if prediction ==1:
            st.caption("Diabetic person")

    

def heart_input():
    col1, col2 = st.columns(2)
    
    with col1:
        a1 = float(st.number_input("Age",step= 0.1))
    with col2:
        b1 = float(st.number_input("Sex(Male(1)/Female(0))",step= 0.1))
    with col1:
        c1 = float(st.number_input("Chest Pain Type",step= 0.1))
    with col2:
        d1 = float(st.number_input("Resting Blood Pressure(mmHg)",step= 0.1))
    with col1:
        e1 = float(st.number_input("Serum Cholestoral(mg/dl)",step= 0.1))
    with col2:
        f1 = float(st.number_input("Fasting Blood Sugar >120 mg/dl(True(1)/ False(0))",step= 0.1))
    with col1:
        g1 = float(st.number_input("Resting Electrocardiographic Result",step= 0.1))
    with col2:
        h1 = float(st.number_input("Maximum Heart Rate Achieved",step= 0.1))
    with col1:
        i1 = float(st.number_input("Exercise Induced Angina(Yes(1)/No(0))",step= 0.1))
    with col2:
        j1 = float(st.number_input("ST Depression Induced By Exercise Relative To Rest",step= 0.1))
    with col1:
        k1 = float(st.number_input("Slope Of The Peak Exercise ST Segment",step= 0.1))
    with col2:
        l1 = float(st.number_input("Number Of Major Vessels Colored By Flourosopy ",step= 0.1))
    with col1:
        m1 = float(st.number_input("Thal(Normal(3)/Fixed Defect(6)/Reversable Defect(7)) ",step= 0.1))


    if st.button("Check Test Result"):

        input_data=[[a1,b1,c1,d1,e1,f1,g1,h1,i1,j1,k1,l1,m1]]
        input_data_scaled = scaler_heart.transform(input_data)
        prediction = model_heart.predict(input_data_scaled)[0]

        if prediction == 0:
            st.caption("No Heart Disease Found")

        if prediction ==1:
            st.caption("Heart Disease Found")



# session state management
if 'submits' not in st.session_state:
    st.session_state.submits = False

def callback():
    st.session_state.submits =True

def callback_state():
    if 'submits' in st.session_state:
        st.session_state.submits = False



     
    
    
  
def main():

    st.title("Multiple Disease Prediction System")

    st.image("https://nhfm.net/media/cropped-health-education.jpg")

    option = st.selectbox("Choose Disease Prediction",("Diabetes Disease Prediction","Heart Disease Prediction"),on_change=callback_state)

    submit = st.button("Submit",on_click=callback)

    if st.session_state.submits == True:
        if option == 'Diabetes Disease Prediction':
            diabetes_input()
            
        if option ==  'Heart Disease Prediction':
            heart_input()
        
     
    
                   
   
if __name__ == "__main__":
    main()



