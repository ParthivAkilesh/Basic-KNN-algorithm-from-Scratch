import numpy as np
import pickle
import pandas as pd
import streamlit as st 
from sklearn.datasets import  make_classification
x,y = make_classification(n_samples=2500, n_features=2, n_redundant=0,n_classes=2, class_sep=1.5)

mod = open('model.pkl', 'rb')
model = pickle.load(mod)

def welcome():
    return "Welcome All"

def knn(x,y):
     
    p = np.array([x,y]).reshape(1,-1)
    prediction = model.predict(p)
    print(prediction[0])
    return prediction[0]



def main():
    st.title("KNN Classification")
    html_temp = """
    <div style="background-color:rgb(4, 22, 82);padding:10px">
    <h2 style="color:rgb(241, 153, 20);text-align:center;">WebApp for KNN Classification</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    
    x = st.slider('Enter x :',min_value=0, max_value=5, value=0, step=1)
    x = int(x)

    y = st.slider('Enter y :',min_value=0, max_value=5, value=0, step=1)
    y = int(y)


    	
	
    result=""
    if st.button("Classify"):
        result=knn(x,y)
        st.success('The result of the classification is  {}'.format(result))
    
    
    if st.button("About"):
        st.text("This is a KNN classification model made from scratch using StreamLit")
        st.text("This model is built by Parthiv Akilesh A S")

if __name__=='__main__':
    main()
    
