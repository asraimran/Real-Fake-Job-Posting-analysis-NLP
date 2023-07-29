import streamlit as st
import tensorflow as tf
from PIL import Image
import pandas as pd  
import pickle 
import base64
from tensorflow.keras.preprocessing.sequence import pad_sequences
# import spacy  
import time
from io import StringIO 
from tensorflow.keras.preprocessing.text import one_hot
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('C:/Users/asrai/JupytrtCodeFiles/NLP project/pic3.png')
def header(url):
     st.markdown(f'<p style="background-color:rgb(0,0,0);color:rgb(255,255,255);font-size:18px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)
def main():
    st.header("Job advertisment prediction")
    st.write("This app is created to predict if a job is real or fake")
    words = st.text_area("Enter")
    if words is not None:
        voc_size=5000
        model = tf.keras.models.load_model("C:/Users/asrai/JupytrtCodeFiles/NLP project/job_model.h5")
        onehot_repr=[one_hot(words,voc_size)] 
        sent_length1=40
        embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length1)
        prediction = model.predict(embedded_docs)
        st.spinner(text="This may take a moment...")
        if st.button("Predict"):
            if (prediction[0] >0.000003):
                text1="This advertisement belonging to fake job post category"
            else:
                text1="This advertisement belonging to real job post category" 
            output =text1
            #st.write(output)
            header(output)
            #st.spinner()
            
            
    
if __name__ == '__main__':
	main()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

































































