from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import base64
from transformers import pipeline
import joblib
from PIL import Image

classifier = joblib.load("classifier.pkl")
def get_best_label(predictions):
    max_score = 0
    label = ""
    for p in predictions:
        if p['score'] > max_score:
            max_score = p['score']
            label = p['label']
    return label, max_score

st.markdown('<h1 style="color:black;">Document Classifier</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">This model can classify input image to the following categories:</h2>', unsafe_allow_html=True)
st.markdown('<h3 style="color:gray;"> <ul> <li>Invoice</li> <li>Bank statement</li> <li>Credit bureau</li> </ul> </h3>', unsafe_allow_html=True)



upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
c1, c2= st.columns(2)
if upload is not None:
  image = Image.open(upload)
  c1.header('Input Image')
  c1.image(image)
  print("c1", c1)
  print("c2", c2)
  c2.header('Output')
  c2.subheader('Predicted class :')
  predictions = classifier(image, candidate_labels=["invoice, receipt", "bank statement, financial statement", "credit report"])
  c2.subheader('Predicted class :' + str(get_best_label(predictions)))
  c2.write(str(predictions))
