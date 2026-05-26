import streamlit as st
from fastai.learner import load_learner
from fastai.vision.all import *

learn = load_learner('oht_model.pkl')

st.title("OHT Failure Classifier")

uploaded_file = st.file_uploader("Upload contour image", type=['png','jpg'])

if uploaded_file is not None:
    img = PILImage.create(uploaded_file)

    st.image(img)

    pred,pred_idx,probs = learn.predict(img)

    st.write(f"Prediction: {pred}")
    st.write(f"Confidence: {probs.max():.4f}")