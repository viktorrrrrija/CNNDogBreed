import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score
import requests
from io import BytesIO
from definitions import predict_image 
from transformers import AutoImageProcessor, AutoModelForImageClassification
from safetensors.torch import load_file


st.title("Dog Breed Predictions")

option = st.radio("Choose upload option:", ["Upload image", "Insert image URL"])

image = None
if option == "Upload image":
    uploaded_file = st.file_uploader("Upload JPG slike", type=["jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        
else:
    url = st.text_input("Insert image URL:")
    if url:
        try:
            response = requests.get(url, timeout = 10)
            image = Image.open(BytesIO(response.content))
            
        except:
            st.error("URL not valid.")
            
max_width = 400
max_height = 400

if image:
    
    image.thumbnail((max_width, max_height))
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(image)

model_option = st.selectbox("Choose model:", ["ResNet18", "MobileNetV2"])

class_names = ['golden_retriever', 'husky', 'labrador', 'maltese', 'pug']

if st.button("Predict") and image is not None:
    
    if model_option == "ResNet18":
        predicted_class, confidence = predict_image(
            img=image,
            class_names=class_names,
            model_type="custom",
        )
    else:  # Pretrained
        predicted_class, confidence = predict_image(
            img=image,
            class_names=class_names,
            model_type="vit"
        )
        
    st.success(f"Predicted breed: {predicted_class} (confidence: {confidence*100:.2f}%)")
    


