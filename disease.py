import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
# from keras.preprocessing import image
import keras.utils as image
from PIL import Image
import streamlit as st
import io
def set_bg_hackd_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://wallpaperbat.com/img/146005-ultra-hd-macro-wallpaper-top-free-ultra-hd-macro-background.jpg");
            
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
def set_bg_hackn_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://wallpaperbat.com/img/27368-green-leaf-wallpaper-hd.jpg");
            
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Function to load uploaded image
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Function to check the image for nutrient deficiency
@st.cache(ttl=48*3600)
def check():

    lr = keras.models.load_model('weights.hdf5')
    #Prediction Pipeline
    class Preprocessor(BaseEstimator, TransformerMixin):
        def fit(self,img_object):
            return self
        
        def transform(self,img_object):
            img_array = image.img_to_array(img_object)
            expanded = (np.expand_dims(img_array,axis=0))
            return expanded

    class Predictor(BaseEstimator, TransformerMixin):
        def fit(self,img_array):
            return self
        
        def predict(self,img_array):
            probabilities = lr.predict(img_array)
            predicted_class = ['P_Deficiency', 'Healthy', 'N_Deficiency', 'K_Deficiency'][probabilities.argmax()]
            return predicted_class

    full_pipeline = Pipeline([('preprocessor',Preprocessor()),
                            ('predictor',Predictor())])
    return full_pipeline
# Function to check the image for disease detection
@st.cache(ttl=48*3600)
def checkdisease():

    lr = keras.models.load_model('model.h5')
    #Prediction Pipeline
    class Preprocessor(BaseEstimator, TransformerMixin):
        def fit(self, img_object):
            return self
            
        def transform(self, img_object):
            img_byte_arr = io.BytesIO()
            img_object.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            img = image.load_img(img_byte_arr, target_size=(256, 256))
            img_array = image.img_to_array(img)
            expanded = np.expand_dims(img_array, axis=0)
            return expanded  

    class Predictor(BaseEstimator, TransformerMixin):
        def fit(self,img_array):
            return self
        
        def predict(self,img_array):
            probabilities = lr.predict(img_array)
            predicted_class = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_healthy'] [probabilities.argmax()]
            return predicted_class

    full_pipeline = Pipeline([('preprocessor',Preprocessor()),
                            ('predictor',Predictor())])
    return full_pipeline

def output(full_pipeline, img):
    a = img
    a = a.resize((224, 224))
    predic = full_pipeline.predict(a)
    return predic

def main():
    # Sidebar options
    options = ['Plant Disease Detection', 'Nutrient Deficiency Detection']
    selected_option = st.sidebar.selectbox('Select an option:', options)

    # Page navigation based on the selected option
    if selected_option == 'Plant Disease Detection':
        st.title('Plant Disease Detection')
        disease()
        # Add your code for plant disease detection page here
    elif selected_option == 'Nutrient Deficiency Detection':
        st.title('Nutrient Deficiency Detection')
        nutrient()
    else:
        # Blank page
        pass

def nutrient():
    # Giving a title
    set_bg_hackn_url()
    col1, col2 = st.columns(2)

    with col1:
        st.title('P.N.D')
        image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
        # Code for prediction
        prediction = ''

        # Creating a button for prediction
        if st.button('Predict'):
            if image_file is not None:
                # To see details
                with st.spinner('Loading Image and Model...'):
                    full_pipeline = check()
                file_details = {"filename": image_file.name, "filetype": image_file.type, "filesize": image_file.size}
                st.write(file_details)
                img = load_image(image_file)
                w = img.size[0]
                h = img.size[1]
                if w > h:
                    w = 600
                    st.image(img, width=w)
                else:
                    w = w * (600.0 / h)
                    st.image(img, width=int(w))
                with st.spinner('Predicting...'):
                    prediction = output(full_pipeline, img)
                st.success(prediction)
def disease():
    # Giving a title
    set_bg_hackd_url()
    col1, col2 = st.columns(2)

    with col1:
        st.title('P.D.D')
        
        image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
        # Code for prediction
        prediction = ''

        # Creating a button for prediction
        if st.button('Predict'):
            if image_file is not None:
                # To see details
                with st.spinner('Loading Image and Model...'):
                    full_pipeline = checkdisease()
                file_details = {"filename": image_file.name, "filetype": image_file.type, "filesize": image_file.size}
                st.write(file_details)
                img = load_image(image_file)
                w = img.size[0]
                h = img.size[1]
                if w > h:
                    w = 600
                    st.image(img, width=w)
                else:
                    w = w * (600.0 / h)
                    st.image(img, width=int(w))
                with st.spinner('Predicting...'):
                    prediction = output(full_pipeline, img)
                st.success(prediction)


if __name__ == '__main__':
    main()