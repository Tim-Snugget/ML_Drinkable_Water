import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings

MODEL_NAMES = ['Logistic_Regression', 'Decision_Tree', 'Gradient_Boosting', 'Random_Forest', 'KNeighbors',
               'Gaussian_NB', 'SVC']
PATH = "./models"
st.set_page_config(page_title="Drinkable Water", page_icon="🥤", layout='centered', initial_sidebar_state="collapsed")

col1, col2 = st.columns([2, 2])

def load_models():
    models = []
    for mn in MODEL_NAMES:
        models.append((mn, pickle.load(open(f"{PATH}/{mn}.pkl", "rb"))))
    return models

def load_model(modelfile):
	loaded_model = pickle.load(open(f"{PATH}/{modelfile}.pkl", 'rb'))
	return loaded_model

def display_drinkable(name, prediction):
    if prediction.item() == 1:
        col1.success(f"{name} : It is safe to drink that water!")
    else:
        col1.error(f"{name} : It is NOT safe to drink that water...")


def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Drinkable Water  🥤 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # col1,col2  = st.columns([2,2])

    with col1:
        with st.expander(" ℹ️ Information", expanded=True):
            st.write("""
            In Epitech, Tek4 students have the opportunity to travel all around the globe in different countries.  
            However, tap water is never really the same quality and might be bad for the students' health. It is important they know when they can drink the water. 

            """)
        '''
        ## How does it work ❓ 
        Complete all the parameters and the machine learning model will predict if the water can be drunk, based on various parameters
        '''


    with col2:
        st.subheader(" Find out the most suitable crop to grow in your farm 👨‍🌾")
        pH = st.slider("pH", min_value=0.0, max_value=14.0, value=10.0, step=0.1, help="Acid-base level of the water.")
        hardness = st.slider("Hardness", min_value=47.43, max_value=323.12, value=200.0, step=1.0)
        solids = st.slider("Solids", min_value=320.94, max_value=61227.2, value=20000.0, step=200.0)
        chloramines = st.slider("Chloramines", min_value=0.35, max_value=13.13, value=7.0, step=0.5)
        sulfate = st.slider("Sulfate", min_value=129.00, max_value=481.03, value=250.0, step=1.0)
        conductivity = st.slider("Conductivity", min_value=181.48, max_value=753.34, value=300.0, step=5.0)
        organic_carbon = st.slider("Organic Carbon", min_value=2.20, max_value=28.30, value=10.0, step=0.1)
        trihalomethanes = st.slider("Trihalomethanes", min_value=0.74, max_value=124.00, value=40.0, step=0.5)
        turbidity = st.slider("Turbidity", min_value=1.45, max_value=6.74, value=3.0, step=0.05)
        AI_model = st.selectbox("AI model", ['All'] + MODEL_NAMES, index=0)

        feature_list = [pH, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]
        single_pred = np.array(feature_list).reshape(1,-1)
        
        if st.button('Predict'):
            col1.write('''
                		    ## Results 🔍 
                		    ''')

            if AI_model == 'All':
                models = load_models()
                for name, loaded_model in models:
                    prediction = loaded_model.predict(single_pred)
                    display_drinkable(name, prediction)
            else:
                loaded_model = load_model(AI_model)
                prediction = loaded_model.predict(single_pred)
                display_drinkable(AI_model, prediction)

      #code for html ☘️ 🌾 🌳 👨‍🌾  🍃

    # st.warning("Note: This A.I application is for educational/demo purposes only and cannot be relied upon.")
    # hide_menu_style = """
    # <style>
    # #MainMenu {visibility: hidden;}
    # </style>
    # """

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()
