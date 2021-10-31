import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings
from sklearn.metrics import accuracy_score

MODEL_NAMES = ['Logistic_Regression', 'Decision_Tree', 'Gradient_Boosting', 'Random_Forest', 'KNeighbors',
               'Gaussian_NB', 'SVC']
PATH = "./models"

# y_test = np.loadtxt('./misc/y_test.txt')
# y_test = np.array(y_test).reshape(1, -1)
y_test = pd.Series.from_csv('./misc/y_test.txt')

st.set_page_config(page_title="Drinkable Water", page_icon="🥤", layout='centered', initial_sidebar_state="collapsed")

def load_models():
    models = []
    for mn in MODEL_NAMES:
        models.append((mn, pickle.load(open(f"{PATH}/{mn}.pkl", "rb"))))
    return models

def load_model(modelfile):
	loaded_model = pickle.load(open(f"{PATH}/{modelfile}.pkl", 'rb'))
	return loaded_model


def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Drinkable Water  🥤 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.text(y_test.shape)
    st.text(y_test.shape)
    st.text(type(y_test))

    col1,col2  = st.columns([2,2])

    def display_drinkable(loaded_model, X_test):
        name_model, model = loaded_model
        prediction = model.predict(X_test)
        st.text(prediction.shape)
        if prediction.item() == 1:
            if not disp_accuracy:
                col1.success(f"{name_model} : It is safe to drink that water!")
            else:
                accuracy = accuracy_score(y_test, prediction)
                col1.success("{} : It is safe to drink that water!\nAccuracy of {:.2f}%".format(name_model, accuracy*100))
        else:
            if not disp_accuracy:
                col1.error(f"{name_model} : It is NOT safe to drink that water...")
            else:
                accuracy = accuracy_score(y_test, prediction)
                col1.error("{} : It is NOT safe to drink that water...\nAccuracy of {:.2f}%".format(name_model, accuracy*100))

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
        st.subheader(" Find out if the water is drinkable ‍🥤")
        pH = st.slider("pH", min_value=0.0, max_value=14.0, value=10.0, step=0.1, help="Acid-base level of the water.")
        hardness = st.slider("Hardness", min_value=47.43, max_value=323.12, value=200.0, step=1.0, help="Concentration of calcium and magnesium salts.")
        solids = st.slider("Solids", min_value=320.94, max_value=61227.2, value=20000.0, step=200.0, help="The ability for the water to dissolve solids (minerals).")
        chloramines = st.slider("Chloramines", min_value=0.35, max_value=13.13, value=7.0, step=0.5, help="Water disinfectants concentration (recommended amount is 4mg/L).")
        sulfate = st.slider("Sulfate", min_value=129.00, max_value=481.03, value=250.0, step=1.0, help="Sulfates is present is most minerals, soil and rocks.")
        conductivity = st.slider("Conductivity", min_value=181.48, max_value=753.34, value=300.0, step=5.0, help="Electrical conductivity measures the ionic concentration (should not exceed 400 μS/cm).")
        organic_carbon = st.slider("Organic Carbon", min_value=2.20, max_value=28.30, value=10.0, step=0.1, help="Total Organic Carbon represents the natural decaying organic matter organic matter and synthetic sources.")
        trihalomethanes = st.slider("Trihalomethanes", min_value=0.74, max_value=124.00, value=40.0, step=0.5, help="THMs can be found in water treated with chlorine.")
        turbidity = st.slider("Turbidity", min_value=1.45, max_value=6.74, value=3.0, step=0.05, help="The turbidity of water depends on the quantity of solid matter present in the suspended state.")
        AI_model = st.selectbox("AI model", ['All'] + MODEL_NAMES, index=0, help="Wanted model to predict whether the water is drinkable or not!")
        disp_accuracy = st.checkbox("Display Accuracy Score", value=False, help="Accuracy level of the model displayed.")

        feature_list = [pH, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]
        single_pred = np.array(feature_list).reshape(1,-1)
        
        if st.button('Predict'):
            col1.write('''
                		    ## Results 🔍 
                		    ''')

            if AI_model == 'All':
                models = load_models()
                for loaded_model in models:
                    # prediction = loaded_model.predict(single_pred)
                    display_drinkable(loaded_model, single_pred)
            else:
                loaded_model = load_model(AI_model)
                # prediction = loaded_model.predict(single_pred)
                display_drinkable(loaded_model, single_pred )

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
