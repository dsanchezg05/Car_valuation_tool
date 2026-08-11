import streamlit as st
import pandas as pd
import numpy as np
import random
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import copy
import pickle as pkl

#Import df
last_df = pd.read_parquet("pre_norm_dummied_df.parquet", engine="pyarrow")
log_price_df = copy.deepcopy(last_df)
def log_price(x):
    return np.log1p(x)
log_price_df["price"] = log_price_df['price'].apply(log_price)


#import coef dict
with open("sorted_coef_dict.pkl","rb") as f:
    sorted_coef_dict = pkl.load(f)

#Import the model
loaded_model = joblib.load('elastic_net_model.pkl')
loaded_target = joblib.load('Target_encoder.pkl')
loaded_scaler = joblib.load('StandardScaler.pkl')

# 1. Inject CSS targeting the specific key class
st.markdown("""
    <style>
    /* Target the button inside the container with class 'st-key-custom_submit' */
    .st-key-custom_submit button {
        background-color: #ADD8E6; /* SeaGreen */
        color: #000000;            /* White text */
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    
    /* Optional: Hover effect */
    .st-key-custom_submit button:hover {
        background-color: #87CEEB;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)


st.set_page_config(layout="wide")

if 'is_expanded' not in st.session_state:
    st.session_state.is_expanded = True
if 'expander_version' not in st.session_state:
    st.session_state.expander_version = 0

def collapse_expander():
    st.session_state.is_expanded = False
    st.session_state.expander_version += 1  # forces a brand-new widget identity

st.title('Car Valuation Tool', text_alignment = "center")

with st.expander(
    "Input data",
    expanded=st.session_state.is_expanded,
    key=f"expander_{st.session_state.expander_version}"
):
    with st.form("my_form"):
        Car_brand = st.selectbox("Enter Car brand", ['Acura','Audi','BMW','Bentley','Buick','Cadillac','Chevrolet','Chrysler','Dodge','Ford','GMC','Honda','Hyundai','INFINITI','Jaguar','Jeep','Kia','Land','Lexus','Lincoln','MINI','Maserati','Mazda','Mercedes-Benz','Mitsubishi','Nissan','Porsche','RAM','Subaru','Toyota','Volkswagen','Volvo'], index = None, placeholder="Honda")
        year = st.number_input("Enter Year", min_value=1990, max_value=2026)
        hp = st.number_input("Enter Car HorsePower", min_value=10)
        milage = st.number_input("Enter Car milage", min_value=0.0)
        eng_size = st.number_input("Enter Engine size (Liters)", min_value=1.0)
        cylinders = st.number_input("Enter Cylinders number", min_value=1)
        fuel = st.selectbox("Enter Fuel-type", ['Diesel','Gasoline', 'Hybrid', 'Other_fuel'], index = None, placeholder="Select fuel...")
        gearbox = st.selectbox("Enter Gearbox", ['Automatic', 'CVT', 'Manual','Other_transmission'], index = None, placeholder="Select gearbox...")
        ext_color = st.selectbox("Enter Exterior color", ['Beige', 'Black', 'Blue', 'Brown','Gold', 'Gray', 'Green', 'Orange', 'Red','Silver', 'White', 'Yellow', 'other'], index = None, placeholder="Select color...")
        int_color = st.selectbox("Enter interior color", ['Beige','Black', 'Blue', 'Brown', 'Gray', 'Jet Black','Red', 'White', 'other'], index = None, placeholder="Select color...")
        accident = st.selectbox("Enter accidents", ['At least 1 accident or damage reported', 'None reported'], index = None, placeholder="Select accident...")

        submitted = st.form_submit_button("Submit Info", key="custom_submit", on_click=collapse_expander)

    st.markdown("NOTE, pressing **Enter** will submit the form. Please do not press **Enter** when introducing the text in the boxes")

if submitted:
    if Car_brand != None and year != 1990 and milage != 0.0 and hp != 10 and eng_size != 1.0 and cylinders != 1 and fuel != None and gearbox != None and ext_color != None and int_color != None and accident != None:
        st.success(f"Submitted:  Car = {Car_brand},  Year = {year},  HorsePower = {hp},  Milage = {milage},  Engine = {eng_size}L,  Cylinders = {cylinders},  Fuel = {fuel},  Gearbox = {gearbox},  Exterior = {ext_color},  Interior = {int_color},  Accidents = {accident}")
        st.divider()  # Draws a horizontal rule
        st.header("Actual car value (2026)")
        
        #Obtaining new row for predictions:
        new_list = [float(year), float(milage), float(hp), float(eng_size), float(cylinders)]
        elements_list = [['Diesel','Gasoline', 'Hybrid', 'Other_fuel'],['Automatic', 'CVT', 'Manual','Other_transmission'],['Beige', 'Black', 'Blue', 'Brown','Gold', 'Gray', 'Green', 'Orange', 'Red','Silver', 'White', 'Yellow', 'other'],['Beige','Black', 'Blue', 'Brown', 'Gray', 'Jet Black','Red', 'White', '–', 'other'],['At least 1 accident or damage reported', 'None reported']]
        target_dumm_list = [fuel, gearbox, ext_color, int_color, accident]
        for ele, tat in zip(elements_list, target_dumm_list):
            ele_idx = ele.index(tat)
            dumm_ele = [0.0]*len(ele); dumm_ele[ele_idx] = 1.0
            new_list.extend(dumm_ele)
        enco_brand_df=pd.DataFrame([Car_brand],columns = ["brand"])
        enco_brand = loaded_target.transform(enco_brand_df["brand"])
        new_list.append(enco_brand.values[0][0])
        
        #scaling
        assert(39 == len(new_list))
        new_list_scaled = pd.DataFrame(new_list).transpose()
        new_list_scaled.columns = ['model_year', 'milage', 'HP', 'Engine_Size', 'Cylinders', 'Diesel','Gasoline', 'Hybrid', 'Other_fuel', 'Automatic', 'CVT', 'Manual','Other_transmission', 'ext_Beige', 'ext_Black', 'ext_Blue', 'ext_Brown','ext_Gold', 'ext_Gray', 'ext_Green', 'ext_Orange', 'ext_Red','ext_Silver', 'ext_White', 'ext_Yellow', 'other_ext_col', 'int_Beige','int_Black', 'int_Blue', 'int_Brown', 'int_Gray', 'int_Jet Black','int_Red', 'int_White', 'int_–', 'other_int_col','At least 1 accident or damage reported', 'None reported','encoded_brand']
        cols_to_scale = list(loaded_scaler.feature_names_in_)
        new_list_scaled[cols_to_scale] = loaded_scaler.transform(new_list_scaled[cols_to_scale])
        
        #predicting new value
        y_pred = loaded_model.predict(new_list_scaled)
        final_value = max(0,round(np.expm1(y_pred)[0],2))
        
        col1, col2, col3 = st.columns(3)
        with col2:
            if final_value <= 10000:
                st.markdown(f'''
                    # :red[{final_value:,.2f} €]
                ''', text_alignment = "center")
            elif 10000 < final_value <= 20000:
                st.markdown(f'''
                    # :yellow[{final_value:,.2f} €]
                ''', text_alignment = "center")
            else:
                st.markdown(f'''
                    # :green[{final_value:,.2f} €]
                ''', text_alignment = "center")
            estim_left = final_value - final_value*0.1
            estim_right = final_value + final_value*0.1 
            st.markdown(f"## :blue-background[{estim_left:,.2f} € to {estim_right:,.2f} €]", text_alignment = "center")
        with col3:
            st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; gap: 14px; padding-top: 2rem;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 24px; height: 24px; background-color: red; border-radius: 4px;"></div>
                <span>Less than 10.000 €</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 24px; height: 24px; background-color: gold; border-radius: 4px;"></div>
                <span>Between 10.000 and 20.000 €</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 24px; height: 24px; background-color: green; border-radius: 4px;"></div>
                <span>More than 20.000 €</span>
            </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Other cars arround this price: ")
        exposed_df = last_df[last_df["price"] < estim_right]
        df_shuffled = exposed_df.sample(frac=1).reset_index(drop=True)   
        st.dataframe(df_shuffled[df_shuffled["price"] > estim_left].head(5))
        st.divider()  # Draws a horizontal rule
        
        st.markdown("## <u>Pricing Plots<u>", unsafe_allow_html=True, text_alignment = "center")
        col1, col2 = st.columns(2)
        
        with col1:
            new_list_plot = pd.DataFrame(new_list).transpose()
            new_list_plot.columns = ['model_year', 'milage', 'HP', 'Engine_Size', 'Cylinders', 'Diesel','Gasoline', 'Hybrid', 'Other_fuel', 'Automatic', 'CVT', 'Manual','Other_transmission', 'ext_Beige', 'ext_Black', 'ext_Blue', 'ext_Brown','ext_Gold', 'ext_Gray', 'ext_Green', 'ext_Orange', 'ext_Red','ext_Silver', 'ext_White', 'ext_Yellow', 'other_ext_col', 'int_Beige','int_Black', 'int_Blue', 'int_Brown', 'int_Gray', 'int_Jet Black','int_Red', 'int_White', 'int_–', 'other_int_col','At least 1 accident or damage reported', 'None reported','encoded_brand']
            #st.markdown("Price vs year with query", text_alignment = "center")
            fig, ax = plt.subplots()

            ax.scatter(log_price_df[log_price_df["brand"] == Car_brand]["model_year"], log_price_df[log_price_df["brand"] == Car_brand]["price"], c="blue", label=f'All {Car_brand}')
            ax.scatter(new_list_plot["model_year"], y_pred, color="red", label=f'Submitted {Car_brand}')
            
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            ax.legend()

            ax.set_xlabel("Model Year")
            ax.set_ylabel("np.log(Price)")
            ax.set_title(f"Price vs Year — {Car_brand}")
            
            # Display in Streamlit
            st.pyplot(fig, use_container_width=True)

            
        with col2:
            #st.markdown("Importance of each feature in prediction", text_alignment = "center")
            fig2, ax2 = plt.subplots()
            
            ax2.bar(sorted_coef_dict.keys(), sorted_coef_dict.values())
            
            ax2.tick_params(axis='x', rotation=45)
            ax2.set_xlabel("Column")
            ax2.set_ylabel("Model coef. value")
            ax2.set_title(f"Estimated contribution per column")
            
            # Display in Streamlit
            st.pyplot(fig2, use_container_width=True)
            

    else:
        st.error("Error, all boxes need to be properly filled out")
        st.divider()  # Draws a horizontal rule

