import pandas as pd
import streamlit as st
import requests
import sqlite3
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from dashboard_functions import display_credit_result, plot_gauge, display_feature_importance, get_global_feature_importance, predict_credit_risk

model_threshold = .5

# streamlit run dashboard.py

st.title('Credit risk prediction')
st.markdown('Get credit risk prediction for a client based on his/her ID')


tab1, tab2, tab3 = st.tabs([':clipboard: Credit risk prediction', ':bar_chart: Feature importance', ':chart_with_upwards_trend: Client informations'])

if 'client_id' not in st.session_state:
    st.session_state['client_id'] = None

if 'feature_importance' not in st.session_state:
    st.session_state['feature_importance'] = get_global_feature_importance()

client_id = st.sidebar.number_input(
    label='Client ID', min_value=0, max_value=1000000, value=None,
    step=1, format='%i', placeholder='Enter client ID'    
)
submit_id = st.sidebar.button('Submit client ID', key='submit_id')
if submit_id:
    st.session_state['client_id'] = client_id

with tab1:
    st.write('Current client ID:', st.session_state['client_id'])

    if st.session_state['client_id'] is not None:
        predict_credit_risk(client_id=client_id)
    else:
        st.write('Please enter a client ID in the sidebar section.')

with tab2:
    st.markdown('## Feature importance')
    st.write('Current client ID:', st.session_state['client_id'])

    feature_scale = st.radio('Select feature importance scale', ['Global', 'Local'], index=0, horizontal=True)

    if feature_scale == 'Global':
        st.markdown('### Global feature importance')

        model_type = st.session_state['feature_importance']['model_type']

        if st.session_state['feature_importance']['model_type'] == 'XGBClassifier':
            importance_type = st.radio("Select importance type:", ["weight", "cover", "gain"], index=0, horizontal=True)
        elif st.session_state['feature_importance']['model_type'] == 'RandomForestClassifier':
            pass

        nb_features = st.number_input(
            label='Features nb', min_value=0, max_value=30, value=20,
            step=1, format='%i', placeholder='Enter number of features to display'
        )

        display_feature_importance(model_type, nb_features, importance_type)
    else:
        st.markdown('### Local feature importance')
        if 'client_id' not in st.session_state:
            st.write('Please select a client ID in the sidebar section.')

with tab3:
    st.markdown('## Client infos')
    st.write('Current client ID:', st.session_state['client_id'])


