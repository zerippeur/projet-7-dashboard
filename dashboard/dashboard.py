import pandas as pd
import streamlit as st
import requests
import sqlite3
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from dashboard_functions import display_credit_result, plot_gauge, predict_credit_risk, get_client_infos
from dashboard_functions import display_built_in_global_feature_importance, get_built_in_global_feature_importance
from dashboard_functions import initiate_shap_explainer, get_shap_feature_importance, display_shap_feature_importance
from dashboard_functions import fetch_data, fetch_categorical_features, display_histogram_chart, fetch_split_features
from streamlit_shap import st_shap
import numpy as np

model_threshold = .5

# streamlit run dashboard.py

st.title('Credit risk prediction')
st.markdown('Get credit risk prediction for a client based on his/her ID')


tab1, tab2, tab3, tab4 = st.tabs([':clipboard: Credit risk prediction', ':bar_chart: Feature importance', ':chart_with_upwards_trend: Client comparison', ':wrench: debug'])

if 'client_id' not in st.session_state:
    st.session_state['client_id'] = None

if 'feature_importance' not in st.session_state:
    st.session_state['feature_importance'] = get_built_in_global_feature_importance()

available_importance_types = list(st.session_state['feature_importance']['feature_importance'].keys())

if 'shap_explainer_initiated' not in st.session_state:
    st.session_state['shap_explainer_initiated'] = 'Not initiated'

if 'shap' not in st.session_state:
    st.session_state['shap'] = {
        'Global': {
            'initiated': False,
            'data': None
        },
        'Local': {
            'initiated': False,
            'data': None,
            'client_id': None
        }
    }

importance_scale = ['Global', 'Local']
available_explainer_types = ['Built-in', 'Shap']

# if 'available_features' not in st.session_state:
#     st.session_state['available_features'] = fetch_feature_and_group_values()

client_id = st.sidebar.number_input(
    label='Client ID', min_value=0, max_value=1000000, value=None,
    step=1, format='%i', placeholder='Enter client ID'    
)
submit_id = st.sidebar.button('Submit client ID', key='submit_id')
if submit_id:
    st.session_state['client_id'] = client_id


shap_initiation = st.sidebar.button('Initiate Shap explainer', key='initiate_shap_explainer', disabled=st.session_state['shap_explainer_initiated'] == 'Initiated')
if shap_initiation:
    initiate_shap_explainer()
    st.session_state['shap_explainer_initiated'] = 'Initiated'
    # refresh app
    st.rerun()

with tab1:
    st.write('Current client ID:', st.session_state['client_id'])

    if st.session_state['client_id'] is not None:
        predict = st.button('Predict credit risk', key='predict')
        if predict:
            predict_credit_risk(client_id=st.session_state['client_id'])      
    else:
        st.write('Please enter a client ID in the sidebar section.')

with tab2:
    st.markdown('## Feature importance')
    st.write('Current client ID:', st.session_state['client_id'])

    explainer = st.radio('Select explainer', available_explainer_types, index=None, horizontal=True)

    if explainer == 'Built-in':
        st.markdown('### Built-in feature importance')

        model_type = st.session_state['feature_importance']['model_type']

        if st.session_state['feature_importance']['model_type'] == 'XGBClassifier':
            importance_type = st.radio("Select importance type:", available_importance_types, index=0, horizontal=True)
        elif st.session_state['feature_importance']['model_type'] == 'RandomForestClassifier':
            pass

        nb_features = st.number_input(
            label='Features nb', min_value=0, max_value=30, value=20,
            step=1, format='%i', placeholder='Enter number of features to display'
        )

        display_built_in_global_feature_importance(model_type, nb_features, importance_type)
    elif explainer == 'Shap':
        st.markdown('### Shap feature importance')

        if st.session_state.shap_explainer_initiated == 'Not initiated':
            st.write('Please initiate Shap explainer in the sidebar section.')

        if st.session_state.shap_explainer_initiated == 'Initiated':
            feature_scale = st.radio('Select shap feature importance scale', importance_scale, index=0, horizontal=True)
            if feature_scale == 'Global':
                nb_features = st.number_input(
                    label='Features nb', min_value=0, max_value=30, value=20,
                    step=1, format='%i', placeholder='Enter number of features to display'
                )
                display_shap_feature_importance(client_id=st.session_state['client_id'], scale=feature_scale, nb_features=nb_features)
            elif feature_scale == 'Local':
                if st.session_state['client_id'] is None:
                    st.write('Please enter a client ID in the sidebar section.')
                else:
                    display_shap_feature_importance(client_id=st.session_state['client_id'], scale=feature_scale)
                

with tab3:
    st.markdown('## Client informations')
    st.write('Current client ID:', st.session_state['client_id'])

    if st.session_state['client_id'] is None:
        st.write('Please enter a client ID in the sidebar section.')
    else:
        importance_type = st.radio("Order available feature by feature importance type:", available_importance_types, index=0, horizontal=True)
        st.session_state['available_features'] = {
            'global_features': [key for key, _ in sorted(st.session_state['feature_importance']['feature_importance'][importance_type].items(), key=lambda item: item[1], reverse=True)],
            'categorical_features': fetch_categorical_features(st.session_state['feature_importance']['feature_importance'][importance_type])
        }

        selected_global_feature = st.selectbox('Select Global Feature (Int/Float)', st.session_state['available_features']['global_features'])
        selected_categorical_feature = st.selectbox('Select Categorical Feature for Grouping', [''] + st.session_state['available_features']['categorical_features'])
        
        # Fetch data based on user input
        df, grouped_data, group_values = fetch_data(selected_global_feature, selected_categorical_feature)

        selected_aggregation = st.checkbox('Use Median instead of Mean')
        selected_client = st.session_state['client_id']

        draw_comparison_chart = st.button('Draw client comparison chart', key='client_comparison')
        if draw_comparison_chart:
            display_histogram_chart(df, selected_global_feature, grouped_data, group_values, client_id, selected_aggregation, selected_categorical_feature)

with tab4:
    st.markdown('## Debug')
    st.write('Current client ID:', st.session_state['client_id'])

    st.markdown('### No active debug')

    if st.session_state['client_id'] is None:
        st.write('Please enter a client ID in the sidebar section.')
    else:
        importance_type = st.radio("Order available feature by feature importance type:", available_importance_types, index=0, horizontal=True, key='importance_type_debug')
        st.session_state['available_features'] = {
            'global_features': [key for key, _ in sorted(st.session_state['feature_importance']['feature_importance'][importance_type].items(), key=lambda item: item[1], reverse=True)],
            'categorical_features': fetch_categorical_features(st.session_state['feature_importance']['feature_importance'][importance_type]),
            'split_features': fetch_split_features(st.session_state['feature_importance']['feature_importance'][importance_type])
        }
        st.write(st.session_state['available_features']['global_features'], st.session_state['available_features']['categorical_features'], st.session_state['available_features']['split_features'])