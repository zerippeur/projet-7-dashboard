import pandas as pd
import streamlit as st
import requests
import sqlite3
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from dashboard_functions import display_credit_result, plot_gauge, predict_credit_risk, get_client_infos
from dashboard_functions import display_built_in_global_feature_importance, get_built_in_global_feature_importance
from dashboard_functions import initiate_shap_explainer, get_shap_feature_importance, display_shap_feature_importance
from dashboard_functions import fetch_data, display_histogram_chart, fetch_cat_and_split_features, fetch_violinplot_data, display_violinplot, update_selected_features, interactive_plot_test
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

# if 'disable_feature_violinplot' not in st.session_state:
#     st.session_state['disable_feature_violinplot'] = {
#         'secondary': True,
#         'tertiary': True
#     }

if 'client_comparison' not in st.session_state:
    st.session_state['client_comparison'] = {
        'data': pd.DataFrame()
    }

if 'selected_global_feature' not in st.session_state['client_comparison']:
    st.session_state['client_comparison']['global'] = None
if 'selected_categorical_feature' not in st.session_state['client_comparison']:
    st.session_state['client_comparison']['categorical'] = None
if 'selected_split_feature' not in st.session_state['client_comparison']:
    st.session_state['client_comparison']['split'] = None

available_importance_types = list(st.session_state['feature_importance']['feature_importance'].keys())
importance_scale = ['Global', 'Local']
available_explainer_types = ['Built-in', 'Shap']

client_id = st.sidebar.number_input(
    label='Client ID', min_value=0, max_value=1000000, value=None,
    step=1, format='%i', placeholder='Enter client ID'    
)

def submit_client_id(client_id: int):
    st.session_state['client_id'] = client_id
submit_id = st.sidebar.button('Submit client ID', on_click=submit_client_id(client_id), key='submit_id')


shap_initiation = st.sidebar.button('Initiate Shap explainer', key='initiate_shap_explainer', on_click=initiate_shap_explainer, disabled=st.session_state['shap_explainer_initiated'] == 'Initiated')
# if shap_initiation:
#     initiate_shap_explainer()
#     st.session_state['shap_explainer_initiated'] = 'Initiated'
#     st.rerun()

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

        importance_type = st.radio("Select importance type:", available_importance_types, index=0, horizontal=True)

        nb_features = st.number_input(
            label='Features nb', min_value=0, max_value=30, value=20,
            step=1, format='%i', placeholder='Enter number of features to display'
        )

        display_built_in_global_feature_importance(model_type, nb_features, importance_type)
    elif explainer == 'Shap':
        st.markdown('### Shap feature importance')

        if st.session_state['shap_explainer_initiated'] == 'Not initiated':
            st.write('Please initiate Shap explainer in the sidebar section.')
        else:
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
        selected_global_feature, selected_categorical_feature, selected_split_feature = None, None, None
    else:
        importance_type = st.radio("Order available feature by feature importance type:", available_importance_types, index=0, horizontal=True)
        global_features = [key for key, _ in sorted(st.session_state['feature_importance']['feature_importance'][importance_type].items(), key=lambda item: item[1], reverse=True)]
        categorical_features, split_features = fetch_cat_and_split_features(global_features)

        st.session_state['available_features'] = {
            'global_features': global_features,
            'categorical_features': categorical_features,
            'split_features': split_features
        }

        
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
        col1.subheader('Global', divider='red')
        col1.caption('Main feature for global comparison')
        col1.caption('y-axis scale - Mandatory')
        col2.subheader('Categories', divider='red')
        col2.caption('Second feature for categories')
        col2.caption('x-axis groups - Optionnal')
        col3.subheader('Splits', divider='red')
        col3.caption('Third feature for categories split')
        col3.caption('violin sides - Optionnal')
 
        st.session_state['client_comparison']['global'] = col1.selectbox('GLOBAL FEATURE', [None] + global_features, index=st.session_state['client_comparison']['global'])

        if st.session_state['client_comparison']['global'] is None:
            st.session_state['client_comparison']['categorical'] = None
            st.session_state['client_comparison']['split'] = None
        else:
            st.session_state['client_comparison']['categorical'] = col2.selectbox('CATEGORICAL FEATURE', [None] + categorical_features, index=st.session_state['client_comparison']['categorical'])
            if st.session_state['client_comparison']['categorical'] is None:
                st.session_state['client_comparison']['split'] = None
            else:
                st.session_state['client_comparison']['split'] = col3.selectbox('SPLIT FEATURE', [None] + [split_feature for split_feature in split_features if split_feature != st.session_state['client_comparison']['categorical']], index=st.session_state['client_comparison']['split'])

        fetch_violinplot_data(st.session_state['client_comparison']['global'], st.session_state['client_comparison']['categorical'], st.session_state['client_comparison']['split'])
        # display_violin_plot(df, client_id=st.session_state['client_id'])
        show_data = st.expander('Show selected data')
        show_data.write(st.session_state['client_comparison']['data'])
        display_violinplot(st.session_state['client_comparison']['data'], st.session_state['client_id'])

with tab4:
    st.markdown('## Debug')
    st.write('Current client ID:', st.session_state['client_id'])

    # st.markdown('### No active debug')

    interactive_plot_test()