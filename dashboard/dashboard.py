import pandas as pd
import streamlit as st
from dashboard_functions import toggle_debug_mode
from dashboard_functions import submit_client_id, predict_credit_risk
from dashboard_functions import display_built_in_global_feature_importance, get_built_in_global_feature_importance
from dashboard_functions import initiate_shap_explainer, display_shap_feature_importance
from dashboard_functions import fetch_cat_and_split_features, fetch_violinplot_data, display_violinplot

model_threshold = .5

# streamlit run dashboard.py

st.title('Credit risk prediction')
st.markdown('Get credit risk prediction for a client based on his/her ID')

if 'debug_mode' not in st.session_state:
    st.session_state['debug_mode'] = True

if 'client_id' not in st.session_state:
    st.session_state['client_id'] = None

if 'feature_importance' not in st.session_state:
    st.session_state['feature_importance'] = get_built_in_global_feature_importance()

# if 'shap_explainer_initiated' not in st.session_state:
#     st.session_state['shap_explainer_initiated'] = 'Not initiated'

if 'shap' not in st.session_state:
    st.session_state['shap'] = {
        'initiated': False,
        'Global': {
            'loaded': False,
            'features': None,
            'shap_values': None,
            'feature_names': None,
            'expected_value': None
        },
        'Local': {
            'loaded': False,
            'features': None,
            'shap_values': None,
            'feature_names': None,
            'expected_value': None,
            'client_id': None
        }
    }


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

toggle_debug = st.sidebar.toggle('Debug mode', value=True, on_change=toggle_debug_mode, key='debug')

client_id = st.sidebar.number_input(
    label='Client ID', min_value=0, max_value=1000000, value=None,
    step=1, format='%i', placeholder='Enter client ID'    
)

submit_id = st.sidebar.button('Submit client ID', on_click=submit_client_id(client_id), key='submit_id')


shap_initiation = st.sidebar.button('Initiate Shap explainer', key='initiate_shap_explainer', on_click=initiate_shap_explainer, disabled=st.session_state['shap']['initiated'] == True)

tab1, tab2, tab3, tab4 = st.tabs([':clipboard: Credit risk prediction', ':bar_chart: Feature importance', ':chart_with_upwards_trend: Client comparison', ':wrench: debug'])

with tab1:
    st.write('Current client ID:', st.session_state['client_id'])
    st.header('Credit risk prediction', divider='red')

    if st.session_state['client_id'] is not None:
        predict = st.button('Predict credit risk', key='predict')
        if predict:
            predict_credit_risk(client_id=st.session_state['client_id'], debug=st.session_state['debug_mode'])      
    else:
        st.warning('Please enter a client ID in the sidebar section.')

with tab2:
    st.write('Current client ID:', st.session_state['client_id'])
    st.header('Feature importance', divider='red')
    st.caption('Select an explainer type to show feature importance. You can chose the number of '
               'features to display and, if a compatible explainer is selected, the scale of the '
               'feature importance (Global or Local).')
    st.caption('Feature importance shows the relative contributions of features in the model ouput'
               ' (here, the probability of credit risk). Global scale shows the feature importance'
               ' at the scale of the whole model, while Local scale shows the contributions of '
               ' features in the model output for the current client only.')

    with st.container(border=True):
        st.subheader('Display settings', divider='red')
        explainer = st.radio('Select explainer type', available_explainer_types, index=0, horizontal=True)
        nb_features = st.slider(
            label='Features to display', min_value=1, max_value=len(st.session_state['feature_importance']['feature_importance'][available_importance_types[0]]), value=20,
            step=1, format='%i'
        )

    if explainer == 'Built-in':
        with st.container(border=True):
            st.subheader('Built-in feature importance', divider='red')

            model_type = st.session_state['feature_importance']['model_type']

            importance_type = st.radio("Select importance type:", available_importance_types, index=0, horizontal=True)
            display_built_in_global_feature_importance(model_type, nb_features, importance_type)

    elif explainer == 'Shap':
        with st.container(border=True):
            st.subheader('Shap feature importance', divider='red')

            if st.session_state['shap']['initiated'] == False:
                st.warning('Please initiate Shap explainer in the sidebar section.')
            else:
                feature_scale = st.radio('Select shap feature importance scale', importance_scale, index=0, horizontal=True)

                if feature_scale == 'Global':
                    display_shap_feature_importance(client_id=st.session_state['client_id'], scale=feature_scale, nb_features=nb_features, debug=st.session_state['debug_mode'])

                elif feature_scale == 'Local':

                    if st.session_state['client_id'] is None:
                        st.warning('Please enter a client ID in the sidebar section.')

                    else:
                        display_shap_feature_importance(client_id=st.session_state['client_id'], scale=feature_scale, debug=st.session_state['debug_mode'])
                

with tab3:
    st.write('Current client ID:', st.session_state['client_id'])
    st.header('Client informations', divider='red')

    if st.session_state['client_id'] is None:
        st.warning('Please enter a client ID in the sidebar section.')
        selected_global_feature, selected_categorical_feature, selected_split_feature = None, None, None
    else:
        importance_type = st.radio("Order available feature by feature importance type:", available_importance_types, index=0, horizontal=True)
        global_features = [key for key, _ in sorted(st.session_state['feature_importance']['feature_importance'][importance_type].items(), key=lambda item: item[1], reverse=True)]
        categorical_features, split_features = fetch_cat_and_split_features(global_features, debug=st.session_state['debug_mode'])

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

        fetch_violinplot_data(st.session_state['client_comparison']['global'], st.session_state['client_comparison']['categorical'], st.session_state['client_comparison']['split'], debug=st.session_state['debug_mode'])
        # display_violin_plot(df, client_id=st.session_state['client_id'])
        show_data = st.expander('Show selected data')
        show_data.write(st.session_state['client_comparison']['data'])
        display_violinplot(st.session_state['client_comparison']['data'], st.session_state['client_id'])

with tab4:
    st.write('Current client ID:', st.session_state['client_id'])
    st.header('Debug', divider='red')


    st.subheader('Current debug: None', divider='red')