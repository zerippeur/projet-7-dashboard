# Third-party imports
import pandas as pd
import streamlit as st

# Local imports
from dashboard_functions import get_model_threshold
from dashboard_functions import predict_credit_risk
from dashboard_functions import get_built_in_global_feature_importance
from dashboard_functions import display_built_in_global_feature_importance
from dashboard_functions import initiate_shap_explainer, display_shap_feature_importance
from dashboard_functions import update_available_features, update_violinplot_data
from dashboard_functions import display_violinplot, fetch_cat_and_split_features

# streamlit run dashboard.py
st.set_page_config(layout="wide")
st.title(body='Credit risk prediction app')

tab1, tab2, tab3, tab4 = st.tabs(
    tabs=[
        ':computer: Credit risk prediction',
        ':bar_chart: Feature importance',
        ':mag: Client comparison',
        ':clipboard: About'
    ]
)

if 'client_id' not in st.session_state:
    st.session_state['client_id'] = None

if 'feature_importance' not in st.session_state:
    st.session_state['feature_importance'] = get_built_in_global_feature_importance()

if 'available_features' not in st.session_state:
    imp_dict = st.session_state['feature_importance']['feature_importance']
    first_key = next(iter(imp_dict), None)
    st.session_state['available_features'] = {
        'initiated': False,
        'global_features': [
            key for key, _
            in sorted(
                st.session_state['feature_importance']['feature_importance'][first_key].items(),
                key=lambda item: item[1],
                reverse=True
            )
        ],
        'categorical_features': None,
        'split_features': None
    }

if 'threshold' not in st.session_state:
    st.session_state['threshold'] = get_model_threshold()

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
        'data': pd.DataFrame(),
        'global': None,
        'categorical': None,
        'split': None
    }

imp_dict = st.session_state['feature_importance']['feature_importance']
available_importance_types = list(imp_dict.keys())
importance_scale = ['Global', 'Local']
available_explainer_types = ['Built-in', 'Shap']

st.session_state['client_id'] = st.sidebar.number_input(
    label='Client ID',
    min_value=0,
    max_value=1000000,
    value=None,
    step=1,
    format='%i',
    placeholder='Enter client ID'
)

st.sidebar.caption(
    body='Client ID use cases:\n - 108567: SAFE (Credit approved)\n'
         ' - 110084: NOPE (Credit rejected)',
    unsafe_allow_html=True
)

shap_initiation = st.sidebar.button(
    label='Initiate Shap explainer',
    key='initiate_shap_explainer',
    on_click=initiate_shap_explainer,
    disabled=st.session_state['shap']['initiated'],
    type='primary'
)

available_feature_initiation = st.sidebar.button(
    label='Initiate available features',
    key='update_available_features',
    on_click=fetch_cat_and_split_features,
    disabled=st.session_state['available_features']['initiated'],
    type='primary'
)

with tab1:
    st.write('Current client ID:', st.session_state['client_id'])
    st.header('Credit risk prediction', divider='red')
    st.caption(
        body='Computes credit risk prediction for the current client ID.'
    )
    st.caption(
        body='The resulting graph shows the probability of repaying the loan without any risk. It '
             'also indicates the threshold at which the decision changes between approved (blue) '
             'and rejected (red) credit applications. Finally, we indicate a range within which '
             'the credit application needs further investigation before approval or rejection.'
    )
    if st.session_state['client_id'] is not None:
        with st.container(border=True):
            st.subheader(body='Gauge graph', divider='red')
            predict = st.button('Predict credit risk', key='predict')
            if predict:
                predict_credit_risk(
                    client_id=st.session_state['client_id'],
                    threshold=st.session_state['threshold']
                )
    else:
        st.warning(body='Please enter a client ID in the sidebar section.', icon="⚠️")

with tab2:
    st.write('Current client ID:', st.session_state['client_id'])
    st.header(body='Feature importance', divider='red')
    st.caption(
        body='Select an explainer type to show feature importance. You can chose the number of '
             'features to display and, if a compatible explainer is selected, the scale of the '
             'feature importance (Global or Local).'
    )
    st.caption(
        body='Feature importance shows the relative contributions of features in the model ouput'
             ' (here, the probability of credit risk). Global scale shows the feature importance'
             ' at the scale of the whole model, while Local scale shows the contributions of '
             ' features in the model output for the current client only.'
    )

    with st.container(border=True):
        st.subheader(body='Display settings', divider='red')
        explainer = st.radio(
            label='Select explainer type',
            options=available_explainer_types,
            index=0,
            horizontal=True
        )
        nb_features = st.slider(
            label='Features to display',
            min_value=1,
            max_value=len(imp_dict[available_importance_types[0]]),
            value=20,
            step=1,
            format='%i'
        )

    with st.container(border=True):
        if explainer == 'Built-in':
            st.subheader(body='Built-in feature importance', divider='red')

            model_type = st.session_state['feature_importance']['model_type']

            st.session_state['tab_2_selected_importance_type'] = st.radio(
                label="Select importance type:",
                options=available_importance_types,
                index=0,
                horizontal=True
            )
            st.subheader(
                body='Global feature importance graph',
                divider='grey'
            )
            display_built_in_global_feature_importance(
                model_type=model_type,
                nb_features=nb_features,
                importance_type=st.session_state['tab_2_selected_importance_type']
            )
        elif explainer == 'Shap':
            st.subheader(body='Shap feature importance', divider='red')

            if not st.session_state['shap']['initiated']:
                st.warning(
                    body='Please initiate Shap explainer in the sidebar section.',
                    icon="⚠️"
                )
            else:
                feature_scale = st.radio(
                    label='Select shap feature importance scale',
                    options=importance_scale,
                    index=0,
                    horizontal=True
                )
                st.subheader(
                    body=f'Shap {str.lower(feature_scale)} feature importance graph',
                    divider='grey'
                )
                if feature_scale == 'Global':
                    display_shap_feature_importance(
                        client_id=st.session_state['client_id'],
                        scale=feature_scale,
                        nb_features=nb_features
                    )

                elif feature_scale == 'Local':

                    if st.session_state['client_id'] is None:
                        st.warning(
                            body='Please enter a client ID in the sidebar section.',
                            icon="⚠️"
                        )

                    else:
                        display_shap_feature_importance(
                            client_id=st.session_state['client_id'],
                            scale=feature_scale
                        )
                

with tab3:
    st.write('Current client ID:', st.session_state['client_id'])
    st.header(body='Client comparison', divider='red')
    st.caption(
        body='Shows current client position among the distribution of other randomly sampled '
             'clients for up to three selected features ranked in descending order of global '
             'importance.'
    )
    st.caption(
        body='There are three types of features: Global (any feature), Category (categorical '
             'features with less than 8 categories) and Splits (binary feature with 2 categories).'
             ' Global feature is displayed on the y-axis. Category is displayed on the x-axis. '
             'Split is displayed as violin sides.'
    )

    if st.session_state['client_id'] is None:
        st.warning(
            body='Please enter a client ID in the sidebar section.',
            icon="⚠️"
        )
    if not st.session_state['available_features']['initiated']:
        st.warning(
            body='Please initiate available features in the sidebar section.',
            icon="⚠️"
        )
    if (
        st.session_state['client_id'] is not None
        and st.session_state['available_features']['initiated']
    ):
        with st.container(border=True):
            st.session_state['tab_3_selected_importance_type'] = st.radio(
                label="Order available features by feature importance type:",
                options=available_importance_types,
                index=0,
                horizontal=True,
                on_change=update_available_features
            )
        
        with st.container(border=True):
            st.subheader(body='Features selectors', divider='red')
            col1, col2, col3 = st.columns(3)
        col1.subheader(body='Global feature', divider='grey')
        col1.caption(body='Main feature for global comparison')
        col1.caption(body='*y-axis scale - Mandatory*')

        col2.subheader(body='Category feature', divider='grey')
        col2.caption(body='Second feature for category')
        col2.caption(body='*x-axis groups - Optionnal*')

        col3.subheader(body='Split feature', divider='grey')
        col3.caption(body='Third feature for categories split')
        col3.caption(body='*violin sides - Optionnal*')
 
        update_available_features()

        st.session_state['client_comparison']['global'] = col1.selectbox(
            label='GLOBAL FEATURE',
            options=[None] + st.session_state['available_features']['global_features'],
            index=0,
            on_change=update_available_features
        )

        if st.session_state['client_comparison']['global'] is None:
            st.session_state['client_comparison']['categorical'] = None
            st.session_state['client_comparison']['split'] = None
        else:
            st.session_state['client_comparison']['categorical'] = col2.selectbox(
                label='CATEGORICAL FEATURE',
                options=[None] + [
                    categorical_feature for categorical_feature
                    in st.session_state['available_features']['categorical_features']
                    if categorical_feature != st.session_state['client_comparison']['global']
                ],
                index=0,
                on_change=update_available_features
            )
            if st.session_state['client_comparison']['categorical'] is None:
                st.session_state['client_comparison']['split'] = None
            else:
                st.session_state['client_comparison']['split'] = col3.selectbox(
                    label='SPLIT FEATURE',
                    options=[None] + [
                        split_feature for split_feature
                        in st.session_state['available_features']['split_features']
                        if split_feature != st.session_state['client_comparison']['categorical']
                    ],
                    index=0,
                    on_change=update_available_features
                )

        update_violinplot_data()
        show_data = st.expander(label='Show selected data')
        show_data.write(st.session_state['client_comparison']['data'])
        display_violinplot()

with tab4:
    st.write('Current client ID:', st.session_state['client_id'])
    st.header(body='Project summary', divider='red')
    st.caption(
        body='This is an OpenClassrooms project. The goal of this project is to predict the risk '
             'for a specific client to experience difficulties repaying his credit.'
    )
    with st.container(border=True):
        st.subheader(body='Data sources', divider='red')
        st.caption(
            body='This project relies on the *"Home Credit Default Risk"* kaggle project: '
                 '<p><a href='
                 '"https://www.kaggle.com/competitions/home-credit-default-risk/overview"'
                 ' target="_blank">Project overview</a></p>'
                 'We used the following code from a competitor as a reference for the data '
                 'engineering process: '
                 '<p><a href='
                 '"https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features"'
                 ' target="_blank">Reference project</a></p>',
            unsafe_allow_html=True
        )
    with st.container(border=True):
        st.subheader(body='Dashboard overview', divider='red')
        with st.container():
            st.subheader(body='Sidebar section', divider='grey')
            col1, col2, col3 = st.columns(3)
            col1.markdown(
                body='Client ID input'
            )
            col1.caption(
                body='    ID of the client to predict credit risk for.'
            )
            col2.markdown(
                body='Initiate Shap explainer'
            )
            col2.caption(
                body='    Send request to prediction api to initiate shap explainer and get global'
                    ' scale shap values (see tab 2).'
            )
            col3.markdown(
                body='Initiate available features'
            )
            col3.caption(
                body='    Send request to prediction api to initiate available features for client'
                    ' comparison graphs (see tab 3).'
            )
            with st.container():
                st.subheader(body='Tabs', divider='grey')
                col1, col2, col3 = st.columns(3)
                col1.subheader(body='Tab 1: Credit risk prediction')
                col1.caption(
                    body='Gauge graph showing the prediction for the current client ID.'
                )
                col2.subheader(body='Tab 2: Feature importance')
                col2.caption(
                    body='Built-in and shap feature importance. Global and local scales.'
                        'Initiate Shap explainer for shap feature importance (see sidebar section)'
                        '. Input client ID for local scale.'
                )
                col3.subheader(body='Tab 3: Client comparison')
                col3.caption(
                    body='Compare current client with other clients using violinplots (see tab 3 '
                         'for more details).'
            )