# Standard library imports
import os

# Third-party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import seaborn as sns
import shap
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from typing import Tuple, Literal, Union

# imblearn imports
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

# Local imports
from streamlit_shap import st_shap

# ENVIRONEMENT VARIABLES
# load_dotenv('dashboard.env')
HEROKU_DATABASE_URI = os.getenv("DATABASE_URI")
API_URI = os.getenv("API_URI")

# COMMON FUNCTIONS

# Function to submit client id
def submit_client_id(client_id: int):
    st.session_state['client_id'] = client_id

# Function to get client infos from id number 
def get_client_infos(client_id: int, output: Literal['dict', 'df'] = 'df', db_uri: str=HEROKU_DATABASE_URI)-> Union[dict, pd.DataFrame]:
    engine = create_engine(db_uri)

    table_names = ['train_df', 'test_df']

    query = text(f'SELECT * FROM {table_names[0]} WHERE "SK_ID_CURR" = :client_id UNION ALL SELECT * FROM {table_names[1]} WHERE "SK_ID_CURR" = :client_id ORDER BY "SK_ID_CURR"')
    with engine.connect() as conn:
        result = pd.read_sql_query(query, conn, params={'client_id': client_id}, index_col='SK_ID_CURR')

    if output == 'dict':
        model_dict = result.drop(columns=['level_0', 'index', 'TARGET']).to_dict(orient='index')
        client_infos = model_dict[client_id]
        return client_infos
    elif output == 'df':
        client_infos = result.drop(columns=['level_0', 'index', 'TARGET'])
        return client_infos

# Function to generate accessible palettes based on the number of groups to colorize
def generate_full_palette(num_groups):
    # Define custom colors
    custom_colors = ['MidnightBlue', 'Steelblue', 'Darkturquoise', 'Paleturquoise', 'Gold', 'Coral', 'Firebrick', 'Maroon']

    # If the number of groups is less than or equal to the length of custom_colors, use them directly
    if num_groups <= len(custom_colors):
        # Calculate the indices to select colors centered around 'Paleturquoise' and 'Gold'
        start_index = max(0, len(custom_colors) // 2 - num_groups // 2)
        end_index = min(len(custom_colors), start_index + num_groups)
        selected_colors = custom_colors[start_index:end_index]
    else:
        # If the number of groups is greater than the length of custom_colors, create a blend palette
        selected_colors = sns.blend_palette(custom_colors, num_groups)
    
    return selected_colors

def generate_palette_without_gold(num_groups):
    # Define custom colors without 'Gold'
    custom_colors = ['MidnightBlue', 'Steelblue', 'Darkturquoise', 'Paleturquoise', 'Coral', 'Firebrick', 'Maroon']

    # If the number of groups is less than or equal to the length of custom_colors, use them directly
    if num_groups <= len(custom_colors):
        # Calculate the indices to select colors centered around 'Paleturquoise'
        start_index = max(0, len(custom_colors) // 2 - (num_groups - 1) // 2)
        end_index = min(len(custom_colors), start_index + num_groups)
        selected_colors = custom_colors[start_index:end_index]
    else:
        # If the number of groups is greater than the length of custom_colors, create a blend palette
        selected_colors = sns.blend_palette(custom_colors, num_groups)
    
    return selected_colors

def generate_reduced_palette_without_gold(num_groups):
    # Define custom colors without 'Gold'
    custom_colors = ['MidnightBlue', 'Steelblue', 'Darkturquoise', 'Firebrick', 'Maroon']

    # If the number of groups is less than or equal to the length of custom_colors, use them directly
    if num_groups <= len(custom_colors):
        # Calculate the indices to select colors centered around 'Paleturquoise'
        start_index = max(0, len(custom_colors) // 2 - (num_groups - 1) // 2)
        end_index = min(len(custom_colors), start_index + num_groups)
        selected_colors = custom_colors[start_index:end_index]
    else:
        # If the number of groups is greater than the length of custom_colors, create a blend palette
        selected_colors = sns.blend_palette(custom_colors, num_groups)
    
    return selected_colors

# SIDEBAR FUNCTIONS

# Function to balance classes
def balance_classes(X: pd.DataFrame, y: pd.Series, method: Literal['smote', 'randomundersampler']='randomundersampler')-> Tuple[pd.DataFrame, pd.Series]:
    """
    Balance classes in the dataset using SMOTE or RandomUnderSampler.

    Args:
        X: Features. Accepts either a pandas DataFrame or a numpy array.
        y: Target variable. Accepts either a pandas Series or a numpy array.
        method: Method to use for balancing. Options: 'smote' or 'randomundersampler'.
            Defaults to 'smote'.

    Returns:
        Balanced feature set and target variable. Returns a tuple containing the balanced feature set and target variable,
        which can be either a pandas DataFrame or a numpy array depending on the input types.
    """
    sampler_dict = {'smote': SMOTE(sampling_strategy='auto', random_state=42),
                    'randomundersampler': RandomUnderSampler(sampling_strategy='auto', random_state=42)}
    sampler = sampler_dict.get(method.lower())
    
    if sampler is None:
        raise ValueError("Invalid method. Choose 'smote' or 'randomundersampler'.")
    
    pipeline = Pipeline([('sampler', sampler)])
    X_resampled, y_resampled = pipeline.named_steps['sampler'].fit_resample(X, y)

    return X_resampled, y_resampled

def get_data_for_shap_initiation(db_uri=HEROKU_DATABASE_URI, limit=3000)-> pd.DataFrame:
    engine = create_engine(db_uri)

    query = text(f'SELECT * FROM train_df UNION ALL SELECT * FROM test_df LIMIT {limit*2}')
    result = pd.read_sql_query(query, engine, index_col='SK_ID_CURR')
    # conn.close()

    X_train, y_train = result.drop(columns=['level_0', 'index', 'TARGET']), result['TARGET']
    X_train_resampled, _ = balance_classes(X_train, y_train, method='randomundersampler')

    data_for_shap_initiation = X_train_resampled

    return data_for_shap_initiation

def initiate_shap_explainer(api_url=f"{API_URI}initiate_shap_explainer")-> None:
    data_for_shap_initiation = get_data_for_shap_initiation()
    json_payload = data_for_shap_initiation.to_dict(orient='index')
    response = requests.post(api_url, json=json_payload)
    if response.status_code == 200:
        result = response.json()
        shap_values = result['shap_values']
        feature_names = result['feature_names']
        expected_value = result['expected_value']
        st.success("Shap explainer initiated successfully")
        st.session_state['shap']['initiated'] = True
        st.session_state['shap']['Global']['loaded'] = True
        st.session_state['shap']['Global']['features'] = data_for_shap_initiation
        st.session_state['shap']['Global']['shap_values'] = shap_values
        st.session_state['shap']['Global']['feature_names'] = feature_names
        st.session_state['shap']['Global']['expected_value'] = expected_value

    else:
        st.error("Error initiating shap explainer")


# TAB 1 FUNCTIONS (CREDIT RISK PREDICTION)
        
def get_model_threshold(api_url=f"{API_URI}model_threshold")-> Union[dict,None]:
    
    # Send POST request to API
    response = requests.get(api_url)

    if response.status_code == 200:
        result = response.json()
        return result['threshold']
    else:
        st.error("Error fetching feature importance from API")
        return None

def new_gauge_plot(confidence, threshold, result_color)-> None:

    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence*100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Credit repayment confidence", 'font': {'size': 30}},
        delta = {'reference': (1-threshold)*100, 'increasing': {'color': "Darkturquoise"}, 'decreasing': {'color': "Firebrick"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 3, 'tickfont': {'size': 20, 'color': "white"}, 'tickcolor': "white",
                     'tickvals': [0, 20, 40, 60, 80, (1-threshold)*100, 100], 'ticktext': ['0%', '20%', '40%', '60%', '80%', f'{(1-threshold)*100:.2f}%', '100%'], 'tickangle': 0},
            'bar': {'color': "white"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, (1-threshold)*100 - 5], 'color': 'Firebrick'},
                {'range': [(1-threshold)*100 - 5, (1-threshold)*100], 'color': 'Coral'},
                {'range': [(1-threshold)*100, 100], 'color': 'Darkturquoise'},],
            'threshold': {
                'line': {'color': "Gold", 'width': 4},
                'thickness': 1,
                'value': (1-threshold)*100}},
        number = {'font': {'size': 70, 'color': result_color}, 'suffix': ' %', 'valueformat': '.2f'}))

    # Add legend for gauge steps and threshold
    for name, color, symbol in zip([f"SAFE: {(1-threshold)*100:.2f}% to 100%", f"Threshold: {(1-threshold)*100:.2f}%", f"RISKY: {(1-threshold)*100 - 5:.2f}% to {(1-threshold)*100:.2f}%", f"NOPE: 0% to {(1-threshold)*100 - 5:.2f}%"], ["Darkturquoise", "Gold", "Coral", "Firebrick"], ["square", "line-ew", "square", "square"]):
        fig.add_traces(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=name,
            marker=dict(size=30, color=color, symbol=symbol, line=dict(color="white" if symbol == "square" else color, width=2)),
        ))
        fig.update_traces(
            marker_size=30,
            selector=dict(type='scatter')
        )

    # Remove axes
    fig.update_layout(
        xaxis_visible=False,
        yaxis_visible=False,
        # legend_itemsizing='constant',
        # legend_itemwidth=30
    )

    st.plotly_chart(fig, use_container_width=True)

# Function to display credit result
def display_credit_result(confidence, risk_category, threshold)-> None:

    risk_categories = {
        'SAFE': 'Darkturquoise',
        'RISKY': 'Coral',
        'NOPE': 'Firebrick'
    }

    result_color = risk_categories[risk_category]
    threshold_color = 'Gold'
    # chance_percentage = confidence * 100 if prediction == 0 else 100 - confidence * 100
    st.markdown(f"## Credit result: <span style='color:{result_color}'>{risk_category}</span>", unsafe_allow_html=True)
    st.markdown(f"#### According to our prediction model,"
                f" you have a <span style='color:{result_color}'>{confidence*100:.2f}%</span>"
                " chance to repay your loan without risk.\n"
                f"#### Our threshold is fixed at <span style='color:{threshold_color}'>{(1-threshold)*100:.2f}%</span>."
                " Please see feature importance or client informations for more information.",
                unsafe_allow_html=True)

    # Display confidence as a gauge
    st.markdown("### Credit risk profile:")
    # plot_gauge(confidence, prediction)
    new_gauge_plot(confidence, threshold, result_color)

# Function to predict credit risk
def predict_credit_risk(client_id: int, threshold: float, api_url=f"{API_URI}predict_from_dict")-> None:
    client_infos = get_client_infos(client_id=client_id, output='dict')
    json_payload_predict_from_dict = {
        'client_infos': client_infos,
        'threshold': threshold
    }

    response = requests.post(
        api_url, json=json_payload_predict_from_dict
    )

    if response.status_code == 200:
        prediction_result = response.json()
        prediction = prediction_result['prediction']
        confidence = prediction_result['confidence']
        risk_category = prediction_result['risk_category']

        # Display prediction result
        display_credit_result(confidence, risk_category, threshold)


# TAB 2 FUNCTIONS (FEATURE IMPORTANCE)

# Function to get feature importance from API
def get_built_in_global_feature_importance(api_url=f"{API_URI}global_feature_importance")-> Union[dict,None]:
    # Replace this URL with your actual API endpoint
    # api_url = api_url
    
    # Send POST request to API
    response = requests.get(api_url)

    if response.status_code == 200:
        result = response.json()
        return result
    else:
        st.error("Error fetching feature importance from API")
        return None

def display_built_in_global_feature_importance(model_type, nb_features, importance_type):
    st.markdown("### Feature importance")
    if model_type in ['XGBClassifier', 'RandomForestClassifier', 'LGBMClassifier']:
        feature_importance = st.session_state['feature_importance']['feature_importance'][importance_type]
        importance = importance_type
    else:
        st.error("Error: Unsupported model type")
        return
    
    # Sort features by importance
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:nb_features][-1::-1]
    
    # Extract feature names and importance values
    feature_names = [x[0] for x in top_features]
    importance_values = [x[1] for x in top_features]
    
    # Create horizontal bar plot using Plotly
    fig = go.Figure(go.Bar(
        x=importance_values,
        y=feature_names,
        orientation='h',
        marker=dict(color='Coral'),  # Change the color of bars if needed
    ))

    fig.update_traces(text=importance_values, textposition='outside')

    fig.update_layout(
        title=f'Top {nb_features} Features - {model_type}',
        yaxis=dict(title='Feature', titlefont=dict(size=15), tickfont=dict(size=12)),
        xaxis=dict(title=f'{importance.capitalize()} score', titlefont=dict(size=15), tickfont=dict(size=12)),
        height=300*nb_features/10 + 100,  # Adjust the height based on the number of features

    )
    
    st.plotly_chart(fig, use_container_width=True)

# Function to get shap feature importance from API
def get_shap_feature_importance(client_id: Union[int, None], scale: Literal['Global', 'Local'], api_url: str=f"{API_URI}shap_feature_importance")-> Union[dict, None]:
    if scale == 'Global':
        result = {
        'shap_values': st.session_state['shap']['Global']['shap_values'],
        'feature_names': st.session_state['shap']['Global']['feature_names'],
        'expected_value': None
    }
        return result
    
    elif scale == 'Local' and client_id is not None:    
        client_infos = get_client_infos(client_id=client_id, output='dict')
        json_payload_shap_feature_importance = {
            'client_infos': client_infos,
            'feature_scale': scale
        }

        response = requests.post(
            api_url, json=json_payload_shap_feature_importance
        )

        if response.status_code == 200:
            result = response.json()
            return result
        else:
            st.error("Error fetching feature importance from API")
            return None
    else:
        st.error("Error: Unsupported scale")
        return None

# Function to update shap session state
def update_shap_session_state(scale, features, shap_feature_importance_dict, client_id):	
    st.session_state['shap'][scale]['loaded'] = True
    st.session_state['shap'][scale]['features'] = features
    st.session_state['shap'][scale]['shap_values'] = shap_feature_importance_dict['shap_values']
    st.session_state['shap'][scale]['feature_names'] = shap_feature_importance_dict['feature_names']
    st.session_state['shap'][scale]['expected_value'] = shap_feature_importance_dict['expected_value']
    if scale == 'Local':
        st.session_state['shap'][scale]['client_id'] = client_id

# Function to display shap feature importance
def plot_shap(scale, features, shap_feature_importance_dict, nb_features: int=20):
    shap_values = np.array(shap_feature_importance_dict['shap_values'])
    feature_names = np.array(shap_feature_importance_dict['feature_names'])
    expected_value = shap_feature_importance_dict['expected_value']
    if scale == 'Global':
        plt.clf()
        st_shap(shap.summary_plot(
            shap_values, #Use Shap values array
            features=features, # Use training set features
            feature_names=feature_names, #Use column names
            show=False, #Set to false to output to folder
            max_display=nb_features) # Set max features to display
        )
    elif scale == 'Local':
        plt.clf()
        st_shap(shap.force_plot(
            expected_value, 
            shap_values, 
            features=features,
            feature_names=feature_names,
            show=False)
        )

# Function to display shap feature importance
def display_shap_feature_importance(client_id: Union[int, None], scale: Literal['Global', 'Local'], nb_features: int=20)-> None:
    if scale == 'Global':
        shap_feature_importance_dict = {
            'features': st.session_state['shap'][scale]['features'],
            'shap_values': st.session_state['shap'][scale]['shap_values'],
            'feature_names': st.session_state['shap'][scale]['feature_names'],
            'expected_value': st.session_state['shap'][scale]['expected_value']
        }
        plot_shap(scale=scale, features=st.session_state['shap'][scale]['features'], shap_feature_importance_dict=shap_feature_importance_dict, nb_features=nb_features)
    elif scale == 'Local':
        if not st.session_state['shap'][scale]['loaded'] or client_id is not st.session_state['shap'][scale]['client_id']:
            shap_feature_importance_dict = get_shap_feature_importance(client_id=client_id, scale=scale, api_url="http://127.0.0.1:8000/shap_feature_importance")
            features = st.session_state['shap'][scale]['features']
            plot_shap(scale=scale, features=features, shap_feature_importance_dict=shap_feature_importance_dict, nb_features=nb_features)
            update_shap_session_state(scale=scale, features=features, shap_feature_importance_dict=shap_feature_importance_dict, client_id=client_id)
        else:
            shap_feature_importance_dict = {
                'features': st.session_state['shap'][scale]['features'],
                'shap_values': st.session_state['shap'][scale]['shap_values'],
                'feature_names': st.session_state['shap'][scale]['feature_names'],
                'expected_value': st.session_state['shap'][scale]['expected_value']
            }
            plot_shap(scale=scale, features=st.session_state['shap'][scale]['features'], shap_feature_importance_dict=shap_feature_importance_dict, nb_features=nb_features)


# TAB 3 FUNCTIONS (CLIENT COMPARISON)

def fetch_cat_and_split_features(db_uri=HEROKU_DATABASE_URI, nb_categories=7, limit=3000):
    global_features = st.session_state['available_features']['global_features']
    engine = create_engine(db_uri)

    # return categorical_cols, binary_cols
    connection = engine.connect()

    # Query the first table
    table1 = 'train_df'
    query = f"SELECT * FROM {table1} ORDER BY RANDOM() LIMIT {limit}"
    df1 = pd.read_sql(query, connection)
    table2 = 'test_df'
    query = f"SELECT * FROM {table2} ORDER BY RANDOM() LIMIT {limit}"
    df2 = pd.read_sql(query, connection)
    df = pd.concat([df1, df2], axis=0)
    df.drop(columns=['level_0'], inplace=True)

    # Find categorical columns with no more than 'nb_categories' unique values and binary features
    categorical_cols = []
    binary_cols = []
    for col in df.columns:
        if col != 'SK_ID_CURR':
            unique_values = df[col].nunique()
            if unique_values <= nb_categories:
                categorical_cols.append(col)
            if unique_values == 2:
                binary_cols.append(col)

    # Order both lists using order from global_features
    categorical_cols = [col for col in global_features if col in categorical_cols]
    binary_cols = [col for col in global_features if col in binary_cols]

    # Add 'TARGET' at the beginning of both lists
    categorical_cols.insert(0, 'TARGET')
    binary_cols.insert(0, 'TARGET')

    connection.close()
    st.session_state['available_features']['categorical_features'] = categorical_cols
    st.session_state['available_features']['split_features'] = binary_cols
    st.session_state['available_features']['initiated'] = True

def update_available_features():
    importance_type = st.session_state['tab_3_selected_importance_type']
    global_features = [key for key, _ in sorted(st.session_state['feature_importance']['feature_importance'][importance_type].items(), key=lambda item: item[1], reverse=True)]
    categorical_features = [feature for feature in global_features if feature in st.session_state['available_features']['categorical_features']]
    split_features = [feature for feature in global_features if feature in st.session_state['available_features']['split_features']]
    categorical_features.insert(0, 'TARGET')
    split_features.insert(0, 'TARGET')

    st.session_state['available_features'] = {
        'initiated': True,
        'global_features': global_features,
        'categorical_features': categorical_features,
        'split_features': split_features
    }

def update_violinplot_data(db_uri=HEROKU_DATABASE_URI):

    selected_global_feature = st.session_state['client_comparison']['global']
    selected_categorical_feature = st.session_state['client_comparison']['categorical']
    selected_split_feature = st.session_state['client_comparison']['split']
    client_id = st.session_state['client_id']
    limit = st.session_state['limit']

    # Retrieve data
    # conn = sqlite3.connect('C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/features/clients_infos.db')
    engine = create_engine(db_uri)

    pass_query = False
    query = ""
    # Build the query based on the provided columns
    if selected_global_feature is None and selected_categorical_feature is None and selected_split_feature is None:
        pass_query = True
    else:
        selected_features = [feat for feat in [selected_global_feature, selected_categorical_feature, selected_split_feature] if feat is not None]
        selected_features_str = '", "'.join(selected_features)

        query = text(f'''
            WITH random_train AS (
                SELECT "SK_ID_CURR", "{selected_features_str}"
                FROM train_df
                WHERE "SK_ID_CURR" != {client_id}
                ORDER BY RANDOM()
                LIMIT {limit}
            ),
            random_test AS (
                SELECT "SK_ID_CURR", "{selected_features_str}"
                FROM test_df
                WHERE "SK_ID_CURR" != {client_id}
                ORDER BY RANDOM()
                LIMIT {limit}
            )
            SELECT * FROM random_train
            UNION ALL
            SELECT * FROM random_test
        ''')
        query_client = text(f'SELECT "SK_ID_CURR", "{selected_features_str}" FROM train_df WHERE "SK_ID_CURR" = {client_id} UNION ALL SELECT "SK_ID_CURR", "{selected_features_str}" FROM test_df WHERE "SK_ID_CURR" = {client_id}')

    # Execute the query and read the result into a DataFrame
    df = pd.DataFrame() if pass_query else pd.read_sql_query(query, engine)
    row_client = pd.DataFrame() if pass_query else pd.read_sql_query(query_client, engine)

    concatenated_df = pd.concat([row_client, df], ignore_index=True)

    st.session_state['client_comparison']['data'] = concatenated_df

def display_violinplot():
    df = st.session_state['client_comparison']['data']
    client_id = st.session_state['client_id']
    if df.shape[1] > 0:
        df.set_index('SK_ID_CURR', inplace=True, drop=True)
    else:
        st.warning("Please select at least a global feature.")
        return

    if df.shape[1] == 0:
        st.warning("Please select at least a global feature.")
        return
    elif df.shape[1] == 1:
        plot_global_violin(df, client_id)
    elif df.shape[1] == 2:
        plot_categorical_violin(df, client_id)
    else:
        plot_split_violin(df, client_id)

def plot_split_violin(df: pd.DataFrame, client_id: int):
    global_feature = df.columns[0]
    categorical_feature = df.columns[1]
    split_feature = df.columns[2]
    palette = generate_palette_without_gold(2)
    client_color = palette[0] if df[split_feature][client_id] == df[split_feature].unique()[0] else palette[1]
    client_line_color = 'Gold'

    plt.clf()
    fig = go.Figure()

    for split, side, color in zip(df[split_feature].unique(), ['negative', 'positive'], palette):
        fig.add_trace(go.Violin(
            x=df[categorical_feature][ df[split_feature] == split ],
            y=df[global_feature][ df[split_feature] == split ],
            legendgroup=str(split), scalegroup=str(split), name=str(split), side=side, line_color=color, box_visible=True
        ))
    fig.update_traces(meanline_visible=True, points='all', jitter=0.2, scalemode='count', marker=dict(size=1))
    fig.add_trace(go.Scatter(
        x=[df[categorical_feature][client_id]], y=[df[global_feature][client_id]], name=f'Current client: ID = {client_id} | Split group = {df[split_feature][client_id]}', mode='markers', marker=dict(color=client_color, size=15, line_width=3, line_color=client_line_color)
    ))
    fig.update_traces(text=f'Current client: ID = {client_id} | Split group:{str(df[split_feature][client_id])}', selector=dict(type='scatter'))
    fig.update_layout(violingap=0, violinmode='overlay', title_text='Client comparison', yaxis_title=f"Global feature: {global_feature}", xaxis_title=f"Categorical feature: {categorical_feature}", xaxis_tickformat='.0f', xaxis_tickvals=list(df[categorical_feature].sort_values().unique()))
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, title=f'Split feature: {split_feature}'))
    st.plotly_chart(fig, use_container_width=True)

def plot_categorical_violin(df: pd.DataFrame, client_id: int):
    global_feature = df.columns[0]
    categorical_feature = df.columns[1]
    palette = generate_palette_without_gold(len(df[categorical_feature].unique()))
    client_color = 'Gold'

    plt.clf()
    fig = go.Figure()
    for cat, color in zip(df[categorical_feature].sort_values().unique(), palette):
        fig.add_trace(go.Violin(
            x=df[categorical_feature][ df[categorical_feature] == cat ],
            y=df[global_feature][ df[categorical_feature] == cat ],
            legendgroup=str(cat), scalegroup=str(cat), name=str(cat), line_color=color, box_visible=True
        ))
    fig.update_traces(meanline_visible=True, points='all', jitter=0.2, scalemode='count', marker=dict(size=1))
    fig.add_trace(go.Scatter(
        x=[df[categorical_feature][client_id]], y=[df[global_feature][client_id]], name=f'Current client: ID = {client_id}', mode='markers', marker=dict(color=client_color, size=15)
    ))
    fig.update_traces(text=f'Current client: ID = {client_id}', selector=dict(type='scatter'))
    fig.update_layout(violingap=0, violinmode='overlay', title_text='Client comparison', yaxis_title=f"Global feature: {global_feature}", xaxis_title=f"Categorical feature: {categorical_feature}", xaxis_tickformat='.0f', xaxis_tickvals=list(df[categorical_feature].sort_values().unique()))
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, title=f'Categorical feature: {categorical_feature}'))
    st.plotly_chart(fig, use_container_width=True)

def plot_global_violin(df: pd.DataFrame, client_id: int):
    global_feature = df.columns[0]
    color = 'Paleturquoise'
    client_color = 'Gold'

    plt.clf()
    fig = go.Figure()
    fig.add_trace(go.Violin(
        y=df[global_feature],
        legendgroup=global_feature, scalegroup=global_feature, name=global_feature, line_color=color, box_visible=True
    ))
    fig.update_traces(meanline_visible=True, points='all', jitter=0.2, scalemode='count', marker=dict(size=1))
    fig.add_trace(go.Scatter(
        x=[global_feature], y=[df[global_feature][client_id]], name=f'Current client: ID = {client_id}', mode='markers', marker=dict(color=client_color, size=15)
    ))
    fig.update_traces(text=f'Current client: ID = {client_id}', selector=dict(type='scatter'))
    fig.update_layout(violingap=0, violinmode='overlay', title_text='Client comparison')
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, title=f'Global feature: {global_feature}'))
    st.plotly_chart(fig, use_container_width=True)