import matplotlib.pyplot as plt
import streamlit as st
import requests
import sqlite3
import pandas as pd
from typing import Tuple, Literal, Union
from imblearn.over_sampling import SMOTE
from imblearn .under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import shap
import numpy as np
from streamlit_shap import st_shap
import seaborn as sns
import plotly.graph_objects as go

# COMMON FUNCTIONS

# Function to toggle debug_mode
def toggle_debug_mode():
    st.session_state['debug_mode'] = not st.session_state['debug_mode']
    st.session_state['client_id'] = None
    st.session_state['shap']['initiated'] = False
    st.session_state['shap']['Global']['loaded'] = False
    st.session_state['shap']['Local']['loaded'] = False

# Function to submit client id
def submit_client_id(client_id: int):
    st.session_state['client_id'] = client_id

# Function to get client infos from id number 
def get_client_infos(client_id: int, output: Literal['dict', 'df'] = 'df', debug: bool = False)-> Union[dict, pd.DataFrame]:
    conn = sqlite3.connect(
        'C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/features/clients_infos.db'
    )
    if debug:
        query = f'SELECT * FROM train_df_debug WHERE SK_ID_CURR = ? ORDER BY "index"'
        result = pd.read_sql_query(query, conn, params=[client_id], index_col='SK_ID_CURR')
        conn.close()
    else:
        query = f'SELECT * FROM train_df WHERE SK_ID_CURR = ? UNION ALL SELECT * FROM test_df WHERE SK_ID_CURR = ? ORDER BY "index"'
        result = pd.read_sql_query(query, conn, params=[client_id, client_id], index_col='SK_ID_CURR')
        conn.close()

    if output == 'dict':
        model_dict = result.drop(columns=['index', 'TARGET']).to_dict(orient='index')
        client_infos = model_dict[client_id]
        return client_infos
    elif output == 'df':
        client_infos = result.drop(columns=['index', 'TARGET'])
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

def rename_columns_based_on_col_index(df, realnames):
    # Rename columns based on realnames list
    num_columns = min(len(df.columns), len(realnames))
    for idx in range(num_columns):
        df.rename(columns={df.columns[idx]: realnames[idx]}, inplace=True)
    return df

# SIDEBAR FUNCTIONS FOR SHAP INITIATION

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

def get_data_for_shap_initiation(debug: bool=False)-> pd.DataFrame:
    conn = sqlite3.connect(
        'C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/features/clients_infos.db'
    )
    if debug:
        query = f'SELECT * FROM train_df_debug'
        result = pd.read_sql_query(query, conn, index_col='SK_ID_CURR')
        conn.close()
    else:
        query = f'SELECT * FROM train_df UNION ALL SELECT * FROM test_df'
        result = pd.read_sql_query(query, conn, index_col='SK_ID_CURR')
        conn.close()

    X_train, y_train = result.drop(columns=['index', 'TARGET']), result['TARGET']
    X_train_resampled, _ = balance_classes(X_train, y_train, method='randomundersampler')

    data_for_shap_initiation = X_train_resampled

    return data_for_shap_initiation

def initiate_shap_explainer(api_url="http://127.0.0.1:8000/initiate_shap_explainer")-> None:
    debug = st.session_state['debug_mode']
    data_for_shap_initiation = get_data_for_shap_initiation(debug=debug)
    response = requests.post(api_url)
    if response.status_code == 200:
        st.success("Shap explainer initiated successfully")
        st.session_state['shap']['initiated'] = True
        st.session_state['shap']['Global']['features'] = data_for_shap_initiation
    else:
        st.error("Error initiating shap explainer")


# TAB 1 FUNCTIONS (CREDIT RISK PREDICTION)

def new_gauge_plot(confidence, prediction, threshold, result_color)-> None:
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

    # fig.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

    st.plotly_chart(fig, use_container_width=True)

# Function to display credit result
def display_credit_result(prediction, confidence, risk_category, threshold=.162)-> None:

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
    new_gauge_plot(confidence, prediction, threshold, result_color)

# Function to predict credit risk
def predict_credit_risk(client_id: int, threshold: float = .162, api_url="http://127.0.0.1:8000/predict_from_dict", debug: bool = False)-> None:
    client_infos = get_client_infos(client_id=client_id, output='dict', debug=debug)
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
        display_credit_result(prediction, confidence, risk_category, threshold=threshold)


# TAB 2 FUNCTIONS (FEATURE IMPORTANCE)

# Function to get feature importance from API
def get_built_in_global_feature_importance(api_url="http://127.0.0.1:8000/global_feature_importance")-> dict | None:
    # Replace this URL with your actual API endpoint
    api_url = api_url
    
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
        marker=dict(color='royalblue'),  # Change the color of bars if needed
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
def get_shap_feature_importance(client_id: Union[int, None], scale: Literal['Global', 'Local'], api_url: str="http://127.0.0.1:8000/shap_feature_importance", debug: bool = False)-> Union[dict, None]:
    if scale == 'Global':
        json_payload_shap_feature_importance = {
            'client_infos': None,
            'feature_scale': scale
        }
    elif scale == 'Local' and client_id is not None:    
        client_infos = get_client_infos(client_id=client_id, output='dict', debug=debug)
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
            # plot_size=(10,10)) # Change plot size
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
def display_shap_feature_importance(client_id: Union[int, None], scale: Literal['Global', 'Local'], nb_features: int=20, debug: bool = False)-> None:
    if scale == 'Global':
        if not st.session_state['shap'][scale]['loaded']:
            shap_feature_importance_dict = get_shap_feature_importance(client_id=client_id, scale=scale, api_url="http://127.0.0.1:8000/shap_feature_importance", debug=debug)
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
    elif scale == 'Local':
        if not st.session_state['shap'][scale]['loaded'] or client_id is not st.session_state['shap'][scale]['client_id']:
            shap_feature_importance_dict = get_shap_feature_importance(client_id=client_id, scale=scale, api_url="http://127.0.0.1:8000/shap_feature_importance", debug=debug)
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


def fetch_cat_and_split_features(global_features, debug: bool=False):
    conn = sqlite3.connect('C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/features/clients_infos.db')
    if debug:
        features_query = "PRAGMA table_info(train_df_debug)"
        features_df = pd.read_sql_query(features_query, conn)
        cursor = conn.cursor()
    else:
        features_query = "PRAGMA table_info(train_df)"
        features_df = pd.read_sql_query(features_query, conn)
        cursor = conn.cursor()

    def is_binary_column(column_name, debug: bool=False):
        if debug:
            cursor.execute(f"SELECT COUNT(DISTINCT {column_name}) FROM train_df_debug")
            num_distinct_values = cursor.fetchone()[0]
            return num_distinct_values == 2
        else:
            cursor.execute(f"SELECT COUNT(DISTINCT {column_name}) FROM (SELECT {column_name} FROM train_df UNION ALL SELECT {column_name} FROM test_df) AS concatenated_tables")
            num_distinct_values = cursor.fetchone()[0]
            return num_distinct_values == 2
    
    def has_more_than_7_unique_values(column_name, debug: bool=False):
        if debug:
            cursor.execute(f"SELECT COUNT(DISTINCT {column_name}) FROM train_df_debug")
            num_distinct_values = cursor.fetchone()[0]
            return num_distinct_values > 7
        else:
            cursor.execute(f"SELECT COUNT(DISTINCT {column_name}) FROM (SELECT {column_name} FROM train_df UNION ALL SELECT {column_name} FROM test_df) AS concatenated_tables")
            num_distinct_values = cursor.fetchone()[0]
            return num_distinct_values > 7

    # Filter out features with data type 'int' (assuming they are categorical)
    categorical_features = features_df[features_df['type'] == 'INTEGER']['name'].tolist()

    # Remove 'index' and 'SK_ID_CURR' from the list of categorical features
    categorical_features = [feature for feature in categorical_features if feature not in ['index', 'SK_ID_CURR']]

    # Sort categorical features according to the order of global features and remove features with more than 7 unique values
    ordered_categorical_features = [feature for feature in global_features if (feature in categorical_features and not has_more_than_7_unique_values(feature, debug=debug))]

    # Insert 'TARGET' after an empty string at the beginning of the list
    ordered_categorical_features.insert(0, 'TARGET')

    # remove features with more than two unique values
    split_features = [feature for feature in ordered_categorical_features if is_binary_column(feature, debug=debug)]

    conn.close()

    return ordered_categorical_features, split_features

def update_available_features():
    importance_type = st.session_state['tab_3_selected_importance_type']
    global_features = [key for key, _ in sorted(st.session_state['feature_importance']['feature_importance'][importance_type].items(), key=lambda item: item[1], reverse=True)]
    categorical_features, split_features = fetch_cat_and_split_features(global_features, debug=st.session_state['debug_mode'])

    st.session_state['available_features'] = {
        'global_features': global_features,
        'categorical_features': categorical_features,
        'split_features': split_features
    }

def update_violinplot_data():#selected_global_feature: str, selected_categorical_feature: str, selected_split_feature: str, client_id: int, limit: int, debug: bool=False):

    selected_global_feature = st.session_state['client_comparison']['global']
    selected_categorical_feature = st.session_state['client_comparison']['categorical']
    selected_split_feature = st.session_state['client_comparison']['split']
    client_id = st.session_state['client_id']
    limit = 3000
    debug = st.session_state['debug_mode']

    # Retrieve data
    conn = sqlite3.connect('C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/features/clients_infos.db')

    pass_query = False
    query = ""
    # Build the query based on the provided columns
    if selected_global_feature is None and selected_categorical_feature is None and selected_split_feature is None:
        pass_query = True
    else:
        selected_features = [feat for feat in [selected_global_feature, selected_categorical_feature, selected_split_feature] if feat is not None]
        selected_features_str = ", ".join(selected_features)

        if debug:
            query = f"SELECT SK_ID_CURR, {selected_features_str} FROM train_df_debug WHERE SK_ID_CURR != {client_id} ORDER BY RANDOM() LIMIT {limit}"
            query_client = f"SELECT SK_ID_CURR, {selected_features_str} FROM train_df_debug WHERE SK_ID_CURR = {client_id}"
        else:
            query = f"SELECT SK_ID_CURR, {selected_features_str} FROM train_df WHERE SK_ID_CURR != {client_id} ORDER BY RANDOM() LIMIT {limit} UNION ALL SELECT SK_ID_CURR, {selected_features_str} FROM test_df WHERE SK_ID_CURR != {client_id} ORDER BY RANDOM() LIMIT {limit}"
            query_client = f"SELECT SK_ID_CURR, {selected_features_str} FROM train_df WHERE SK_ID_CURR = {client_id} UNION ALL SELECT SK_ID_CURR, {selected_features_str} FROM test_df WHERE SK_ID_CURR = {client_id}"

    # Execute the query and read the result into a DataFrame
    df = pd.DataFrame() if pass_query else pd.read_sql_query(query, conn)
    row_client = pd.DataFrame() if pass_query else pd.read_sql_query(query_client, conn)

    concatenated_df = pd.concat([df, row_client])

    # Close the connection
    conn.close()

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