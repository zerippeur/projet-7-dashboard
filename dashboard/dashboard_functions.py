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

def plot_gauge(value, prediction)-> None:
    colors = ['Darkturquoise', 'Paleturquoise', 'Gold', 'Coral', 'Firebrick']
    values = [100, 80, 60, 40, 20, 0]
    x_axis_values = [0, 0.628, 1.256, 1.884, 2.512, 3.14]

    plt.clf()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='polar')
    ax.bar(x=x_axis_values[:-1], height=0.5, width=0.63, color=colors, bottom=2, align='edge', linewidth=3, edgecolor='white')

    y_areas = 2.23
    fontsize_areas = 18

    plt.annotate("SAFE", xy=(0.314,y_areas), rotation=-72, color="white", fontweight="bold", fontsize=fontsize_areas, va="center", ha="center")
    plt.annotate("UNCERTAIN", xy=(0.942,y_areas), rotation=-36, color="white", fontweight="bold", fontsize=fontsize_areas, va="center", ha="center")
    plt.annotate("RISKY", xy=(1.57,y_areas), color="white", fontweight="bold", fontsize=fontsize_areas, va="center", ha="center")
    plt.annotate("VERY RISKY", xy=(2.198,y_areas), rotation=36, color="white", fontweight="bold", fontsize=fontsize_areas, va="center", ha="center")
    plt.annotate("NOPE", xy=(2.826,y_areas), rotation=72, color="white", fontweight="bold", fontsize=fontsize_areas, va="center", ha="center")

    for loc, val in zip(x_axis_values, values):
        plt.annotate(f'{val}%', xy=(loc, 2.5), va='bottom', ha='right' if val<=40 else 'left', fontsize=14)

    gauge_value = value if prediction == 0 else 100 - value
    gauge_position = 3.14-(value*0.0314) if prediction == 0 else 3.14-((100 - value)*0.0314)
    plt.annotate(f'{gauge_value:.2f}%', xytext=(0,0), xy=(gauge_position, 2.0),
             arrowprops=dict(arrowstyle="wedge, tail_width=0.5", color="black", shrinkA=0),
             bbox=dict(boxstyle="circle", facecolor="black", linewidth=2.0, ),
             fontsize=25, color="white", ha="center", va='center', fontweight="bold"
            )


    plt.title("Confidence meter", loc="center", pad=20, fontsize=20, fontweight="bold")

    ax.set_axis_off()
    st.pyplot(fig)

def display_credit_result(prediction, confidence, risk_category, threshold=.5)-> None:

    risk_categories = {
        'SAFE': 'Darkturquoise',
        'UNCERTAIN': 'Paleturquoise',
        'RISKY': 'Gold',
        'VERY RISKY': 'Coral',
        'NOPE': 'Firebrick'
    }

    font_color = risk_categories[risk_category]
    chance_percentage = confidence * 100 if prediction == 0 else 100 - confidence * 100
    st.markdown(f"## Credit approval: <span style='color:{font_color}'>{risk_category}</span>", unsafe_allow_html=True)
    st.markdown(f"#### According to our prediction model,"
                f" you have a <span style='color:{font_color}'>{chance_percentage:.2f}%</span>"
                " chance to repay your loan.\n"
                f"#### Our threshold is fixed at {threshold*100:.2f}%."
                " Please see feature importance or client informations for more information.",
                unsafe_allow_html=True)

    # Display confidence as a gauge
    st.markdown("### Credit risk profile:")
    plot_gauge(confidence * 100, prediction)

# Function to get feature importance from API
def get_built_in_global_feature_importance(api_url="http://127.0.0.1:8000/global_feature_importance")-> dict | None:
    # Replace this URL with your actual API endpoint
    api_url = api_url
    
    # Send POST request to API
    response = requests.post(api_url)

    if response.status_code == 200:
        result = response.json()
        return result
    else:
        st.error("Error fetching feature importance from API")
        return None
    
def display_built_in_global_feature_importance(model_type, nb_features, importance_type)-> None:
    st.markdown("### Feature importance")
    if model_type == 'XGBClassifier':
        feature_importance = st.session_state['feature_importance']['feature_importance'][importance_type]
        importance = importance_type
    elif model_type == 'RandomForestClassifier':
        feature_importance = st.session_state['feature_importance']['feature_importance']
        importance = 'Importance'
    else:
        st.error("Error: Unsupported model type")
        return

    # Sort features by importance
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:nb_features]

    # Create horizontal bar plot
    plt.clf()
    fig = plt.figure(figsize=(10, 6*nb_features/10))
    plt.barh(range(len(top_features) - 1, -1, -1), [x[1] for x in top_features], tick_label=[x[0] for x in top_features])
    plt.xlabel(f'{importance}')
    plt.ylabel('Feature')
    plt.title(f'Top {nb_features} Features - {model_type}')
    st.pyplot(fig)

def get_client_infos(client_id: int, output: Literal['dict', 'df'] = 'df')-> dict | pd.DataFrame:
    conn = sqlite3.connect(
        'C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/features/clients_infos.db'
    )
    query = f'SELECT * FROM train_df_debug WHERE SK_ID_CURR = ? ORDER BY "index"'
    result = pd.read_sql_query(query, conn, params=[client_id], index_col='SK_ID_CURR')
    conn.close()

    if output == 'dict':
        model_dict = result.drop(columns=['index', 'TARGET']).to_dict(orient='index')
        client_infos = model_dict[client_id]
        return client_infos
    elif output == 'df':
        client_infos = result.drop(columns=['index', 'TARGET'])
        return client_infos

def predict_credit_risk(client_id: int, threshold: float = .5, api_url="http://127.0.0.1:8000/predict_from_dict")-> None:
    client_infos = get_client_infos(client_id=client_id, output='dict')
    json_payload_predict_from_dict = client_infos

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
    # return pipeline.fit_resample(X, y)
    return X_resampled, y_resampled

def get_data_for_shap_initiation()-> dict:
    conn = sqlite3.connect(
        'C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/features/clients_infos.db'
    )
    query = f'SELECT * FROM train_df_debug'
    result = pd.read_sql_query(query, conn, index_col='SK_ID_CURR')
    conn.close()

    X_train, y_train = result.drop(columns=['index', 'TARGET']), result['TARGET']
    X_train_resampled, _ = balance_classes(X_train, y_train, method='randomundersampler')

    data_for_shap_initiation = X_train_resampled.to_dict(orient='index')

    return data_for_shap_initiation

def initiate_shap_explainer(api_url="http://127.0.0.1:8000/initiate_shap_explainer")-> None:
    data_for_shap_initiation = get_data_for_shap_initiation()
    response = requests.post(api_url, json=data_for_shap_initiation)
    if response.status_code == 200:
        st.session_state.shap_explainer_initiated = True
        st.success("Shap explainer initiated successfully")
    else:
        st.error("Error initiating shap explainer")
        st.session_state.shap_explainer_initiated = False

def get_feature_names_in(api_url="http://127.0.0.1:8000/feature_names_in")-> Union[dict, None]:
    response = requests.get(api_url)
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        st.error("Error fetching feature names from API")
        return None

def get_shap_feature_importance(client_id: int|None, scale: Literal['Global', 'Local'], api_url: str="http://127.0.0.1:8000/shap_feature_importance")-> Union[dict, None]:
    if scale == 'Global':
        json_payload_shap_feature_importance = {
            'client_infos': None,
            'feature_scale': scale
        }
    elif scale == 'Local':    
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

def update_shap_session_state(client_id, scale, data):
    st.session_state['shap'][scale]['initiated'] = True
    st.session_state['shap'][scale]['data'] = data
    st.session_state['shap'][scale]['client_id'] = client_id

def plot_shap(shap_feature_importance_dict, scale, nb_features: int=20):
    data = pd.DataFrame.from_dict(shap_feature_importance_dict['data'], orient='index')
    shap_values = np.array(shap_feature_importance_dict['shap_values'])
    feature_names = np.array(shap_feature_importance_dict['feature_names'])
    expected_value = shap_feature_importance_dict['expected_value']
    if scale == 'Global':
        plt.clf()
        st_shap(shap.summary_plot(
            shap_values, #Use Shap values array
            features=data, # Use training set features
            feature_names=feature_names, #Use column names
            show=False, #Set to false to output to folder
            max_display=nb_features, # Set max features to display
            plot_size=(10,10)) # Change plot size
        )
    elif scale == 'Local':
        plt.clf()
        st_shap(shap.force_plot(
            expected_value, 
            shap_values, 
            features=data,
            feature_names=data.columns,
            show=False)
        )

def display_shap_feature_importance(client_id: int|None, scale: Literal['Global', 'Local'], nb_features: int=20)-> None:
    if scale == 'Global':
        if not st.session_state['shap'][scale]['initiated']:
            shap_feature_importance_dict = get_shap_feature_importance(client_id=client_id, scale=scale, api_url="http://127.0.0.1:8000/shap_feature_importance")
            plot_shap(shap_feature_importance_dict, scale=scale, nb_features=nb_features)
            update_shap_session_state(client_id=client_id, scale=scale, data=shap_feature_importance_dict)
        else:
            plot_shap(st.session_state['shap'][scale]['data'], scale=scale, nb_features=nb_features)
    elif scale == 'Local':
        if not st.session_state['shap'][scale]['initiated'] or client_id is not st.session_state['shap'][scale]['client_id']:
            shap_feature_importance_dict = get_shap_feature_importance(client_id=client_id, scale=scale, api_url="http://127.0.0.1:8000/shap_feature_importance")
            plot_shap(shap_feature_importance_dict, scale=scale, nb_features=nb_features)
            update_shap_session_state(client_id=client_id, scale=scale, data=shap_feature_importance_dict)
        else:
            plot_shap(st.session_state['shap'][scale]['data'], scale=scale, nb_features=nb_features)