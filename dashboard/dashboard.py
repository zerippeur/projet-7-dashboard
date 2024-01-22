import pandas as pd
import streamlit as st
import requests
from client_infos import ClientInfosDebugSplit
import sqlite3
import matplotlib.pyplot as plt

risk_categories = {
    'SAFE': 'Darkturquoise',
    'UNCERTAIN': 'Paleturquoise',
    'RISKY': 'Gold',
    'VERY RISKY': 'Coral',
    'NOPE': 'Firebrick'
}

def plot_gauge(value, prediction):
    colors = ['Darkturquoise', 'Paleturquoise', 'Gold', 'Coral', 'Firebrick']
    values = [100, 80, 60, 40, 20, 0]
    x_axis_values = [0, 0.628, 1.256, 1.884, 2.512, 3.14]

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

def display_credit_result(prediction, confidence, risk_category):
    font_color = risk_categories[risk_category]
    st.markdown(f"## Credit approval: <span style='color:{font_color}'>{risk_category}</span>", unsafe_allow_html=True)

    # Display confidence as a gauge
    st.markdown("### Credit risk profile:")
    plot_gauge(confidence * 100, prediction)

st.set_page_config(
    page_title='Credit risk prediction', page_icon=':clipboard:', layout='wide'
)

# Function to get feature importance from API
def get_global_feature_importance():
    # Replace this URL with your actual API endpoint
    api_url = "http://127.0.0.1:8000/global_feature_importance"
    
    
    # Send POST request to API
    response = requests.post(api_url)

    if response.status_code == 200:
        result = response.json()
        return result
    else:
        st.error("Error fetching feature importance from API")
        return None
    
def display_feature_importance(model_type, nb_features, importance_type):
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
    fig = plt.figure(figsize=(10, 6*nb_features/10))
    plt.barh(range(len(top_features) - 1, -1, -1), [x[1] for x in top_features], tick_label=[x[0] for x in top_features])
    plt.xlabel(f'{importance}')
    plt.ylabel('Feature')
    plt.title(f'Top {nb_features} Features - {model_type}')
    st.pyplot(fig)

# streamlit run dashboard.py

st.title('Credit risk prediction')
st.markdown('Get credit risk prediction for a client based on his/her ID')


tab1, tab2, tab3 = st.tabs([':clipboard: Credit risk prediction', ':bar_chart: Feature importance', ':chart_with_upwards_trend: Client infos'])

if 'client_id' not in st.session_state:
    st.session_state['client_id'] = None

if 'feature_importance' not in st.session_state:
    st.session_state['feature_importance'] = get_global_feature_importance()

with tab1:
    client_id = st.number_input(
        label='Client ID', min_value=0, max_value=1000000, value=st.session_state['client_id'],
        step=1, format='%i', placeholder='Enter client ID'
    )
    submit = st.button('Predict credit risk', key='submit')

    if submit:
        st.session_state['client_id'] = client_id
        st.write('Current client ID:', st.session_state['client_id'])

        conn = sqlite3.connect(
            'C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/features/clients_infos.db'
        )
        query = f"SELECT * FROM train_df_debug WHERE SK_ID_CURR={client_id}"
        result = pd.read_sql_query(query, conn, index_col='SK_ID_CURR')
        conn.close()

        model_dict = result.drop(columns=['index', 'TARGET']).to_dict(orient='index')
        client_infos = model_dict[client_id]
        json_payload_predict_from_dict = client_infos

        response = requests.post(
            'http://127.0.0.1:8000/predict_from_dict', json=json_payload_predict_from_dict
        )

        if response.status_code == 200:
            prediction_result = response.json()
            prediction = prediction_result['prediction']
            confidence = prediction_result['confidence']
            risk_category = prediction_result['risk_category']

            # Display prediction result
            display_credit_result(prediction, confidence, risk_category)

with tab2:
    st.markdown('## Feature importance')
    st.write('Current client ID:', st.session_state['client_id'])

    st.markdown('### Global feature importance')

    model_type = st.session_state['feature_importance']['model_type']

    if st.session_state['feature_importance']['model_type'] == 'XGBClassifier':
        importance_type = st.radio("Select importance type:", ["weight", "cover", "gain"], index=0)
    elif st.session_state['feature_importance']['model_type'] == 'RandomForestClassifier':
        pass

    nb_features = st.number_input(
        label='Features nb', min_value=0, max_value=30, value=20,
        step=1, format='%i', placeholder='Enter number of features to display'
    )

    display_feature_importance(model_type, nb_features, importance_type)

with tab3:
    st.markdown('## Client infos')
    st.write('Current client ID:', st.session_state['client_id'])


