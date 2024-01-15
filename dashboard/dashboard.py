import pandas as pd
import streamlit as st
import requests
from client_infos import ClientInfosDebugSplit
import sqlite3

# streamlit run dashboard.py

conn = sqlite3.connect('C:/Users/emile/DEV/WORKSPACE/projet-7-cours-oc/model/model/features/clients_infos.db')

st.title('Credit risk prediction')

client_id = st.number_input(
    label='Client ID', min_value=0, max_value=1000000, value=108567, step=1,
    format='%i', placeholder='Enter client ID'
)

query = f"SELECT * FROM train_df_debug WHERE SK_ID_CURR={client_id}"
result = pd.read_sql_query(query, conn, index_col='SK_ID_CURR')
conn.close()

model_dict = result.drop(columns=['index', 'TARGET']).to_dict(orient='index')

client_infos = model_dict[client_id]

json_payload_predict_from_dict = client_infos

response = requests.post('http://127.0.0.1:8000/predict_from_dict', json=json_payload_predict_from_dict)

if response.status_code == 200:
    prediction_result = response.json()