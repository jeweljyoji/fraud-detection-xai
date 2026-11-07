import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

THRESHOLD = 0.3
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '..', '.env'))

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")


@st.cache_resource
def load_everything():
    model = joblib.load(os.path.join(BASE_DIR, 'fraud_model.pkl'))
    feature_cols = joblib.load(os.path.join(BASE_DIR, 'feature_cols.pkl'))
    results_path = os.path.join(BASE_DIR, '..', 'data', 'dashboard_results.csv')
    results = pd.read_csv(results_path, index_col=0)
    background = results[feature_cols].sample(100, random_state=42)
    explainer = shap.TreeExplainer(model, background, model_output='probability')
    return model, feature_cols, results, explainer


model, feature_cols, results, explainer = load_everything()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

st.title("Fraud Detection Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Transactions analysed", f"{len(results):,}")
col2.metric("Flagged for review", int(results['predicted_fraud'].sum()))
col3.metric("Decision threshold", THRESHOLD)

show_only_flagged = st.checkbox("Show only flagged transactions", value=True)
if show_only_flagged:
    display_df = results[results['predicted_fraud'] == 1]
else:
    display_df = results

st.write(f"Showing {len(display_df)} transactions")
table_cols = ['Time', 'Amount', 'predicted_proba', 'predicted_fraud', 'Class']
st.dataframe(display_df[table_cols].head(50))

st.subheader("Inspect a transaction")
selected_id = st.selectbox("Transaction ID", display_df.index[:50])

row = results.loc[selected_id]
st.write(f"**Amount:** £{row['Amount']:.2f}  |  "
         f"**Predicted fraud probability:** {row['predicted_proba']:.2%}")

X_row = results.loc[[selected_id], feature_cols]
shap_values = explainer.shap_values(X_row)[0]
pairs = sorted(zip(feature_cols, shap_values), key=lambda pair: abs(pair[1]), reverse=True)
pairs = pairs[:5]

fig, ax = plt.subplots()
names = [pair[0] for pair in pairs]
values = [pair[1] for pair in pairs]
colors = ['red' if value > 0 else 'blue' for value in values]
ax.barh(names, values, color=colors)
ax.set_xlabel("Push towards fraud (red) or away from fraud (blue)")
st.pyplot(fig)

if st.button("Generate plain-English explanation"):
    top_features = '\n'.join(f'{name}: {value:+.4f}' for name, value in pairs)
    probability = row['predicted_proba']
    prompt = f"""A fraud detection model gave a transaction a fraud probability
of {probability:.2%}.

The five factors that most influenced this score were (positive pushes towards fraud,
negative pushes away from fraud):
{top_features}

Write a short, plain-English explanation (2-3 sentences) for a bank fraud analyst.

Requirements:
- Mention each factor by name with its value, and say whether it pushed the score up or down.
- These factors are anonymised transaction attributes. Do not guess or invent what they mean.
- Make clear this describes how the model behaved, not proof that the transaction is fraud.
- Do not use technical machine learning jargon.
"""
    with st.spinner("Generating explanation..."):
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role': 'user', 'content': prompt}],
        )
    st.success(response.choices[0].message.content)
