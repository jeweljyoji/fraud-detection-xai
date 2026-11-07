import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

THRESHOLD = 0.3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load(filename):
    return joblib.load(os.path.join(BASE_DIR, filename))


model = load('fraud_model.pkl')
feature_cols = load('feature_cols.pkl')
amount_scaler = load('amount_scaler.pkl')
time_scaler = load('time_scaler.pkl')

app = FastAPI(title="Fraud Detection API")


class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


@app.get("/")
def root():
    return {"status": "Fraud detection API is running"}


@app.post("/predict")
def predict(transaction: Transaction):
    data = transaction.model_dump()

    row = {col: data[col] for col in feature_cols if col.startswith('V')}

    amount_df = pd.DataFrame({'Amount': [data['Amount']]})
    time_df = pd.DataFrame({'Time': [data['Time']]})
    row['Amount_scaled'] = amount_scaler.transform(amount_df)[0][0]
    row['Time_scaled'] = time_scaler.transform(time_df)[0][0]

    X = pd.DataFrame([row])[feature_cols]
    probability = float(model.predict_proba(X)[0][1])

    return {
        "fraud_probability": round(probability, 4),
        "is_fraud": probability >= THRESHOLD,
        "threshold_used": THRESHOLD,
    }
