# Credit Card Fraud Detection with Explainable AI

An end-to-end fraud detection system. It scores card transactions for fraud risk, explains each decision with SHAP, turns that explanation into plain English with a large language model, and serves everything through a REST API and an analyst dashboard.

> Built as a portfolio project using the public Kaggle *Credit Card Fraud Detection* dataset. No real customer data is used.

---

## The problem

Card fraud is rare, and that makes it hard to catch. In this dataset only **492 of 284,807 transactions (0.17%)** are fraud. A model that predicts "not fraud" every time would be **99.8% accurate** and completely useless. So this project:

- judges models with **precision, recall, F1 and PR-AUC**, not accuracy;
- treats the **decision threshold** as a business choice: catching more fraud costs more false alarms;
- explains **why** each transaction was flagged, because a fraud analyst needs to act on the alert and a regulator may need it justified.

## What it does

| Stage | What happens | Tools |
|---|---|---|
| Data preparation | Chronological 80/20 train/test split. Amount and Time are scaled with scalers fitted on training data only, which prevents data leakage. | pandas, scikit-learn |
| Modelling | Logistic Regression baseline compared with XGBoost | scikit-learn, XGBoost |
| Evaluation | Precision, recall, F1, confusion matrix, precision-recall curve and PR-AUC; threshold tuning | scikit-learn, matplotlib |
| Explainability | SHAP force plots for single predictions, including one fraud the model caught and one it missed | SHAP |
| GenAI explanations | The top SHAP factors go into a constrained prompt, which returns a plain-English summary for an analyst | OpenAI API (gpt-4o-mini) |
| Serving | `POST /predict` returns the fraud probability and a flag | FastAPI, pydantic, uvicorn |
| Testing | Automated tests check the endpoints and input validation | pytest |
| Dashboard | Shows flagged transactions with a SHAP chart and a one-click AI explanation | Streamlit |

```text
creditcard.csv ──> notebook (explore, train, evaluate, explain) ──> saved model (.pkl)
                                                                       │
                                     ┌─────────────────────────────────┼──────────────────┐
                                     ▼                                 ▼                  ▼
                              FastAPI /predict                Streamlit dashboard     pytest tests
                                                                       │
                                                                       ▼
                                                     OpenAI: plain-English explanation
```

## Results

Test set: the most recent 20% of transactions by time (75 fraud cases).

| Model | Threshold | Precision | Recall | F1 | Alerts | Fraud caught (of 75) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.5 | 0.89 | 0.55 | 0.68 | 46 | 41 |
| Logistic Regression | 0.3 | 0.78 | 0.61 | 0.69 | 59 | 46 |
| Logistic Regression | 0.1 | 0.73 | 0.69 | 0.71 | 71 | 52 |
| XGBoost | 0.5 | 0.79 | 0.64 | 0.71 | 61 | 48 |
| **XGBoost (used in API and dashboard)** | **0.3** | **0.78** | **0.65** | **0.71** | **63** | **49** |

| Model | PR-AUC |
|---|---|
| Logistic Regression | **0.74** |
| XGBoost | 0.63 |

Accuracy is misleading here: every model scores about 99.9%.

### Key findings

1. **Neither model wins outright.** At the chosen threshold of 0.3, XGBoost catches slightly more fraud than Logistic Regression (49 vs 46) at the same precision. But Logistic Regression has the higher PR-AUC (0.74 vs 0.63), so it ranks transactions better across thresholds overall. XGBoost's curve dips at low recall, because some of the transactions it was most confident about were false alarms.
2. **The threshold is a business decision.** For Logistic Regression, lowering the threshold from 0.5 to 0.3 catches 5 more frauds at the cost of 8 more false alarms. XGBoost's scores sit almost entirely near 0 or 1, so changing its threshold barely changes anything (58 to 72 alerts across thresholds from 0.7 to 0.1). That suggests its probabilities are overconfident (poorly calibrated).
3. **The model has blind spots.** SHAP shows that transaction 229712 was flagged with a probability of 1.00. The main driver was a very low V14 (−9.15), backed up by V4, V12 and V10. Transaction 231978 was also fraud but scored 0.00. Its V14 (−3.69) wasn't extreme enough, so the feature that usually signals fraud actually pushed its score down, and nothing else outweighed it. At the chosen threshold, 26 of the 75 frauds are still missed.

## Explainability and responsible AI

- **SHAP** splits every prediction into each feature's push towards or away from fraud, measured in probability units.
- **The GenAI prompt is constrained** to reduce hallucination. The model must name each factor with its value, must **not** invent meanings for the anonymised features (V1–V28), and must say the explanation describes model behaviour, not proof of fraud.
- **Secrets stay private:** the OpenAI API key lives in a `.env` file that is excluded from git.

## Project structure

```text
├── data/
│   └── creditcard.csv          # download from Kaggle (not included)
├── notebooks/
│   └── exploration.ipynb       # exploration, training, evaluation, SHAP, GenAI
├── src/
│   ├── main.py                 # FastAPI service
│   ├── test_api.py             # pytest tests
│   ├── dashboard.py            # Streamlit dashboard
│   └── *.pkl                   # saved model, feature list and scalers
├── requirements.txt
└── .gitignore
```

## How to run it

**1. Set up**

```bash
git clone https://github.com/jeweljyoji/fraud-detection-xai.git
cd fraud-detection-xai
python -m venv venv
venv\Scripts\activate            # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) into `data/`. For the AI explanations, copy `.env.example` to `.env` and add your `OPENAI_API_KEY`.

**2. Train the model.** Run every cell in `notebooks/exploration.ipynb`. This saves the model files into `src/` and the predictions into `data/dashboard_results.csv`.

**3. Start the API**

```bash
cd src
uvicorn main:app --reload
```

Open <http://127.0.0.1:8000/docs> to try `POST /predict` interactively. The response looks like this:

```json
{ "fraud_probability": 0.9731, "is_fraud": true, "threshold_used": 0.3 }
```

**4. Run the tests**

```bash
cd src
pytest -v
```

**5. Start the dashboard**

```bash
cd src
streamlit run dashboard.py
```

## Limitations and next steps

- **Threshold chosen on the test set.** A separate validation set would give a fairer estimate.
- **XGBoost is untuned and overconfident.** XGBoost is used in the API because it catches the most fraud at the chosen threshold and works directly with SHAP's fast tree explainer. However, Logistic Regression's higher PR-AUC makes it a serious contender. The next steps are hyperparameter tuning, `scale_pos_weight` for the class imbalance, and probability calibration.
- **Anonymised features.** V1–V28 are PCA components, so explanations can't be tied to real-world meaning such as merchant or location.
- **The dataset is old** (European cardholders, 2013, two days of data), so real fraud patterns drift over time. A production system would need drift monitoring and regular retraining.
- **Not deployed.** Next steps are containerising with Docker and adding CI with GitHub Actions to run the tests on every push.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), released by the Machine Learning Group at Université Libre de Bruxelles (ULB) and Worldline. It contains anonymised European card transactions from September 2013.
