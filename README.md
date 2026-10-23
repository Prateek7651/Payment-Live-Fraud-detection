# 💳 Online Payment Fraud Detection

A machine learning project that detects fraudulent online payment transactions, with an interactive Streamlit app for real-time predictions.

## Overview

This project trains and compares multiple classification models to flag potentially fraudulent transactions from transaction data (payment type, amount, sender/receiver balances, time of day). The final model is deployed as a Streamlit **Live Fraud Monitor**: a simulated fraud-operations screen where transactions stream in, each one is scored by the model, and risky ones raise alerts that an analyst can Approve, Hold or Escalate.

## Dataset

Source: the PaySim synthetic mobile-money dataset (6.36 million transactions, ~470 MB). It was reduced to a small, training-ready file, `fraud_clean_training.csv` (90,343 rows, 10 columns, ~5 MB):

- **No missing values, no duplicates** in the source data
- **Only `TRANSFER` and `CASH_OUT` rows kept**, because all fraud in the data occurs in these two types (`PAYMENT`, `CASH_IN` and `DEBIT` contain 0 fraud)
- **Removed** `nameOrig` and `nameDest` (unique customer IDs that a model would just memorise), `isFlaggedFraud` (set on only 16 rows by a fixed rule) and `step` (replaced by `hour`)
- **Engineered features**
  - `errorBalanceOrig = newbalanceOrig + amount − oldbalanceOrg`
  - `errorBalanceDest = oldbalanceDest + amount − newbalanceDest`
  - `hour = step % 24`
  - `isTransfer` (1 = TRANSFER, 0 = CASH_OUT)
- **Class balance:** all 8,213 fraud rows kept, legitimate rows randomly down-sampled to 82,130 (10 legitimate per fraud), so fraud is ~9% of the training file instead of 0.13%

Final columns: `isTransfer`, `hour`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `errorBalanceOrig`, `errorBalanceDest`, `isFraud` (target).

## Approach

1. **Preprocessing** — stratified 80/20 train/test split first, then `StandardScaler` fitted on the training data only (no test-data leakage)
2. **Modeling** — trained and compared three classifiers, all using `class_weight="balanced"`:
   - Logistic Regression
   - Decision Tree
   - Random Forest
3. **Evaluation** — models compared using precision, recall, F1-score and confusion matrices rather than accuracy alone
4. **Model selection** — Random Forest selected as the production model
5. **Deployment** — the trained model and scaler were serialized with `joblib` and served through a Streamlit dashboard (see below)

## Results

Held-out test set (18,069 transactions, 1,643 of them fraud):

| Model | Precision | Recall | F1-score |
|---|---|---|---|
| Logistic Regression | 0.60 | 0.90 | 0.72 |
| Decision Tree | 1.00 | 1.00 | 1.00 |
| **Random Forest** | **1.00** | **1.00** (1,636 of 1,643 caught) | **1.00** |

(Decision Tree: 0.996 / 0.996 / 0.996. Random Forest: 1.000 / 0.996 / 0.998.)

Most important features: `errorBalanceOrig` (~47%), `oldbalanceOrg` (~19%), `newbalanceOrig` (~12%).

**Note on the test set:** because legitimate rows were down-sampled, the test set has far more fraud (9%) than real traffic (~0.3% of TRANSFER/CASH_OUT). Recall is unaffected by this, but precision is optimistic. Checking the Random Forest on 2.68 million legitimate transactions that were not used for training gave a false-positive rate of about 0.007%, which corresponds to a real-world precision of roughly 98% at about 99.6% recall.

## Live Fraud Monitor (dashboard)

The app simulates how a fraud team works day to day:

- **Live feed:** transactions arrive one after another and are scored by the Random Forest in real time
- **KPI cards:** transactions scanned, alerts raised, amount flagged, alerts pending review, amount held/escalated
- **Alert queue:** risky transactions (High / Medium risk) with plain-language risk signals, such as "sender's entire balance was moved out", and **Approve / Hold / Escalate** buttons
- **Charts:** risk score of the latest transactions and alerts by payment type
- **Controls:** pause/resume, speed, alert sensitivity (threshold) and reset
- **Analyst decision log** and a **model scorecard** (caught / missed / false alarms)
- **Manual Check tab:** score a single transaction by hand

The stream comes from `demo_transactions.csv` (20,000 transactions, 2% fraud). It contains only transactions the model never saw during training. It is a simulation, not a real bank feed, and its fraud rate is higher than real traffic so that alerts appear often enough to demo. The "risk signals" are simple rule checks shown to help the analyst; the risk score itself comes from the model.

## Limitations

- Trained on **synthetic** PaySim data; real-world fraud patterns may differ
- Only `TRANSFER` and `CASH_OUT` transactions are supported (other payment types have no fraud in the data)
- The model relies heavily on the balance-mismatch features, so it expects origin and destination balances before and after the transaction

## Tech Stack

- **Language:** Python
- **Data & ML:** pandas, NumPy, scikit-learn, Matplotlib, Seaborn
- **Model persistence:** joblib
- **Deployment:** Streamlit

## Project Structure

```
├── Train_model.ipynb          # Data exploration, preprocessing, model training & evaluation
├── fraud_clean_training.csv   # Cleaned, reduced training dataset
├── app.py                     # Streamlit Live Fraud Monitor dashboard
├── demo_transactions.csv      # Unseen transactions used for the live demo stream
├── Fraud_Detection_Model.pkl  # Trained Random Forest model
├── scaler.pkl                 # Fitted StandardScaler
└── requirements.txt           # Project dependencies
```

## Running Locally

```bash
# Clone the repository
git clone https://github.com/Prateek7651/Payment-Fraud-Detection.git
cd Payment-Fraud-Detection

# Install dependencies
pip install -r requirements.txt

# (Optional) retrain the model - regenerates Fraud_Detection_Model.pkl and scaler.pkl
jupyter notebook Train_model.ipynb

# Launch the app
streamlit run app.py
```

## Usage

1. Run `streamlit run app.py` - the live feed starts automatically
2. Watch the KPI cards and the live feed; when a transaction looks risky it appears in the **Alert queue**
3. Review the risk signals and click **Approve**, **Hold** or **Escalate**
4. Use the sidebar to pause, change the speed, adjust alert sensitivity, or reset the demo
5. Use the **Manual Check** tab to score one transaction by hand

## Author

**Prateek Rai**
[GitHub](https://github.com/Prateek7651) · [LinkedIn](https://www.linkedin.com/in/prateek-rai-9b2715285/)
