"""
Live Fraud Monitor
------------------
A simulated "fraud operations" screen:
  * transactions stream in one after another (from demo_transactions.csv,
    which contains rows the model never saw during training)
  * every transaction is scored by the trained Random Forest
  * risky ones raise an alert that an analyst can Approve / Hold / Escalate
  * a Manual Check tab lets you score a single transaction by hand
"""

from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Live Fraud Monitor",
    page_icon="🛡️",
    layout="wide",
)

# Feature order must match the training notebook exactly
FEATURES = [
    "isTransfer",
    "hour",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
]

HIGH_RISK = 0.90      # risk score at/above this is shown as HIGH
FEED_SIZE = 12        # rows shown in the live feed
HISTORY_SIZE = 150    # points shown in the risk chart
QUEUE_SIZE = 4        # alerts shown at once in the action queue


# -------------------------
# Load model, scaler and demo stream (cached)
# -------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("Fraud_Detection_Model.pkl")
    scaler = joblib.load("scaler.pkl")
    demo = pd.read_csv("demo_transactions.csv")
    # Score the whole demo file once, so the live feed is fast
    probs = model.predict_proba(scaler.transform(demo[FEATURES]))[:, 1]
    big_amount = float(demo["amount"].quantile(0.90))
    return model, scaler, demo, probs, big_amount


try:
    MODEL, SCALER, DEMO, PROBS, BIG_AMOUNT = load_assets()
except FileNotFoundError as err:
    st.error(
        f"Missing file: {err.filename}. Keep app.py, Fraud_Detection_Model.pkl, "
        "scaler.pkl and demo_transactions.csv in the same folder."
    )
    st.stop()


# -------------------------
# Helpers
# -------------------------
def money(x):
    return f"{x:,.0f}"


def level(p, threshold):
    if p < threshold:
        return "OK"
    return "High" if p >= HIGH_RISK else "Medium"


def risk_signals(r):
    """Simple rule checks that explain WHY a transaction looks risky.
    (The risk score itself comes from the model; these are shown to help the analyst.)"""
    out = []
    if r["oldbalanceOrg"] > 0 and r["newbalanceOrig"] == 0 and abs(r["amount"] - r["oldbalanceOrg"]) < 0.01:
        out.append("Sender's entire balance was moved out (account emptied)")
    if r["amount"] > 0 and r["newbalanceDest"] == r["oldbalanceDest"]:
        out.append("Receiver's balance did not change after the transfer")
    elif r["errorBalanceDest"] > 0.01 * max(r["amount"], 1):
        out.append("Receiver's balance rose by less than the amount sent")
    if r["oldbalanceDest"] == 0:
        out.append("Receiver account was empty before this transaction")
    if r["hour"] <= 5:
        out.append("Unusual hour (12 am to 5 am)")
    if r["amount"] >= BIG_AMOUNT:
        out.append("Very high amount (top 10% of transactions)")
    return out


# -------------------------
# Session state (the "memory" of the live stream)
# -------------------------
def reset_stream():
    ss = st.session_state
    ss.ptr = 0                    # position in the demo file
    ss.scanned = 0
    ss.feed = []                  # latest transactions
    ss.history = []              # latest risk scores
    ss.alerts = {}                # tx id -> alert record
    ss.decisions = []             # analyst decisions (newest first)
    ss.alert_count = 0
    ss.amount_flagged = 0.0
    ss.by_type = {"TRANSFER": 0, "CASH_OUT": 0}
    ss.truth = {"fraud": 0, "caught": 0, "missed": 0, "false_alarms": 0}


if "ptr" not in st.session_state:
    reset_stream()
st.session_state.setdefault("running", True)
st.session_state.setdefault("speed", 3)
st.session_state.setdefault("threshold", 0.50)


def advance(n):
    """Pull the next n transactions from the stream and score them."""
    ss = st.session_state
    th = ss.threshold
    total = len(DEMO)
    for _ in range(n):
        i = ss.ptr % total
        ss.ptr += 1
        row = DEMO.iloc[i]
        p = float(PROBS[i])
        ss.scanned += 1
        actual = int(row["isFraud"])
        flagged = p >= th

        rec = {
            "num": ss.scanned,
            "id": f"TX-{ss.scanned:06d}",
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": "TRANSFER" if int(row["isTransfer"]) == 1 else "CASH_OUT",
            "sender": str(row["sender"]),
            "receiver": str(row["receiver"]),
            "amount": float(row["amount"]),
            "risk": p,
            "status": level(p, th),
            "signals": [],
            "decision": None,
            "actual": actual,
        }

        if flagged:
            rec["signals"] = risk_signals(row)
            ss.alerts[rec["id"]] = rec
            ss.alert_count += 1
            ss.amount_flagged += rec["amount"]
            ss.by_type[rec["type"]] += 1
            if not actual:
                ss.truth["false_alarms"] += 1
        if actual:
            ss.truth["fraud"] += 1
            ss.truth["caught" if flagged else "missed"] += 1

        ss.feed.insert(0, rec)
        ss.feed = ss.feed[:FEED_SIZE]
        ss.history.append(p)
        ss.history = ss.history[-HISTORY_SIZE:]


def decide(tx_id, decision):
    rec = st.session_state.alerts.get(tx_id)
    if rec and rec["decision"] is None:
        rec["decision"] = decision
        rec["decided_at"] = datetime.now().strftime("%H:%M:%S")
        st.session_state.decisions.insert(0, rec)


# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.title("🛡️ Controls")
    st.toggle("Run live feed", key="running")
    st.slider("Speed (transactions per second)", 1, 15, key="speed")
    st.slider(
        "Alert sensitivity (alert when risk ≥)",
        0.10, 0.95, step=0.05, key="threshold",
        help="Lower = more alerts (catches more fraud but more false alarms). "
             "Higher = fewer alerts.",
    )
    st.button("🔄 Reset demo", on_click=reset_stream, width="stretch")

    st.divider()
    st.subheader("About")
    st.info(
        """
Machine Learning Project

Algorithms compared:
- Logistic Regression
- Decision Tree
- Random Forest

Scores come from the trained Random Forest model.

Fraud only occurs in TRANSFER and CASH_OUT
transactions, so the model is trained for
these two payment types.
"""
    )

# -------------------------
# Header
# -------------------------
status = "🟢 LIVE" if st.session_state.running else "⏸️ PAUSED"
st.title("🛡️ Live Fraud Monitor")
st.caption(
    f"{status}  ·  Simulated transaction stream built from data the model never saw during training."
)

tab_live, tab_manual = st.tabs(["🔴 Live Monitor", "🔍 Manual Check"])


# -------------------------
# Live monitor (auto-refreshes every second while running)
# -------------------------
def live_view():
    ss = st.session_state

    if ss.running:
        advance(ss.speed)

    pending = [r for r in ss.alerts.values() if r["decision"] is None]
    held_amount = sum(r["amount"] for r in ss.decisions if r["decision"] in ("Held", "Escalated"))
    rate = (100 * ss.alert_count / ss.scanned) if ss.scanned else 0.0

    # ---- KPI cards ----
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Transactions scanned", f"{ss.scanned:,}", border=True)
    k2.metric("Alerts raised", f"{ss.alert_count:,}", f"{rate:.1f}% of traffic", delta_color="off", border=True)
    k3.metric("Amount flagged", money(ss.amount_flagged), border=True)
    k4.metric("Pending review", f"{len(pending):,}", border=True)
    k5.metric("Amount held / escalated", money(held_amount), border=True)

    left, right = st.columns([3, 2])

    # ---- Alert queue ----
    with left:
        st.subheader("🚨 Alert queue")
        if not pending:
            st.info("No pending alerts. Suspicious transactions will appear here.")
        queue = sorted(pending, key=lambda r: (-r["risk"], -r["num"]))
        for rec in queue[:QUEUE_SIZE]:
            tag = "🔴 HIGH RISK" if rec["status"] == "High" else "🟠 MEDIUM RISK"
            with st.container(border=True):
                st.markdown(f"**{tag}**  ·  {rec['id']}  ·  {rec['type']}  ·  amount **{money(rec['amount'])}**")
                st.caption(
                    f"{rec['sender']} → {rec['receiver']}  ·  risk score {rec['risk'] * 100:.0f}%  ·  {rec['time']}"
                )
                if rec["signals"]:
                    for s in rec["signals"][:3]:
                        st.markdown(f"- {s}")
                else:
                    st.markdown("- Flagged by the model's overall pattern (no single rule triggered)")
                b1, b2, b3 = st.columns(3)
                b1.button("✅ Approve", key=f"ap_{rec['id']}", on_click=decide, args=(rec["id"], "Approved"), width="stretch")
                b2.button("⏸️ Hold", key=f"ho_{rec['id']}", on_click=decide, args=(rec["id"], "Held"), width="stretch")
                b3.button("🚨 Escalate", key=f"es_{rec['id']}", on_click=decide, args=(rec["id"], "Escalated"), width="stretch")
        if len(queue) > QUEUE_SIZE:
            st.caption(f"+ {len(queue) - QUEUE_SIZE} more pending (highest risk shown first)")

    # ---- Charts ----
    with right:
        st.subheader("📈 Risk score (latest transactions)")
        if ss.history:
            chart = pd.DataFrame({"Risk score": ss.history, "Alert threshold": ss.threshold})
            st.line_chart(chart, height=220)
        else:
            st.caption("Waiting for transactions…")
        st.subheader("Alerts by payment type")
        st.bar_chart(pd.Series(ss.by_type, name="Alerts"), height=160)

    # ---- Live feed ----
    st.subheader("📡 Live transaction feed")
    if ss.feed:
        icon = {"OK": "🟢 OK", "Medium": "🟠 Medium", "High": "🔴 High"}
        table = pd.DataFrame(
            [
                {
                    "Time": r["time"],
                    "ID": r["id"],
                    "Type": r["type"],
                    "Sender → Receiver": f"{r['sender']} → {r['receiver']}",
                    "Amount": money(r["amount"]),
                    "Risk": f"{r['risk'] * 100:.0f}%",
                    "Status": icon[r["status"]],
                }
                for r in ss.feed
            ]
        )
        st.dataframe(table, hide_index=True, width="stretch")

    # ---- Decision log ----
    with st.expander(f"🗂️ Analyst decisions ({len(ss.decisions)})"):
        if ss.decisions:
            log = pd.DataFrame(
                [
                    {
                        "Decided at": r["decided_at"],
                        "ID": r["id"],
                        "Type": r["type"],
                        "Amount": money(r["amount"]),
                        "Risk": f"{r['risk'] * 100:.0f}%",
                        "Decision": r["decision"],
                    }
                    for r in ss.decisions[:10]
                ]
            )
            st.dataframe(log, hide_index=True, width="stretch")
        else:
            st.caption("No decisions yet. Use the buttons on an alert.")

    # ---- Model scorecard (needs true labels, so demo only) ----
    with st.expander("📊 Model scorecard (demo only: uses the true labels)"):
        t = ss.truth
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Real fraud in stream", t["fraud"])
        s2.metric("Caught by model", t["caught"])
        s3.metric("Missed", t["missed"])
        s4.metric("False alarms", t["false_alarms"])
        st.caption(
            "In real life the true label arrives later (for example after a customer dispute), "
            "so this panel exists only because the demo data is labelled. "
            "The demo stream has about 2% fraud; real traffic has far less, so real analysts "
            "would see fewer alerts."
        )


with tab_live:
    st.fragment(live_view, run_every=(1.0 if st.session_state.running else None))()


# -------------------------
# Manual check (single transaction)
# -------------------------
with tab_manual:
    st.subheader("Check a single transaction")
    st.caption("Enter the details of one transaction to see the model's risk score.")

    c1, c2 = st.columns(2)
    with c1:
        payment_type = st.selectbox("Payment Type", ["TRANSFER", "CASH_OUT"], key="m_type")
        hour = st.number_input("Hour of Day (0-23)", min_value=0, max_value=23, value=12, key="m_hour")
        amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0, key="m_amt")
        old_o = st.number_input("Old Balance Origin", min_value=0.0, value=0.0, key="m_oo")
    with c2:
        new_o = st.number_input("New Balance Origin", min_value=0.0, value=0.0, key="m_no")
        old_d = st.number_input("Old Balance Destination", min_value=0.0, value=0.0, key="m_od")
        new_d = st.number_input("New Balance Destination", min_value=0.0, value=0.0, key="m_nd")

    if st.button("🔍 Check Transaction", width="stretch", key="m_go"):
        row = {
            "isTransfer": 1 if payment_type == "TRANSFER" else 0,
            "hour": hour,
            "amount": amount,
            "oldbalanceOrg": old_o,
            "newbalanceOrig": new_o,
            "oldbalanceDest": old_d,
            "newbalanceDest": new_d,
            "errorBalanceOrig": round(new_o + amount - old_o, 2),
            "errorBalanceDest": round(old_d + amount - new_d, 2),
        }
        p = float(MODEL.predict_proba(SCALER.transform(pd.DataFrame([row], columns=FEATURES)))[0][1])

        if p >= 0.5:
            st.error("🚨 Fraudulent transaction suspected")
        else:
            st.success("✅ Looks legitimate")
        st.progress(min(p, 1.0), text=f"Fraud probability: {p * 100:.1f}%")

        sig = risk_signals(row)
        if sig:
            st.markdown("**Risk signals noticed:**")
            for s in sig:
                st.markdown(f"- {s}")
