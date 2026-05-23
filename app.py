import streamlit as st
import pickle
import pandas as pd

with open("clf_model.pkl", "rb") as f:
    clf_model = pickle.load(f)
with open("reg_model.pkl", "rb") as f:
    reg_model = pickle.load(f)

st.set_page_config(page_title="🎫 Ticket Predictor", layout="centered")
st.title("🎫 Customer Support Ticket Predictor")
st.markdown("กรอกข้อมูลเคส แล้วกดปุ่มเพื่อทำนาย")

with st.form("predict_form"):
    st.subheader("📂 ข้อมูลเคส")
    priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
    ticket_type = st.selectbox("Ticket Type", ["Account", "Billing", "Other", "Product", "Technical"])
    channel = st.selectbox("Channel", ["Chat", "Email", "Phone"])

    st.subheader("👤 ข้อมูลพนักงานและลูกค้า")
    avg_response = st.slider("Avg Response (นาที)", 1, 60, 15)
    messages_count = st.slider("Messages Count", 3, 80, 10)
    agent_exp = st.slider("Agent Experience (ปี)", 0.5, 15.0, 3.0, step=0.5)
    customer_tenure = st.slider("Customer Tenure (เดือน)", 1, 119, 36)

    submitted = st.form_submit_button("🔮 ทำนายผลลัพธ์", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([{
        "priority": priority,
        "ticket_type": ticket_type,
        "channel": channel,
        "avg_response_time_minutes": avg_response,
        "messages_count": messages_count,
        "agent_experience_years": agent_exp,
        "customer_tenure_months": customer_tenure,
    }])

    escalated = clf_model.predict(input_df)[0]
    resolution_time = reg_model.predict(input_df)[0]

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        label = "🔴 Escalated" if escalated == 1 else "🟢 Not Escalated"
        st.metric("Escalation Status", label)
    with col2:
        st.metric("Resolution Time", f"{resolution_time:.1f} ชั่วโมง")
