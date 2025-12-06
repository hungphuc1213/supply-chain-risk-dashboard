# app.py – CHẠY ĐƯỢC 100% TRÊN STREAMLIT CLOUD
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Dự báo Lợi nhuận & Rủi ro", layout="wide")
st.title("HỆ THỐNG DỰ BÁO LỢI NHUẬN & CẢNH BÁO RỦI RO")
st.markdown("**Capstone 2025 – Supply Chain Risk Dashboard**")

# Load model
@st.cache_resource
def load_models():
    profit_model = joblib.load("models/profit_model.pkl")
    risk_model = joblib.load("models/risk_model_final.pkl")
    profit_cols = joblib.load("models/profit_columns.pkl")
    risk_cols = joblib.load("models/risk_columns.pkl")
    return profit_model, risk_model, profit_cols, risk_cols

profit_model, risk_model, profit_cols, risk_cols = load_models()

# Form nhập liệu
col1, col2 = st.columns(2)
with col1:
    quantity = st.number_input("Số lượng", 1, 100, 1)
    discount = st.slider("Giảm giá (%)", 0, 50, 10) / 100
    price = st.number_input("Giá sản phẩm ($)", 10, 10000, 500)
    delay = st.number_input("Ngày giao chậm", 0, 30, 0)
    payment = st.selectbox("Thanh toán", ["DEBIT", "CASH", "PAYMENT", "TRANSFER"])

with col2:
    market = st.selectbox("Thị trường", ["Pacific Asia", "Europe", "LATAM", "US"])
    region = st.text_input("Khu vực", "Southeast Asia")
    category = st.text_input("Danh mục sản phẩm", "Cleats")
    cust_orders = st.number_input("Số đơn cũ của khách", 0, 1000, 1)

if st.button("DỰ ĐOÁN NGAY", type="primary"):
    data = {
        "Order_Item_Quantity": quantity,
        "Order_Item_Discount_Rate": discount,
        "Product_Price": price,
        "shipping_delay_days": delay,
        "Type": payment,
        "Market": market,
        "Category_Name": category,
        "Order_Region": region,
        "customer_total_orders": cust_orders,
        "order_hour": 12,
        "is_weekend": 0
    }

    df = pd.DataFrame([data])
    df_enc = pd.get_dummies(df, drop_first=True)

    # Profit
    X_p = df_enc.reindex(columns=profit_cols, fill_value=0)
    profit = float(profit_model.predict(X_p)[0])

    # Risk
    X_r = df_enc.reindex(columns=risk_cols, fill_value=0)
    risk = risk_model.predict(X_r)[0]
    prob = risk_model.predict_proba(X_r)[0].max()

    # Alert
    alert = []
    if discount > 0.10: alert.append("Giảm giá cao")
    if payment in ["PAYMENT", "TRANSFER"]: alert.append("Thanh toán online")
    if delay > 5: alert.append("Giao chậm")

    col1, col2 = st.columns(2)
    with col1:
        if profit >= 0:
            st.success(f"**Lợi nhuận dự kiến: ${profit:,.0f}**")
        else:
            st.error(f"**Dự kiến lỗ: ${abs(profit):,.0f}**")

    with col2:
        color = "red" if "High" in risk else "orange" if "Medium" in risk else "green"
        st.markdown(f"**RỦI RO:** <span style='color:{color};font-size:30px'>{risk}</span>", unsafe_allow_html=True)
        st.progress(prob)
        if alert:
            st.warning(" | ".join(alert))
        else:
            st.success("Đơn hàng an toàn")

    st.balloons()