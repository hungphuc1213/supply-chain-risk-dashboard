# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from google.cloud import bigquery
import warnings
warnings.filterwarnings("ignore")

# ========================= CONFIG =========================
st.set_page_config(page_title="Cảnh báo Rủi ro Đơn hàng", layout="wide")
st.title("HỆ THỐNG DỰ BÁO LỢI NHUẬN & CẢNH BÁO RỦI RO")
st.markdown("**Dự án Supply Chain - Nhóm Capstone 2025**")

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("Quản lý Model")
    if st.button("Đào tạo lại Model từ BigQuery", type="primary"):
        with st.spinner("Đang kết nối BigQuery và huấn luyện model (3-5 phút)..."):
            try:
                # Dùng secrets từ Streamlit Cloud
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcp.json"
                with open("/tmp/gcp.json", "w") as f:
                    f.write(st.secrets["gcp_service_account"])

                client = bigquery.Client(project="cap2-476009")

                # Query dữ liệu
                queries = {
                    "customer": "SELECT * FROM `cap2-476009.cap2.Dim_Customer`",
                    "order": "SELECT * FROM `cap2-476009.cap2.Dim_Order`",
                    "product": "SELECT * FROM `cap2-476009.cap2.Dim_Product`",
                    "time": "SELECT * FROM `cap2-476009.cap2.Dim_Time`",
                    "fact": "SELECT * FROM `cap2-476009.cap2.Fact_Sales`"
                }
                dfs = {name: client.query(q).to_dataframe() for name, q in queries.items()}

                # Merge
                data = dfs["fact"].merge(dfs["customer"], on="Customer_ID", how="left")\
                                  .merge(dfs["order"], on="Order_ID", how="left")\
                                  .merge(dfs["product"], on="Product_ID", how="left")\
                                  .merge(dfs["time"], on="TimeID", how="left")

                # Feature Engineering (gọn nhất có thể)
                data['shipping_delay_days'] = (data['shipping_date_DateOrders'] - data['order_date_DateOrders']).dt.days
                data['order_hour'] = data['order_date_DateOrders'].dt.hour
                data['is_weekend'] = data['order_date_DateOrders'].dt.dayofweek.isin([5,6]).astype(int)
                data['customer_total_orders'] = data.groupby('Customer_ID')['Order_ID'].transform('count')
                data['is_new_customer'] = (data['customer_total_orders'] == 1).astype(int)

                # Risk Level (rule-based + score)
                def get_risk(row):
                    if row['Order_Status'] in ['SUSPECTED_FRAUD', 'CANCELED']: return 'High Risk'
                    if row['Order_Item_Discount_Rate'] > 0.20: return 'High Risk'
                    if row['shipping_delay_days'] > 7: return 'High Risk'
                    if row['Order_Profit_Per_Order'] < -50: return 'High Risk'
                    if row['Order_Status'] in ['PENDING', 'ON_HOLD'] or row['shipping_delay_days'] > 4: return 'Medium Risk'
                    return 'Low Risk'
                data['risk_level'] = data.apply(get_risk, axis=1)

                # === TRAIN PROFIT MODEL ===
                from xgboost import XGBRegressor
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler

                profit_cols = ['Order_Item_Quantity','Product_Price','Order_Item_Discount_Rate',
                               'shipping_delay_days','order_hour','is_weekend','customer_total_orders']
                cat_cols = ['Type','Market','Order_Region','Category_Name']
                X = pd.get_dummies(data[profit_cols + cat_cols], drop_first=True)
                y = data['Order_Profit_Per_Order']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                profit_model = XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.05, random_state=42)
                profit_model.fit(X_train_s, y_train)

                joblib.dump(profit_model, "models/profit_model.pkl")
                joblib.dump(scaler, "models/profit_scaler.pkl")
                joblib.dump(X.columns.tolist(), "models/profit_columns.pkl")

                # === TRAIN RISK MODEL ===
                from sklearn.ensemble import RandomForestClassifier
                risk_cols = profit_cols + cat_cols
                X_r = pd.get_dummies(data[risk_cols], drop_first=True)
                y_r = data['risk_level']

                risk_model = RandomForestClassifier(n_estimators=500, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
                risk_model.fit(X_r, y_r)

                joblib.dump(risk_model, "models/risk_model_final.pkl")
                joblib.dump(X_r.columns.tolist(), "models/risk_columns.pkl")

                st.success("Đào tạo thành công! Model đã được cập nhật.")
                st.balloons()

            except Exception as e:
                st.error(f"Lỗi: {str(e)}")

# ========================= LOAD MODEL =========================
@st.cache_resource
def load_models():
    try:
        profit_model = joblib.load("models/profit_model.pkl")
        scaler = joblib.load("models/profit_scaler.pkl")
        profit_cols = joblib.load("models/profit_columns.pkl")
        risk_model = joblib.load("models/risk_model_final.pkl")
        risk_cols = joblib.load("models/risk_columns.pkl")
        return profit_model, scaler, profit_cols, risk_model, risk_cols
    except:
        st.error("Chưa có model! Vui lòng bấm nút 'Đào tạo lại Model' ở sidebar.")
        st.stop()

profit_model, scaler, profit_cols, risk_model, risk_cols = load_models()

# ========================= PREDICTION FORM =========================
st.markdown("### Nhập thông tin đơn hàng mới")
col1, col2 = st.columns(2)
with col1:
    quantity = st.number_input("Số lượng sản phẩm", 1, 100, 1)
    price = st.number_input("Giá sản phẩm ($)", 10, 10000, 250)
    discount = st.slider("Tỷ lệ giảm giá (%)", 0, 50, 8) / 100
    delay = st.number_input("Số ngày giao chậm", 0, 20, 0)
    payment = st.selectbox("Hình thức thanh toán", ["DEBIT", "CASH", "PAYMENT", "TRANSFER"])

with col2:
    market = st.selectbox("Thị trường", ["Pacific Asia", "Europe", "LATAM", "US", "Africa"])
    region = st.selectbox("Khu vực", ["Southeast Asia", "Western Europe", "Central America", "South America", "Eastern Asia"])
    category = st.text_input("Danh mục sản phẩm", "Cleats")
    prev_orders = st.number_input("Số đơn hàng cũ của khách", 0, 500, 3)
    hour = st.slider("Giờ đặt hàng (0-23)", 0, 23, 14)
    weekend = 1 if st.checkbox("Cuối tuần?", value=False) else 0

if st.button("DỰ ĐOÁN NGAY", type="primary"):
    input_data = {
        'Order_Item_Quantity': quantity,
        'Product_Price': price,
        'Order_Item_Discount_Rate': discount,
        'shipping_delay_days': delay,
        'order_hour': hour,
        'is_weekend': weekend,
        'customer_total_orders': prev_orders,
        'Type': payment,
        'Market': market,
        'Order_Region': region,
        'Category_Name': category
    }

    df = pd.DataFrame([input_data])
    df_enc = pd.get_dummies(df, drop_first=True)

    # Profit
    df_p = df_enc.reindex(columns=profit_cols, fill_value=0)
    profit_pred = profit_model.predict(scaler.transform(df_p))[0]

    # Risk
    df_r = df_enc.reindex(columns=risk_cols, fill_value=0)
    risk_pred = risk_model.predict(df_r)[0]
    risk_prob = risk_model.predict_proba(df_r)[0].max()

    col1, col2 = st.columns(2)
    with col1:
        if profit_pred > 0:
            st.metric("Lợi nhuận dự kiến", f"${profit_pred:,.0f}", delta=None)
        else:
            st.metric("Dự kiến lỗ", f"${abs(profit_pred):,.0f}", delta=None)

    with col2:
        color = {"High Risk": "red", "Medium Risk": "orange", "Low Risk": "green"}[risk_pred]
        st.markdown(f"**MỨC ĐỘ RỦI RO:** <span style='color:{color};font-size:32px'>{risk_pred}</span>", unsafe_allow_html=True)
        st.progress(risk_prob)
        st.caption(f"Độ tin cậy: {risk_prob:.1%}")

    if risk_pred == "High Risk":
        st.error("CẢNH BÁO: Đơn hàng có nguy cơ cao! Cần kiểm tra thủ công ngay!")
    elif risk_pred == "Medium Risk":
        st.warning("Theo dõi sát sao đơn hàng này.")
    else:
        st.success("Đơn hàng an toàn, xử lý bình thường.")