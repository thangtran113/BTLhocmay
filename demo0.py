import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Hàm huấn luyện mô hình
@st.cache_resource
def train_models():
    df = pd.read_csv('Gold_Price.csv')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Trích xuất đặc trưng từ cột "Date"
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_year'] = df['Date'].dt.dayofyear

    # Xử lý dữ liệu cho mô hình
    df = df.drop(columns=['Date']).dropna()
    X = df.drop(columns=['Price'])
    y = df['Price']
    
    # Tạo các mô hình
    model_lr = LinearRegression().fit(X, y)
    model_ridge = Ridge().fit(X, y)
    model_nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500).fit(X, y)

    estimators = [
        ('lr', model_lr),
        ('ridge', model_ridge),
        ('nn', model_nn)
    ]
    model_stack = StackingRegressor(estimators=estimators, final_estimator=Ridge()).fit(X, y)

    return {
        'LinearRegression': model_lr,
        'Ridge': model_ridge,
        'NeuralNetwork': model_nn,
        'Stacking': model_stack
    }

# Tải mô hình
models = train_models()

# Giao diện người dùng
st.title("Dự đoán Giá Vàng")

# Đọc và hiển thị bảng CSV
df = pd.read_csv('Gold_Price.csv')
st.subheader("Dữ liệu Giá Vàng")
st.dataframe(df)

# Nhập liệu từ người dùng
if 'input_data' not in st.session_state:
    st.session_state.input_data = {
        'Open': 29435,
        'High': 29598,
        'Low': 29340,
        'Volume': 2390,
        'Chg%': 0.25,
        'Date': pd.Timestamp("2014-01-01")  # Giá trị ngày mặc định
    }

# Nhập ngày tháng
input_date = st.date_input("Chọn ngày:", value=st.session_state.input_data['Date'])
input_year = input_date.year
input_month = input_date.month
input_day = input_date.day
input_day_of_week = input_date.weekday()
input_day_of_year = input_date.timetuple().tm_yday

# Nhập các đặc trưng khác
open_price = st.number_input("Nhập giá mở cửa (Open):", value=st.session_state.input_data['Open'])
high_price = st.number_input("Nhập giá cao nhất (High):", value=st.session_state.input_data['High'])
low_price = st.number_input("Nhập giá thấp nhất (Low):", value=st.session_state.input_data['Low'])
volume = st.number_input("Nhập khối lượng (Volume):", value=st.session_state.input_data['Volume'])
chg = st.number_input("Nhập thay đổi (%) (Chg%):", value=st.session_state.input_data['Chg%'])

# Lưu số liệu nhập trước đó
st.session_state.input_data['Open'] = open_price
st.session_state.input_data['High'] = high_price
st.session_state.input_data['Low'] = low_price
st.session_state.input_data['Volume'] = volume
st.session_state.input_data['Chg%'] = chg
st.session_state.input_data['Date'] = input_date

# Dự đoán khi nhấn nút
if st.button("Dự đoán"):
    input_data = [open_price, high_price, low_price, volume, chg, input_year, input_month, input_day, input_day_of_week, input_day_of_year]

    # Dự đoán bằng các mô hình đã huấn luyện
    predictions = {name: model.predict([input_data])[0] for name, model in models.items()}

    # Hiển thị kết quả dự đoán
    st.subheader("Kết quả dự đoán:")
    for name, pred in predictions.items():
        st.write(f"{name}: {pred:.2f}")
