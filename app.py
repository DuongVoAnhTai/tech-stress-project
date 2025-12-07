import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Dự đoán Stress", page_icon="🧠")

# --- HÀM LOAD MODEL ---
@st.cache_resource
def load_resources():
    path = 'models' # Thư mục chứa file
    try:
        dt = joblib.load(os.path.join(path, 'decision_tree.pkl'))
        rf = joblib.load(os.path.join(path, 'random_forest.pkl'))
        km = joblib.load(os.path.join(path, 'kmeans_model.pkl'))
        scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
        features = joblib.load(os.path.join(path, 'features.pkl'))
        return dt, rf, km, scaler, features
    except Exception as e:
        st.error(f"Lỗi load file: {e}. Hãy chắc chắn bạn đã giải nén vào thư mục 'models'.")
        return None, None, None, None

model_dt, model_rf, model_km, scaler, feature_names = load_resources()

# --- GIAO DIỆN CHÍNH ---
st.title("🧠 Ứng dụng Dự đoán Mức độ Stress")
st.write("Dựa trên thói quen sử dụng công nghệ và sinh hoạt.")

if model_dt and model_rf:
    # --- CỘT BÊN TRÁI: NHẬP LIỆU ---
    with st.sidebar:
        st.header("🔧 Nhập thông tin")
        
        # Chọn thuật toán
        algo = st.selectbox("Chọn thuật toán:", ["Random Forest (Khuyên dùng)", "Decision Tree"])
        
        st.subheader("Thông tin cá nhân")
        age = st.slider("Tuổi", 10, 80, 25)
        gender_txt = st.selectbox("Giới tính", ["Male", "Female", "Other"])
        
        st.subheader("Thói quen Công nghệ (Giờ/Ngày)")
        screen_time = st.number_input("Tổng thời gian dùng màn hình", 0.0, 24.0, 6.0)
        social_time = st.number_input("Thời gian Mạng xã hội", 0.0, 24.0, 2.0)
        work_time = st.number_input("Thời gian làm việc trên máy", 0.0, 24.0, 4.0)
        game_time = st.number_input("Thời gian chơi game", 0.0, 24.0, 1.0)
        
        st.subheader("Sinh hoạt")
        sleep_time = st.number_input("Thời gian ngủ (Giờ)", 0.0, 24.0, 7.0)

    # --- XỬ LÝ DỮ LIỆU ĐẦU VÀO ---
    # 1. Chuyển đổi giới tính sang số (Giống lúc train)
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    gender_val = gender_map[gender_txt]
    
    # 2. Tạo DataFrame từ input (Đúng thứ tự features lúc train)
    # Features gốc: ['age', 'gender', 'daily_screen_time_hours', 'sleep_duration_hours', 
    #                'social_media_hours', 'work_related_hours', 'gaming_hours']
    
    input_data = pd.DataFrame([[
        age, gender_val, screen_time, sleep_time, 
        social_time, work_time, game_time
    ]], columns=feature_names)

    # 3. Chuẩn hóa dữ liệu (Scaling)
    input_scaled = scaler.transform(input_data)

    # --- NÚT DỰ ĐOÁN ---
    if st.button("🚀 Phân tích Hồ sơ", type="primary"):
        
        col1, col2 = st.columns(2)
        
        # === PHẦN 1: DỰ ĐOÁN STRESS (CLASSIFICATION) ===
        with col1:
            st.subheader("1. Dự báo Stress")
            if algo == "Random Forest":
                stress_pred = model_rf.predict(input_scaled)[0]
            else:
                stress_pred = model_dt.predict(input_scaled)[0]
            
            if stress_pred == 0:
                st.success("🟢 Mức độ: THẤP\n\nTâm lý bạn đang rất ổn định.")
            elif stress_pred == 1:
                st.warning("🟡 Mức độ: TRUNG BÌNH\n\nCần chú ý cân bằng lại.")
            else:
                st.error("🔴 Mức độ: CAO\n\nCảnh báo! Bạn cần nghỉ ngơi ngay.")

        # === PHẦN 2: PHÂN CỤM NGƯỜI DÙNG (CLUSTERING) ===
        # <--- ĐÂY LÀ PHẦN MỚI CỦA K-MEANS --->
        with col2:
            st.subheader("2. Phân loại Hồ sơ")
            cluster_id = model_km.predict(input_scaled)[0]
            
            # CHÚ Ý: Bạn cần chỉnh sửa nội dung bên dưới dựa trên kết quả Bước 2
            # Ví dụ: Nếu lúc train bạn thấy Nhóm 0 là dùng nhiều, thì viết content cho Nhóm 0 là "Nghiện Tech"
            
            if cluster_id == 0:
                st.info(f"🏷️ Bạn thuộc nhóm: **Digital Native (Thích công nghệ)**")
                st.write("- Đặc điểm: Người trẻ, thời gian on-screen cao.")
                st.write("- Lời khuyên: Hãy thử 'Digital Detox' vào cuối tuần.")
                
            elif cluster_id == 1:
                st.info(f"🏷️ Bạn thuộc nhóm: **Balanced User (Cân bằng)**")
                st.write("- Đặc điểm: Sử dụng thiết bị vừa phải phục vụ công việc.")
                st.write("- Lời khuyên: Duy trì thói quen hiện tại.")
                
            else:
                st.info(f"🏷️ Bạn thuộc nhóm: **Minimalist (Sống tối giản)**")
                st.write("- Đặc điểm: Ít phụ thuộc vào công nghệ, ngủ đủ giấc.")
                st.write("- Lời khuyên: Hãy chia sẻ lối sống này với người khác!")

        # === PHẦN 3: VISUALIZATION (BIỂU ĐỒ) ===
        st.divider()
        st.subheader("📊 So sánh với mức trung bình")
        # Giả lập số liệu trung bình (hoặc lấy từ data thật)
        chart_data = pd.DataFrame({
            "Chỉ số": ["Màn hình", "Giờ ngủ", "MXH"],
            "Bạn": [screen_time, sleep_time, social_time],
            "Khuyến nghị": [4, 8, 1] # Số liệu giả định
        })
        st.bar_chart(chart_data.set_index("Chỉ số"))

else:
    st.warning("Đang tải model... Vui lòng đợi.")