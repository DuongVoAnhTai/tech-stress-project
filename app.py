import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="AI Mental Health & Stress Analytics",
    page_icon="🧠",
    layout="wide"
)

# --- CSS TÙY CHỈNH (GIAO DIỆN ĐẸP) ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. LOAD MODELS & RESOURCES ---
@st.cache_resource
def load_resources():
    # Đường dẫn folder chứa model (đổi tên nếu bạn giải nén ra tên khác)
    folder = 'models_mental_health_analysis' 
    
    try:
        data = {
            'rf': joblib.load(os.path.join(folder, 'random_forest.pkl')),
            'dt': joblib.load(os.path.join(folder, 'decision_tree.pkl')),
            'kmeans': joblib.load(os.path.join(folder, 'kmeans_model.pkl')),
            'scaler': joblib.load(os.path.join(folder, 'scaler.pkl')),
            'features': joblib.load(os.path.join(folder, 'features.pkl')),
            'cluster_info': joblib.load(os.path.join(folder, 'cluster_info.pkl'))
        }
        return data
    except Exception as e:
        st.error(f"⚠️ LỖI: Không tìm thấy file model. Hãy đảm bảo bạn đã giải nén folder '{folder}' vào cùng thư mục với app.py")
        st.error(f"Chi tiết lỗi: {e}")
        return None

resources = load_resources()

# --- 2. HÀM TÍNH TOÁN MENTAL HEALTH FEATURES (LOGIC GIỐNG FILE TRAIN) ---
def calculate_features(raw_input):
    """
    Tính toán các chỉ số phái sinh từ dữ liệu thô nhập vào
    """
    # 1. Health Score (Giả định = mental_health_score)
    health_score = float(raw_input['mental_health_score'])
    
    # 2. Sleep Health Index
    sleep_health_index = (
        (raw_input['sleep_quality'] / 5 * 50) + 
        (np.clip(raw_input['sleep_duration_hours'], 0, 10) / 10 * 50)
    )
    
    # 3. Emotional Balance
    emotional_balance = raw_input['mood_rating'] * 10
    
    # 4. Overall Wellness
    overall_wellness = (health_score + sleep_health_index + emotional_balance) / 3
    
    # 5. Digital Stress Score
    digital_stress_score = (
        (raw_input['daily_screen_time_hours'] / 24 * 40) +
        (np.clip(raw_input['social_media_hours'], 0, 10) / 10 * 30) +
        (np.clip(raw_input['phone_usage_hours'], 0, 10) / 10 * 30)
    )
    
    # 6. Work-Life Balance
    work_life_balance = 100 - (np.clip(raw_input['work_related_hours'], 0, 16) / 16 * 100)
    
    # Trả về dictionary chứa tất cả features (đúng tên cột lúc train)
    features_dict = {
        'age': raw_input['age'],
        'gender': raw_input['gender'],
        'daily_screen_time_hours': raw_input['daily_screen_time_hours'],
        'sleep_duration_hours': raw_input['sleep_duration_hours'],
        'social_media_hours': raw_input['social_media_hours'],
        'work_related_hours': raw_input['work_related_hours'],
        'gaming_hours': raw_input['gaming_hours'],
        'phone_usage_hours': raw_input['phone_usage_hours'],
        'laptop_usage_hours': raw_input['laptop_usage_hours'],
        'sleep_quality': raw_input['sleep_quality'],
        'health_score': health_score,
        # New features
        'sleep_health_index': sleep_health_index,
        'emotional_balance': emotional_balance,
        'overall_wellness': overall_wellness,
        'digital_stress_score': digital_stress_score,
        'work_life_balance': work_life_balance
    }
    return features_dict

# --- 3. GIAO DIỆN CHÍNH ---

if resources:
    # Sidebar: Nhập liệu
    with st.sidebar:
        st.title("🔧 Thông số đầu vào")
        st.write("Nhập thông tin hành vi & sức khỏe:")
        
        # Chọn thuật toán
        algo_choice = st.selectbox("Thuật toán dự đoán:", ["Random Forest (Khuyên dùng)", "Decision Tree"])
        
        # Nhóm 1: Thông tin cơ bản
        with st.expander("👤 Thông tin cá nhân", expanded=True):
            age = st.slider("Tuổi", 10, 80, 25)
            gender = st.selectbox("Giới tính", ["Male", "Female", "Other"])
            # Mapping gender giống file train
            gender_val = 0 if gender == "Male" else 1 if gender == "Female" else 2
            
            # Input mới cần thiết cho logic tính toán
            mental_score = st.slider("Điểm sức khỏe tinh thần tự đánh giá (0-100)", 0, 100, 70, help="Bạn cảm thấy sức khỏe tinh thần mình thế nào?")
            mood = st.slider("Chấm điểm tâm trạng hôm nay (1-10)", 1, 10, 7)
        
        # Nhóm 2: Công nghệ
        with st.expander("📱 Thói quen Công nghệ", expanded=True):
            screen_time = st.number_input("Tổng giờ dùng màn hình/ngày", 0.0, 24.0, 6.0)
            phone_time = st.number_input("Giờ dùng điện thoại", 0.0, 24.0, 3.0)
            social_time = st.number_input("Giờ mạng xã hội", 0.0, 24.0, 2.0)
            laptop_time = st.number_input("Giờ dùng Laptop", 0.0, 24.0, 4.0)
            game_time = st.number_input("Giờ chơi Game", 0.0, 24.0, 0.5)
            work_time = st.number_input("Giờ làm việc (trên máy)", 0.0, 24.0, 5.0)

        # Nhóm 3: Sức khỏe & Giấc ngủ
        with st.expander("💤 Sức khỏe & Giấc ngủ", expanded=True):
            sleep_time = st.number_input("Thời gian ngủ (giờ)", 0.0, 24.0, 7.0)
            sleep_quality = st.slider("Chất lượng giấc ngủ (1-5)", 1, 5, 4, help="1: Rất tệ, 5: Rất tốt")

        # Nút phân tích
        analyze_btn = st.button("🚀 PHÂN TÍCH NGAY")

    # Màn hình chính
    st.title("🧠 AI Mental Health & Stress Analytics")
    st.markdown("---")
    
    if analyze_btn:
        # 1. Tạo input dictionary thô
        raw_input = {
            'age': age,
            'gender': gender_val,
            'daily_screen_time_hours': screen_time,
            'sleep_duration_hours': sleep_time,
            'social_media_hours': social_time,
            'work_related_hours': work_time,
            'gaming_hours': game_time,
            'phone_usage_hours': phone_time,
            'laptop_usage_hours': laptop_time,
            'sleep_quality': sleep_quality,
            'mental_health_score': mental_score,
            'mood_rating': mood
        }

        # 2. Tính toán các features phái sinh & Tạo DataFrame đúng chuẩn
        processed_features = calculate_features(raw_input)
        
        # Đảm bảo thứ tự cột đúng y hệt lúc train (quan trọng!)
        feature_order = resources['features']
        input_df = pd.DataFrame([processed_features])[feature_order]
        
        # 3. Chuẩn hóa dữ liệu
        input_scaled = resources['scaler'].transform(input_df)
        
        # 4. Dự đoán
        # A. Stress Prediction
        if algo_choice.startswith("Random Forest"):
            model = resources['rf']
            model_name = "Random Forest"
        else:
            model = resources['dt']
            model_name = "Decision Tree"
            
        stress_pred = model.predict(input_scaled)[0]
        stress_map = {0: "Low (Thấp)", 1: "Medium (Trung bình)", 2: "High (Cao)"}
        stress_color = {0: "green", 1: "orange", 2: "red"}
        
        # B. Cluster Prediction
        cluster_pred = resources['kmeans'].predict(input_scaled)[0]
        cluster_info = resources['cluster_info']
        cluster_name = cluster_info['cluster_names'][cluster_pred]

        # --- HIỂN THỊ KẾT QUẢ ---
        
        # Cột 1: Kết quả Stress & Cluster
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.subheader("1. Kết quả Dự báo")
            
            # Card hiển thị Stress
            st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {stress_color[stress_pred]}">
                <h3 style="color: {stress_color[stress_pred]}">Mức độ Stress</h3>
                <h1 style="color: {stress_color[stress_pred]}">{stress_map[stress_pred]}</h1>
                <p style="color: {stress_color[stress_pred]}">Dự báo bởi: {model_name}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("") # Spacer
            
            # Card hiển thị Nhóm người dùng (Cluster)
            st.info(f"🏷️ **Hồ sơ định danh:**\n\n**{cluster_name}**")
            
            # Lời khuyên dựa trên Stress
            if stress_pred == 2:
                st.error("🚨 **Cảnh báo:** Các chỉ số cho thấy bạn đang chịu áp lực lớn. Hãy giảm thời gian sử dụng thiết bị và nghỉ ngơi ngay.")
            elif stress_pred == 1:
                st.warning("⚠️ **Lưu ý:** Bạn đang ở mức cân bằng. Hãy chú ý đến Work-Life Balance.")
            else:
                st.success("✅ **Tuyệt vời:** Bạn đang duy trì lối sống lành mạnh. Hãy tiếp tục phát huy!")

        # Cột 2: Radar Chart (Mental Health Profile)
        with col2:
            st.subheader("2. Biểu đồ Sức khỏe Tinh thần (Radar Chart)")
            
            # Lấy các giá trị đã tính toán
            categories = ['Health Score', 'Wellness', 'Sleep Health', 'Emotional', 'Digital Wellness', 'Work-Life']
            
            # Digital Stress càng cao càng tệ -> đảo ngược để vẽ lên biểu đồ (càng to càng tốt)
            digital_wellness = 100 - processed_features['digital_stress_score']
            
            values = [
                processed_features['health_score'],
                processed_features['overall_wellness'],
                processed_features['sleep_health_index'],
                processed_features['emotional_balance'],
                digital_wellness,
                processed_features['work_life_balance']
            ]
            
            # Vẽ biểu đồ Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Chỉ số của bạn',
                line_color='#3498db'
            ))
            
            # Thêm đường tham chiếu (Mức tốt = 70)
            fig.add_trace(go.Scatterpolar(
                r=[70]*6,
                theta=categories,
                name='Mức khuyến nghị',
                line_color='green',
                line_dash='dot'
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=400,
                margin=dict(t=20, b=20, l=40, r=40)
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- PHẦN CHI TIẾT CHỈ SỐ (METRICS) ---
        st.markdown("---")
        st.subheader("3. Chi tiết các chỉ số phân tích (Mental Health Features)")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Wellness", f"{processed_features['overall_wellness']:.1f}/100", 
                  delta="Tốt" if processed_features['overall_wellness']>70 else "-Cần cải thiện")
        m2.metric("Digital Stress", f"{processed_features['digital_stress_score']:.1f}/100", 
                  delta="-Cao" if processed_features['digital_stress_score']>50 else "Ổn", delta_color="inverse")
        m3.metric("Sleep Health", f"{processed_features['sleep_health_index']:.1f}/100")
        m4.metric("Work-Life Balance", f"{processed_features['work_life_balance']:.1f}/100")
        
        with st.expander("ℹ️ Giải thích ý nghĩa các chỉ số"):
            st.write("""
            - **Overall Wellness:** Điểm tổng hợp sức khỏe thể chất và tinh thần.
            - **Digital Stress Score:** Áp lực do sử dụng thiết bị điện tử (tính từ Screen time, Social media...).
            - **Sleep Health Index:** Chỉ số chất lượng giấc ngủ kết hợp thời lượng ngủ.
            - **Emotional Balance:** Mức độ cân bằng cảm xúc dựa trên Mood Rating.
            """)

else:
    st.info("👋 Xin chào! Đang tải dữ liệu mô hình...")