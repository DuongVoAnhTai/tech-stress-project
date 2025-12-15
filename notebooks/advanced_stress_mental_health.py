"""
=============================================================================
PHÂN TÍCH VÀ DỰ ĐOÁN MỨC ĐỘ STRESS - MENTAL HEALTH & DENDROGRAM VERSION
Thêm Mental Health Features + Interactive Hierarchical Clustering Dendrogram
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.decomposition import PCA
from math import pi
from google.colab import files
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Cấu hình hiển thị
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("="*80)
print("🚀 CHƯƠNG TRÌNH PHÂN TÍCH STRESS - MENTAL HEALTH ANALYSIS VERSION")
print("   📌 Phân tích chuyên sâu Mental Health với 3D Interactive Visualization")
print("="*80)

# =============================================================================
# BƯỚC 1: TẢI DỮ LIỆU
# =============================================================================
print("\n📂 BƯỚC 1: Upload file data.csv...")
uploaded = files.upload()
df = pd.read_csv('data.csv')

print(f"✅ Đã tải {len(df)} người dùng với {len(df.columns)} đặc trưng")
print(f"\n📋 Các cột có sẵn trong data:")
print(f"   {', '.join(df.columns[:10])}...")

# =============================================================================
# BƯỚC 2: TÍNH TOÁN HEALTH_SCORE VÀ MENTAL HEALTH FEATURES
# =============================================================================
print("\n" + "="*80)
print("🔧 BƯỚC 2: TÍNH TOÁN HEALTH_SCORE VÀ MENTAL HEALTH FEATURES")
print("="*80)

# Kiểm tra và xử lý missing values
print(f"\n🔍 Kiểm tra missing values...")
missing_cols = df.isnull().sum()
if missing_cols.sum() > 0:
    print(f"   ⚠️ Tìm thấy {missing_cols.sum()} missing values")
    print(missing_cols[missing_cols > 0])
    # Fill missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
    print("   ✅ Đã xử lý missing values bằng median")
else:
    print("   ✅ Không có missing values")

# 1. Health Score = Mental Health Score (đơn giản)
print("\n💡 Health Score:")
print("   health_score = mental_health_score (0-100)")

df['health_score'] = df['mental_health_score'].astype(float)  # ← Đảm bảo là float

# 2. Mental Health Features mới
print("\n💚 Các chỉ số Mental Health mới:")
print("   1. sleep_health_index = (sleep_quality/5 * 50 + sleep_duration_hours/10 * 50)")
print("   2. emotional_balance = mood_rating * 10 (0-100)")
print("   3. overall_wellness = (health_score + sleep_health_index + emotional_balance) / 3")

# Sleep Health Index (chất lượng giấc ngủ + thời lượng)
df['sleep_health_index'] = (
    (df['sleep_quality'].astype(float) / 5 * 50) +          # Chất lượng giấc ngủ
    (np.clip(df['sleep_duration_hours'].astype(float), 0, 10) / 10 * 50)  # Thời lượng (0-10h)
)

# Emotional Balance (cân bằng cảm xúc)
df['emotional_balance'] = df['mood_rating'].astype(float) * 10  # 0-10 → 0-100

# Overall Wellness Score (tổng hợp)
df['overall_wellness'] = (
    df['health_score'].astype(float) +           
    df['sleep_health_index'].astype(float) + 
    df['emotional_balance'].astype(float)
) / 3

# 3. Digital Wellness Index (sức khỏe kỹ thuật số - ảnh hưởng của công nghệ)
print("   4. digital_stress_score = (screen_time/24 * 40 + social_media/10 * 30 + phone_usage/10 * 30)")

df['digital_stress_score'] = (
    (df['daily_screen_time_hours'].astype(float) / 24 * 40) +  # Tổng screen time
    (np.clip(df['social_media_hours'].astype(float), 0, 10) / 10 * 30) +  # Social media
    (np.clip(df['phone_usage_hours'].astype(float), 0, 10) / 10 * 30)     # Phone usage
)

# 4. Work-Life Balance Score
print("   5. work_life_balance = 100 - (work_hours/16 * 100)")

df['work_life_balance'] = 100 - (np.clip(df['work_related_hours'].astype(float), 0, 16) / 16 * 100)

print(f"\n✅ Đã tính toán {5} chỉ số Mental Health (health_score = mental_health_score)")

# Thống kê các chỉ số Mental Health
print("\n📊 Thống kê các chỉ số Mental Health:")
mental_cols = ['health_score', 'sleep_health_index', 
               'emotional_balance', 'overall_wellness', 'digital_stress_score', 
               'work_life_balance']

stats_df = df[mental_cols].describe().T[['min', 'max', 'mean', 'std']]
print(stats_df.round(2))

# Hiển thị phân bố Mental Health Features
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

mental_features = [
    ('health_score', 'Health Score (Mental Health)', '#e74c3c'),
    ('overall_wellness', 'Overall Wellness', '#2ecc71'),
    ('sleep_health_index', 'Sleep Health Index', '#9b59b6'),
    ('emotional_balance', 'Emotional Balance', '#f39c12'),
    ('digital_stress_score', 'Digital Stress', '#e67e22'),
    ('work_life_balance', 'Work-Life Balance', '#1abc9c')
]

for idx, (col, title, color) in enumerate(mental_features):
    ax = axes[idx]
    ax.hist(df[col], bins=30, color=color, edgecolor='black', alpha=0.7)
    ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {df[col].mean():.1f}')
    ax.set_xlabel(title, fontsize=10, fontweight='bold')
    ax.set_ylabel('Số lượng', fontsize=10)
    ax.set_title(f'📊 {title}', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# BƯỚC 3: CHỌN FEATURES (Mở rộng thêm Mental Health)
# =============================================================================
print("\n" + "="*80)
print("📋 BƯỚC 3: CHỌN FEATURES CHO PHÂN TÍCH")
print("="*80)

# Mở rộng thêm 5 features Mental Health → Tổng 16 features
selected_features = [
    # Original features (11)
    'age', 
    'gender', 
    'daily_screen_time_hours', 
    'sleep_duration_hours', 
    'social_media_hours', 
    'work_related_hours', 
    'gaming_hours',
    'phone_usage_hours', 
    'laptop_usage_hours',
    'sleep_quality',
    'health_score',              # ← health_score = mental_health_score
    # New Mental Health features (5)
    'sleep_health_index',       # ← Mới
    'emotional_balance',        # ← Mới
    'overall_wellness',         # ← Mới
    'digital_stress_score',     # ← Mới
    'work_life_balance'         # ← Mới
]

target = 'stress_level'

X = df[selected_features].copy()
y = df[target].copy()

# Đảm bảo tất cả features là numeric
print(f"\n🔍 Kiểm tra kiểu dữ liệu của features...")
for col in X.columns:
    if X[col].dtype == 'object':
        print(f"   ⚠️ {col} đang là object, chuyển sang numeric...")
        X[col] = pd.to_numeric(X[col], errors='coerce')
    # Fill any NaN values
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].median(), inplace=True)

# Kiểm tra variance
print(f"\n🔍 Kiểm tra variance của features...")
zero_var_features = []
for col in X.columns:
    if X[col].var() == 0:
        zero_var_features.append(col)
        print(f"   ⚠️ {col} có variance = 0 (tất cả giá trị giống nhau)")

if zero_var_features:
    print(f"\n   ⚠️ Loại bỏ {len(zero_var_features)} features có variance = 0...")
    X = X.drop(columns=zero_var_features)
    selected_features = [f for f in selected_features if f not in zero_var_features]
    print(f"   ✅ Còn lại {len(selected_features)} features")
else:
    print("   ✅ Tất cả features có variance > 0")
        
print(f"\n✅ Tất cả features đã sẵn sàng")

print(f"\n✅ Đã chọn {len(selected_features)} features:")
print("\n🔹 Original Features (11):")
for i, feat in enumerate(selected_features[:11], 1):
    if feat == 'health_score':
        print(f"   {i:2d}. {feat} ⭐ (= mental_health_score)")
    else:
        print(f"   {i:2d}. {feat}")

print("\n💚 Mental Health Features (5 - MỚI):")
for i, feat in enumerate(selected_features[11:], 12):
    print(f"   {i:2d}. {feat} 🆕")

# Correlation với stress_level
print("\n📊 Mức độ tương quan với Stress Level:")
correlations = X.copy()
correlations['stress_level'] = y
corr_with_stress = correlations.corr()['stress_level'].drop('stress_level').sort_values(ascending=False)

# Loại bỏ NaN values
corr_with_stress = corr_with_stress.dropna()

print("-" * 70)
for feat, corr in corr_with_stress.items():
    bar = '█' * int(abs(corr) * 50)
    sign = '+' if corr > 0 else '-'
    marker = '🆕' if feat in selected_features[11:] else '  '
    print(f"{marker} {feat:<30} {sign} {bar} {corr:.3f}")
print("-" * 70)

# Heatmap correlation cho Mental Health features
print("\n📊 Correlation Heatmap - Mental Health Features:")
mental_health_cols = selected_features[11:] + ['stress_level']
corr_matrix = df[mental_health_cols].corr()

# Fill NaN values trong correlation matrix
corr_matrix = corr_matrix.fillna(0)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('🔥 Correlation Matrix - Mental Health Features', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# =============================================================================
# BƯỚC 4: TIỀN XỬ LÝ DỮ LIỆU
# =============================================================================
print("\n" + "="*80)
print("⚙️ BƯỚC 4: TIỀN XỬ LÝ DỮ LIỆU")
print("="*80)

# Gom nhóm Stress Level thành 3 mức
def bin_stress(val):
    if val <= 3: return 0      # Low
    elif val <= 6: return 1    # Medium
    else: return 2             # High

y = y.apply(bin_stress)
target_names = ['Low', 'Medium', 'High']
print("\n📊 Phân bố nhãn Stress:")
stress_dist = y.value_counts().sort_index()
for idx, count in stress_dist.items():
    print(f"   {target_names[idx]}: {count} người ({count/len(y)*100:.1f}%)")

# Mã hóa giới tính
le_gender = LabelEncoder()
X['gender'] = le_gender.fit_transform(X['gender'].astype(str))
print("\n✅ Đã mã hóa giới tính:")
gender_mapping = {i: label for i, label in enumerate(le_gender.classes_)}
for code, label in gender_mapping.items():
    print(f"   {label} → {code}")

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✅ Chia dữ liệu:")
print(f"   Train: {len(X_train)} người ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Test:  {len(X_test)} người ({len(X_test)/len(X)*100:.1f}%)")

# Chuẩn hóa
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\n✅ Đã chuẩn hóa dữ liệu về khoảng [0, 1]")

# =============================================================================
# BƯỚC 5: TÌM SỐ CỤM TỐI ƯU
# =============================================================================
print("\n" + "="*80)
print("📈 BƯỚC 5: TÌM SỐ CỤM TỐI ƯU")
print("="*80)

print("\n🔍 Đang thử nghiệm từ 3 đến 10 cụm...")
inertias = []
silhouette_scores_list = []
K_range = range(3, 11)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans_temp.fit(X_train_scaled)
    inertias.append(kmeans_temp.inertia_)
    silhouette_scores_list.append(silhouette_score(X_train_scaled, kmeans_temp.labels_))
    print(f"   K={k}: Inertia={kmeans_temp.inertia_:.2f}, Silhouette={silhouette_scores_list[-1]:.4f}")

# Tìm K tối ưu
optimal_k = K_range[np.argmax(silhouette_scores_list)]
print(f"\n✅ Số cụm tối ưu: {optimal_k} (Silhouette Score: {max(silhouette_scores_list):.4f})")

# Vẽ biểu đồ Elbow + Silhouette
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Elbow Method
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Inertia', fontsize=12, fontweight='bold')
axes[0].set_title('📉 Elbow Method', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Silhouette Score
axes[1].plot(K_range, silhouette_scores_list, 'go-', linewidth=2, markersize=8)
axes[1].axvline(optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
axes[1].axhline(max(silhouette_scores_list), color='orange', linestyle='--', 
                linewidth=1.5, alpha=0.7, label=f'Max: {max(silhouette_scores_list):.4f}')
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
axes[1].set_title('📊 Silhouette Score', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# BƯỚC 6: K-MEANS CLUSTERING
# =============================================================================
print("\n" + "="*80)
print(f"🎯 BƯỚC 6: K-MEANS CLUSTERING (K={optimal_k})")
print("="*80)

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
kmeans.fit(X_train_scaled)
clusters_train = kmeans.labels_
clusters_test = kmeans.predict(X_test_scaled)

print(f"\n✅ Đã phân cụm {len(X_train)} người vào {optimal_k} nhóm")
print(f"\n📊 Phân bố các cụm:")
for i in range(optimal_k):
    count = np.sum(clusters_train == i)
    print(f"   Cluster {i}: {count} người ({count/len(clusters_train)*100:.1f}%)")

# =============================================================================
# BƯỚC 7: PHÂN TÍCH CLUSTER PROFILES (Có Mental Health)
# =============================================================================
print("\n" + "="*80)
print("📊 BƯỚC 7: PHÂN TÍCH CLUSTER PROFILES")
print("="*80)

X_train_df = pd.DataFrame(X_train, columns=selected_features)
X_train_df['Cluster'] = clusters_train
X_train_df['Stress_Level'] = y_train.values

cluster_profiles = []
cluster_names = []
cluster_to_stress = {}

print("\n📋 Phân tích từng cụm:")
print("="*70)

for i in range(optimal_k):
    cluster_data = X_train_df[X_train_df['Cluster'] == i]
    
    profile = {
        'Cluster': i,
        'Size': len(cluster_data),
        'Percentage': f"{len(cluster_data)/len(X_train_df)*100:.1f}%"
    }
    
    # Thống kê cho từng feature
    for feat in selected_features:
        profile[f'{feat}_mean'] = cluster_data[feat].mean()
        profile[f'{feat}_std'] = cluster_data[feat].std()
    
    # Stress level trội
    dominant_stress = cluster_data['Stress_Level'].mode()[0]
    profile['Dominant_Stress'] = target_names[dominant_stress]
    cluster_to_stress[i] = dominant_stress
    
    # Mental Health Summary
    profile['Health_Score_Avg'] = cluster_data['health_score'].mean()
    profile['Overall_Wellness_Avg'] = cluster_data['overall_wellness'].mean()
    profile['Digital_Stress_Avg'] = cluster_data['digital_stress_score'].mean()
    profile['Sleep_Health_Avg'] = cluster_data['sleep_health_index'].mean()
    profile['Work_Life_Balance_Avg'] = cluster_data['work_life_balance'].mean()
    profile['Screen_Time_Avg'] = cluster_data['daily_screen_time_hours'].mean()
    profile['Age_Avg'] = cluster_data['age'].mean()
    
    cluster_profiles.append(profile)
    
    # =====================================================================
    # ĐẶT TÊN CỤM THÔNG MINH DỰA TRÊN ĐẶC ĐIỂM
    # =====================================================================
    
    wellness = profile['Overall_Wellness_Avg']
    health = profile['Health_Score_Avg']
    digital_stress = profile['Digital_Stress_Avg']
    screen_time = profile['Screen_Time_Avg']
    work_balance = profile['Work_Life_Balance_Avg']
    sleep_health = profile['Sleep_Health_Avg']
    age = profile['Age_Avg']
    stress_level = target_names[dominant_stress]
    
    # Phân loại theo wellness level
    if wellness >= 75:
        wellness_category = "Excellent"
        wellness_emoji = "🌟"
    elif wellness >= 65:
        wellness_category = "Good"
        wellness_emoji = "✅"
    elif wellness >= 55:
        wellness_category = "Moderate"
        wellness_emoji = "⚠️"
    elif wellness >= 45:
        wellness_category = "Fair"
        wellness_emoji = "🟡"
    else:
        wellness_category = "Poor"
        wellness_emoji = "🔴"
    
    # Phân loại theo digital stress
    if digital_stress >= 50:
        digital_category = "High Digital Stress"
    elif digital_stress >= 35:
        digital_category = "Moderate Digital Use"
    else:
        digital_category = "Low Digital Stress"
    
    # Phân loại theo work-life balance
    if work_balance >= 70:
        balance_category = "Great Balance"
    elif work_balance >= 50:
        balance_category = "Fair Balance"
    else:
        balance_category = "Poor Balance"
    
    # Phân loại theo tuổi
    if age >= 60:
        age_category = "Seniors"
    elif age >= 40:
        age_category = "Middle-aged"
    elif age >= 25:
        age_category = "Young Adults"
    else:
        age_category = "Youth"
    
    # TẠO TÊN CỤM DỰA TRÊN ĐẶC ĐIỂM NỔI BẬT
    name_parts = []
    
    # Thêm wellness status
    name_parts.append(f"{wellness_emoji} {wellness_category}")
    
    # Thêm đặc điểm nổi bật
    if digital_stress >= 50 and screen_time >= 7:
        name_parts.append("- Heavy Tech Users")
    elif digital_stress <= 30 and screen_time <= 4:
        name_parts.append("- Minimal Tech Users")
    elif screen_time >= 6:
        name_parts.append("- High Screen Time")
    
    if work_balance <= 40:
        name_parts.append("- Overworked")
    elif work_balance >= 75:
        name_parts.append("- Well-balanced")
    
    if health >= 75:
        name_parts.append("- Healthy")
    elif health <= 50:
        name_parts.append("- Health Concerns")
    
    if sleep_health >= 75:
        name_parts.append("- Good Sleep")
    elif sleep_health <= 55:
        name_parts.append("- Sleep Issues")
    
    # Ghép tên
    cluster_name = " ".join(name_parts)
    
    # Thêm age category và stress level
    full_name = f"Cluster {i+1}: {cluster_name} ({age_category}, {stress_level} Stress)"
    
    cluster_names.append(full_name)
    
    # In ra thông tin chi tiết
    print(f"\n🔹 {full_name}")
    print(f"   Size: {profile['Size']} people ({profile['Percentage']})")
    print(f"   Overall Wellness: {wellness:.1f}/100")
    print(f"   Health Score: {health:.1f}/100")
    print(f"   Digital Stress: {digital_stress:.1f}/100")
    print(f"   Work-Life Balance: {work_balance:.1f}/100")
    print(f"   Sleep Health: {sleep_health:.1f}/100")
    print(f"   Avg Screen Time: {screen_time:.1f} hrs/day")
    print(f"   Avg Age: {age:.1f} years")
    print(f"   Dominant Stress: {stress_level}")

print("\n" + "="*70)

cluster_profiles = pd.DataFrame(cluster_profiles)

print("\n📊 Summary Table:")
summary_df = pd.DataFrame([
    {
        'Cluster': i,
        'Name': cluster_names[i].split(': ')[1].split(' (')[0][:40] + '...' if len(cluster_names[i]) > 50 else cluster_names[i].split(': ')[1],
        'Size': cluster_profiles.iloc[i]['Size'],
        'Wellness': f"{cluster_profiles.iloc[i]['Overall_Wellness_Avg']:.1f}",
        'Health': f"{cluster_profiles.iloc[i]['Health_Score_Avg']:.1f}",
        'Digital_Stress': f"{cluster_profiles.iloc[i]['Digital_Stress_Avg']:.1f}",
        'Stress_Level': target_names[cluster_to_stress[i]]
    }
    for i in range(optimal_k)
])
print(summary_df.to_string(index=False))

# =============================================================================
# BƯỚC 8: CLUSTER CHARACTERISTICS VISUALIZATION
# =============================================================================
print("\n" + "="*80)
print("📊 BƯỚC 8: CLUSTER CHARACTERISTICS VISUALIZATION")
print("="*80)

# Tạo comprehensive cluster profile visualization
fig, axes = plt.subplots(3, 3, figsize=(22, 18))
axes = axes.flatten()

# 1. Overall Wellness by Cluster
ax = axes[0]
wellness_values = [cluster_profiles.iloc[i]['Overall_Wellness_Avg'] for i in range(optimal_k)]
colors_wellness = ['#27ae60' if w >= 70 else '#f39c12' if w >= 55 else '#e74c3c' for w in wellness_values]
bars = ax.bar(range(optimal_k), wellness_values, color=colors_wellness, edgecolor='black', linewidth=2)
ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax.set_ylabel('Overall Wellness', fontsize=11, fontweight='bold')
ax.set_title('🌟 Overall Wellness Score by Cluster', fontsize=12, fontweight='bold')
ax.set_xticks(range(optimal_k))
ax.set_xticklabels([f'C{i+1}' for i in range(optimal_k)])
ax.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Good (>70)')
ax.axhline(y=55, color='orange', linestyle='--', alpha=0.5, label='Moderate (>55)')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(wellness_values):
    ax.text(i, v + 1.5, f'{v:.1f}', ha='center', fontweight='bold', fontsize=10)

# 2. Health Score by Cluster
ax = axes[1]
health_values = [cluster_profiles.iloc[i]['Health_Score_Avg'] for i in range(optimal_k)]
colors_health = ['#2ecc71' if h >= 70 else '#f39c12' if h >= 55 else '#e74c3c' for h in health_values]
bars = ax.bar(range(optimal_k), health_values, color=colors_health, edgecolor='black', linewidth=2)
ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax.set_ylabel('Health Score', fontsize=11, fontweight='bold')
ax.set_title('💚 Mental Health Score by Cluster', fontsize=12, fontweight='bold')
ax.set_xticks(range(optimal_k))
ax.set_xticklabels([f'C{i+1}' for i in range(optimal_k)])
ax.axhline(y=70, color='green', linestyle='--', alpha=0.5)
ax.axhline(y=55, color='orange', linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(health_values):
    ax.text(i, v + 1.5, f'{v:.1f}', ha='center', fontweight='bold', fontsize=10)

# 3. Digital Stress by Cluster
ax = axes[2]
digital_values = [cluster_profiles.iloc[i]['Digital_Stress_Avg'] for i in range(optimal_k)]
colors_digital = ['#e74c3c' if d >= 50 else '#f39c12' if d >= 35 else '#2ecc71' for d in digital_values]
bars = ax.bar(range(optimal_k), digital_values, color=colors_digital, edgecolor='black', linewidth=2)
ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax.set_ylabel('Digital Stress Score', fontsize=11, fontweight='bold')
ax.set_title('📱 Digital Stress Level by Cluster', fontsize=12, fontweight='bold')
ax.set_xticks(range(optimal_k))
ax.set_xticklabels([f'C{i+1}' for i in range(optimal_k)])
ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='High (>50)')
ax.axhline(y=35, color='orange', linestyle='--', alpha=0.5, label='Moderate (>35)')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(digital_values):
    ax.text(i, v + 1.5, f'{v:.1f}', ha='center', fontweight='bold', fontsize=10)

# 4. Work-Life Balance by Cluster
ax = axes[3]
balance_values = [cluster_profiles.iloc[i]['Work_Life_Balance_Avg'] for i in range(optimal_k)]
colors_balance = ['#2ecc71' if b >= 70 else '#f39c12' if b >= 50 else '#e74c3c' for b in balance_values]
bars = ax.bar(range(optimal_k), balance_values, color=colors_balance, edgecolor='black', linewidth=2)
ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax.set_ylabel('Work-Life Balance', fontsize=11, fontweight='bold')
ax.set_title('⚖️ Work-Life Balance by Cluster', fontsize=12, fontweight='bold')
ax.set_xticks(range(optimal_k))
ax.set_xticklabels([f'C{i+1}' for i in range(optimal_k)])
ax.axhline(y=70, color='green', linestyle='--', alpha=0.5)
ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(balance_values):
    ax.text(i, v + 1.5, f'{v:.1f}', ha='center', fontweight='bold', fontsize=10)

# 5. Sleep Health by Cluster
ax = axes[4]
sleep_values = [cluster_profiles.iloc[i]['Sleep_Health_Avg'] for i in range(optimal_k)]
colors_sleep = ['#9b59b6' if s >= 70 else '#3498db' if s >= 55 else '#95a5a6' for s in sleep_values]
bars = ax.bar(range(optimal_k), sleep_values, color=colors_sleep, edgecolor='black', linewidth=2)
ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax.set_ylabel('Sleep Health Index', fontsize=11, fontweight='bold')
ax.set_title('😴 Sleep Health by Cluster', fontsize=12, fontweight='bold')
ax.set_xticks(range(optimal_k))
ax.set_xticklabels([f'C{i+1}' for i in range(optimal_k)])
ax.axhline(y=70, color='green', linestyle='--', alpha=0.5)
ax.axhline(y=55, color='orange', linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(sleep_values):
    ax.text(i, v + 1.5, f'{v:.1f}', ha='center', fontweight='bold', fontsize=10)

# 6. Screen Time by Cluster
ax = axes[5]
screen_values = [cluster_profiles.iloc[i]['Screen_Time_Avg'] for i in range(optimal_k)]
colors_screen = ['#e74c3c' if s >= 7 else '#f39c12' if s >= 5 else '#2ecc71' for s in screen_values]
bars = ax.bar(range(optimal_k), screen_values, color=colors_screen, edgecolor='black', linewidth=2)
ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax.set_ylabel('Screen Time (hours/day)', fontsize=11, fontweight='bold')
ax.set_title('📺 Daily Screen Time by Cluster', fontsize=12, fontweight='bold')
ax.set_xticks(range(optimal_k))
ax.set_xticklabels([f'C{i+1}' for i in range(optimal_k)])
ax.axhline(y=7, color='red', linestyle='--', alpha=0.5, label='High (>7h)')
ax.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Moderate (>5h)')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(screen_values):
    ax.text(i, v + 0.2, f'{v:.1f}h', ha='center', fontweight='bold', fontsize=10)

# 7. Age Distribution by Cluster
ax = axes[6]
age_values = [cluster_profiles.iloc[i]['Age_Avg'] for i in range(optimal_k)]
colors_age = ['#8e44ad' if a >= 60 else '#3498db' if a >= 40 else '#1abc9c' if a >= 25 else '#f1c40f' for a in age_values]
bars = ax.bar(range(optimal_k), age_values, color=colors_age, edgecolor='black', linewidth=2)
ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Age', fontsize=11, fontweight='bold')
ax.set_title('👥 Average Age by Cluster', fontsize=12, fontweight='bold')
ax.set_xticks(range(optimal_k))
ax.set_xticklabels([f'C{i+1}' for i in range(optimal_k)])
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(age_values):
    ax.text(i, v + 1, f'{v:.0f}', ha='center', fontweight='bold', fontsize=10)

# 8. Cluster Size Distribution
ax = axes[7]
size_values = [cluster_profiles.iloc[i]['Size'] for i in range(optimal_k)]
colors_size = plt.cm.Set3(np.linspace(0, 1, optimal_k))
bars = ax.bar(range(optimal_k), size_values, color=colors_size, edgecolor='black', linewidth=2)
ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of People', fontsize=11, fontweight='bold')
ax.set_title('👫 Cluster Size Distribution', fontsize=12, fontweight='bold')
ax.set_xticks(range(optimal_k))
ax.set_xticklabels([f'C{i+1}' for i in range(optimal_k)])
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(size_values):
    pct = v / sum(size_values) * 100
    ax.text(i, v + max(size_values)*0.02, f'{v}\n({pct:.1f}%)', 
            ha='center', fontweight='bold', fontsize=9)

# 9. Stress Level Distribution by Cluster
ax = axes[8]
stress_data = []
for i in range(optimal_k):
    cluster_stress = X_train_df[X_train_df['Cluster']==i]['Stress_Level'].value_counts().sort_index()
    stress_data.append([cluster_stress.get(j, 0) for j in range(3)])

stress_data = np.array(stress_data).T
x = np.arange(optimal_k)
width = 0.25
colors_stress = ['#2ecc71', '#f39c12', '#e74c3c']

for i, (stress_level, color) in enumerate(zip(['Low', 'Medium', 'High'], colors_stress)):
    ax.bar(x + i*width - width, stress_data[i], width, label=stress_level, 
           color=color, edgecolor='black', linewidth=1)

ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of People', fontsize=11, fontweight='bold')
ax.set_title('📊 Stress Level Distribution by Cluster', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'C{i+1}' for i in range(optimal_k)])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Cluster characteristics visualization completed!")

# =============================================================================
# BƯỚC 9: INTERACTIVE 3D VISUALIZATION
# =============================================================================
print("\n" + "="*80)
print("🎨 BƯỚC 9: INTERACTIVE 3D VISUALIZATION")
print("="*80)

# PCA để giảm về 3D
pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X_train_scaled)

print(f"✅ PCA explained variance: {pca_3d.explained_variance_ratio_.sum()*100:.2f}%")

# Tạo DataFrame cho visualization
viz_df = pd.DataFrame({
    'PC1': X_pca_3d[:, 0],
    'PC2': X_pca_3d[:, 1],
    'PC3': X_pca_3d[:, 2],
    'Cluster': clusters_train,
    'Stress_Level': [target_names[i] for i in y_train.values],
    'Health_Score': X_train_df['health_score'].values,
    'Overall_Wellness': X_train_df['overall_wellness'].values,
    'Digital_Stress': X_train_df['digital_stress_score'].values,
    'Cluster_Name': [cluster_names[c] for c in clusters_train]
})

# 3D Scatter Plot - Colored by Cluster
fig_3d_cluster = px.scatter_3d(
    viz_df,
    x='PC1', y='PC2', z='PC3',
    color='Cluster_Name',
    size='Overall_Wellness',
    hover_data=['Stress_Level', 'Health_Score', 'Digital_Stress'],
    title='🎨 3D Cluster Visualization (Sized by Overall Wellness)',
    labels={'PC1': 'PC1', 'PC2': 'PC2', 'PC3': 'PC3'},
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig_3d_cluster.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey')))
fig_3d_cluster.update_layout(
    width=1200, height=800,
    scene=dict(
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        zaxis_title='Principal Component 3',
        bgcolor='#f8f9fa'
    ),
    font=dict(family='Arial', size=12)
)
fig_3d_cluster.show()

# 3D Scatter Plot - Colored by Stress Level
fig_3d_stress = px.scatter_3d(
    viz_df,
    x='PC1', y='PC2', z='PC3',
    color='Stress_Level',
    size='Health_Score',
    hover_data=['Cluster_Name', 'Overall_Wellness', 'Digital_Stress'],
    title='🔥 3D Stress Level Visualization (Sized by Health Score)',
    labels={'PC1': 'PC1', 'PC2': 'PC2', 'PC3': 'PC3'},
    color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
)

fig_3d_stress.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey')))
fig_3d_stress.update_layout(
    width=1200, height=800,
    scene=dict(
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        zaxis_title='Principal Component 3',
        bgcolor='#f8f9fa'
    ),
    font=dict(family='Arial', size=12)
)
fig_3d_stress.show()

# =============================================================================
# BƯỚC 10: TRAIN MODELS
# =============================================================================
print("\n" + "="*80)
print("🤖 BƯỚC 10: TRAIN PREDICTION MODELS")
print("="*80)

# Decision Tree
print("\n🌲 Training Decision Tree...")
dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42)
dt_model.fit(X_train_scaled, y_train)
y_pred_dt = dt_model.predict(X_test_scaled)
dt_acc = accuracy_score(y_test, y_pred_dt)
print(f"   Decision Tree Accuracy: {dt_acc*100:.2f}%")

# Random Forest
print("\n🌳 Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, y_pred_rf)
print(f"   Random Forest Accuracy: {rf_acc*100:.2f}%")

# K-Means (dùng cluster_to_stress để predict)
print("\n🎯 Evaluating K-Means...")
y_pred_kmeans = [cluster_to_stress[c] for c in clusters_test]
kmeans_acc = accuracy_score(y_test, y_pred_kmeans)
print(f"   K-Means Accuracy: {kmeans_acc*100:.2f}%")

# So sánh models
print("\n" + "="*80)
print("🏆 MODEL COMPARISON")
print("="*80)
print(f"\n   K-Means:        {kmeans_acc*100:.2f}%")
print(f"   Decision Tree:  {dt_acc*100:.2f}%")
print(f"   Random Forest:  {rf_acc*100:.2f}% ⭐ BEST")

# =============================================================================
# BƯỚC 10A: CLASSIFICATION REPORTS
# =============================================================================
print("\n" + "="*80)
print("📊 BƯỚC 10A: CLASSIFICATION REPORTS")
print("="*80)

print("\n" + "="*70)
print("1️⃣ K-MEANS CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_test, y_pred_kmeans, target_names=target_names, digits=3))

print("\n" + "="*70)
print("2️⃣ DECISION TREE CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_test, y_pred_dt, target_names=target_names, digits=3))

print("\n" + "="*70)
print("3️⃣ RANDOM FOREST CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_test, y_pred_rf, target_names=target_names, digits=3))

# =============================================================================
# BƯỚC 10B: CONFUSION MATRICES & MODEL COMPARISON
# =============================================================================
print("\n" + "="*80)
print("📈 BƯỚC 10B: CONFUSION MATRICES & MODEL COMPARISON")
print("="*80)

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Row 1: Confusion Matrices
ax1 = fig.add_subplot(gs[0, 0])
cm_kmeans = confusion_matrix(y_test, y_pred_kmeans)
sns.heatmap(cm_kmeans, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'Count'}, ax=ax1)
ax1.set_title('🎯 K-Means Confusion Matrix', fontsize=13, fontweight='bold', pad=10)
ax1.set_ylabel('True Label', fontsize=11, fontweight='bold')
ax1.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')

ax2 = fig.add_subplot(gs[0, 1])
cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens',
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'Count'}, ax=ax2)
ax2.set_title('🌲 Decision Tree Confusion Matrix', fontsize=13, fontweight='bold', pad=10)
ax2.set_ylabel('True Label', fontsize=11, fontweight='bold')
ax2.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')

ax3 = fig.add_subplot(gs[0, 2])
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Oranges',
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'Count'}, ax=ax3)
ax3.set_title('🌳 Random Forest Confusion Matrix', fontsize=13, fontweight='bold', pad=10)
ax3.set_ylabel('True Label', fontsize=11, fontweight='bold')
ax3.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')

# Row 2: Accuracy Comparison
ax4 = fig.add_subplot(gs[1, :])
models = ['K-Means', 'Decision Tree', 'Random Forest']
accuracies = [kmeans_acc * 100, dt_acc * 100, rf_acc * 100]
colors = ['#3498db', '#2ecc71', '#e74c3c']

bars = ax4.bar(models, accuracies, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
ax4.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax4.set_title('🏆 Model Accuracy Comparison', fontsize=15, fontweight='bold', pad=15)
ax4.set_ylim([0, 100])
ax4.axhline(y=80, color='green', linestyle='--', linewidth=2, alpha=0.5, label='80% threshold')
ax4.axhline(y=60, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='60% threshold')
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{acc:.2f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add rank
    rank = ['🥉', '🥈', '🥇'][np.argsort(accuracies)[i]]
    ax4.text(bar.get_x() + bar.get_width()/2., height - 5,
            rank, ha='center', va='top', fontsize=20)

# Row 3: Per-Class Metrics Comparison
metrics_data = []
for model_name, y_pred in [('K-Means', y_pred_kmeans), 
                           ('Decision Tree', y_pred_dt), 
                           ('Random Forest', y_pred_rf)]:
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    for class_name in target_names:
        metrics_data.append({
            'Model': model_name,
            'Class': class_name,
            'Precision': report[class_name]['precision'],
            'Recall': report[class_name]['recall'],
            'F1-Score': report[class_name]['f1-score']
        })

metrics_df = pd.DataFrame(metrics_data)

# Precision comparison
ax5 = fig.add_subplot(gs[2, 0])
precision_pivot = metrics_df.pivot(index='Class', columns='Model', values='Precision')
precision_pivot.plot(kind='bar', ax=ax5, color=['#3498db', '#2ecc71', '#e74c3c'], 
                     edgecolor='black', linewidth=1.5, alpha=0.8)
ax5.set_title('📊 Precision by Class', fontsize=12, fontweight='bold')
ax5.set_ylabel('Precision', fontsize=11, fontweight='bold')
ax5.set_xlabel('Stress Level', fontsize=11, fontweight='bold')
ax5.legend(title='Model', fontsize=9)
ax5.grid(axis='y', alpha=0.3)
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=0)
ax5.set_ylim([0, 1.1])

# Recall comparison
ax6 = fig.add_subplot(gs[2, 1])
recall_pivot = metrics_df.pivot(index='Class', columns='Model', values='Recall')
recall_pivot.plot(kind='bar', ax=ax6, color=['#3498db', '#2ecc71', '#e74c3c'],
                  edgecolor='black', linewidth=1.5, alpha=0.8)
ax6.set_title('📊 Recall by Class', fontsize=12, fontweight='bold')
ax6.set_ylabel('Recall', fontsize=11, fontweight='bold')
ax6.set_xlabel('Stress Level', fontsize=11, fontweight='bold')
ax6.legend(title='Model', fontsize=9)
ax6.grid(axis='y', alpha=0.3)
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=0)
ax6.set_ylim([0, 1.1])

# F1-Score comparison
ax7 = fig.add_subplot(gs[2, 2])
f1_pivot = metrics_df.pivot(index='Class', columns='Model', values='F1-Score')
f1_pivot.plot(kind='bar', ax=ax7, color=['#3498db', '#2ecc71', '#e74c3c'],
              edgecolor='black', linewidth=1.5, alpha=0.8)
ax7.set_title('📊 F1-Score by Class', fontsize=12, fontweight='bold')
ax7.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
ax7.set_xlabel('Stress Level', fontsize=11, fontweight='bold')
ax7.legend(title='Model', fontsize=9)
ax7.grid(axis='y', alpha=0.3)
ax7.set_xticklabels(ax7.get_xticklabels(), rotation=0)
ax7.set_ylim([0, 1.1])

plt.suptitle('📈 COMPREHENSIVE MODEL EVALUATION DASHBOARD', 
             fontsize=18, fontweight='bold', y=0.995)
plt.show()

print("✅ Model comparison and evaluation completed!")

# =============================================================================
# BƯỚC 11: FEATURE IMPORTANCE - Mental Health Focus
# =============================================================================
print("\n" + "="*80)
print("📊 BƯỚC 11: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n🏆 Top 10 Most Important Features:")
for i in range(min(10, len(selected_features))):
    idx = indices[i]
    marker = '🆕' if selected_features[idx] in selected_features[11:] else '  '
    print(f"   {i+1:2d}. {marker} {selected_features[idx]:<30} {importances[idx]*100:.2f}%")

# =============================================================================
# BƯỚC 11A: STRESS FACTORS ANALYSIS (YẾU TỐ ẢNH HƯỞNG ĐẾN STRESS)
# =============================================================================
print("\n" + "="*80)
print("🔍 BƯỚC 11A: STRESS FACTORS ANALYSIS")
print("="*80)

# Tạo comprehensive stress factors visualization
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# 1. Feature Importance (Top Factors)
ax1 = fig.add_subplot(gs[0, :2])
colors_imp = ['#e74c3c' if f in selected_features[11:] else '#3498db' 
              for f in [selected_features[i] for i in indices[:12]]]
bars = ax1.barh(range(12), importances[indices[:12]], color=colors_imp, 
                edgecolor='black', linewidth=1.5)
ax1.set_yticks(range(12))
ax1.set_yticklabels([selected_features[i] for i in indices[:12]], fontsize=10)
ax1.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax1.set_title('🏆 Top 12 Factors Affecting Stress (🔴 = Mental Health Features)', 
              fontsize=13, fontweight='bold', pad=15)
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, imp) in enumerate(zip(bars, importances[indices[:12]])):
    width = bar.get_width()
    ax1.text(width + 0.005, bar.get_y() + bar.get_height()/2,
            f'{imp*100:.1f}%', va='center', fontweight='bold', fontsize=9)

# 2. Correlation with Stress (Positive vs Negative)
ax2 = fig.add_subplot(gs[0, 2])
correlations_full = X_train_df.copy()
correlations_full['Stress_Level'] = y_train.values
corr_with_stress_full = correlations_full.corr()['Stress_Level'].drop('Stress_Level')
corr_with_stress_full = corr_with_stress_full.dropna().sort_values()

# Separate positive and negative correlations
pos_corr = corr_with_stress_full[corr_with_stress_full > 0].sort_values(ascending=False)[:8]
neg_corr = corr_with_stress_full[corr_with_stress_full < 0].sort_values()[:8]

y_pos_pos = np.arange(len(pos_corr))
y_pos_neg = np.arange(len(neg_corr))

ax2.barh(y_pos_pos, pos_corr.values, color='#e74c3c', alpha=0.7, 
         edgecolor='black', linewidth=1, label='Increase Stress')
ax2.barh(y_pos_neg + len(pos_corr) + 0.5, neg_corr.values, color='#2ecc71', alpha=0.7,
         edgecolor='black', linewidth=1, label='Reduce Stress')

all_labels = list(pos_corr.index) + list(neg_corr.index)
y_positions = list(y_pos_pos) + list(y_pos_neg + len(pos_corr) + 0.5)
ax2.set_yticks(y_positions)
ax2.set_yticklabels(all_labels, fontsize=8)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
ax2.set_xlabel('Correlation', fontsize=10, fontweight='bold')
ax2.set_title('📊 Factors Correlation\n(Positive vs Negative)', 
              fontsize=11, fontweight='bold', pad=10)
ax2.legend(fontsize=8)
ax2.grid(axis='x', alpha=0.3)

# 3. Mental Health Impact on Stress
ax3 = fig.add_subplot(gs[1, 0])
mental_health_features = [f for f in selected_features[11:] if f in X_train_df.columns]
mental_impact = []
for feat in mental_health_features:
    corr = correlations_full[[feat, 'Stress_Level']].corr().iloc[0, 1]
    if not np.isnan(corr):
        mental_impact.append((feat, abs(corr)))

mental_impact.sort(key=lambda x: x[1], reverse=True)
features_mh = [x[0] for x in mental_impact]
impacts_mh = [x[1] for x in mental_impact]

colors_mh = ['#e74c3c' if correlations_full[[f, 'Stress_Level']].corr().iloc[0, 1] > 0 
             else '#2ecc71' for f in features_mh]
ax3.barh(range(len(features_mh)), impacts_mh, color=colors_mh, 
         edgecolor='black', linewidth=1.5, alpha=0.8)
ax3.set_yticks(range(len(features_mh)))
ax3.set_yticklabels(features_mh, fontsize=9)
ax3.set_xlabel('Absolute Correlation', fontsize=10, fontweight='bold')
ax3.set_title('💚 Mental Health Factors\nImpact on Stress', 
              fontsize=11, fontweight='bold', pad=10)
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3)

# 4. Screen Time Impact
ax4 = fig.add_subplot(gs[1, 1])
screen_features = ['daily_screen_time_hours', 'phone_usage_hours', 
                   'social_media_hours', 'gaming_hours']
screen_impacts = []
for feat in screen_features:
    if feat in X_train_df.columns:
        corr = correlations_full[[feat, 'Stress_Level']].corr().iloc[0, 1]
        if not np.isnan(corr):
            screen_impacts.append(corr)
        else:
            screen_impacts.append(0)

colors_screen = ['#e74c3c' if x > 0 else '#2ecc71' for x in screen_impacts]
bars_screen = ax4.bar(range(len(screen_features)), screen_impacts, 
                      color=colors_screen, edgecolor='black', linewidth=1.5, alpha=0.8)
ax4.set_xticks(range(len(screen_features)))
ax4.set_xticklabels(['Screen Time', 'Phone', 'Social Media', 'Gaming'], 
                    rotation=45, ha='right', fontsize=9)
ax4.set_ylabel('Correlation with Stress', fontsize=10, fontweight='bold')
ax4.set_title('📱 Digital Usage Impact\non Stress', fontsize=11, fontweight='bold', pad=10)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax4.grid(axis='y', alpha=0.3)

for bar, val in zip(bars_screen, screen_impacts):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.01),
            f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top',
            fontweight='bold', fontsize=9)

# 5. Work-Life Balance Impact
ax5 = fig.add_subplot(gs[1, 2])
work_features = ['work_related_hours', 'work_life_balance']
work_impacts = []
for feat in work_features:
    if feat in X_train_df.columns:
        corr = correlations_full[[feat, 'Stress_Level']].corr().iloc[0, 1]
        if not np.isnan(corr):
            work_impacts.append(corr)
        else:
            work_impacts.append(0)

colors_work = ['#e74c3c' if x > 0 else '#2ecc71' for x in work_impacts]
bars_work = ax5.bar(range(len(work_features)), work_impacts,
                    color=colors_work, edgecolor='black', linewidth=1.5, alpha=0.8)
ax5.set_xticks(range(len(work_features)))
ax5.set_xticklabels(['Work Hours', 'Work-Life\nBalance'], fontsize=9)
ax5.set_ylabel('Correlation with Stress', fontsize=10, fontweight='bold')
ax5.set_title('⚖️ Work Factors Impact\non Stress', fontsize=11, fontweight='bold', pad=10)
ax5.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax5.grid(axis='y', alpha=0.3)

for bar, val in zip(bars_work, work_impacts):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.01),
            f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top',
            fontweight='bold', fontsize=9)

# 6. Age Impact on Stress
ax6 = fig.add_subplot(gs[2, 0])
age_groups = pd.cut(X_train_df['age'], bins=[0, 25, 40, 60, 100], 
                    labels=['<25', '25-40', '40-60', '60+'])
stress_by_age = X_train_df.groupby(age_groups)['Stress_Level'].mean()

colors_age = ['#f1c40f', '#1abc9c', '#3498db', '#8e44ad']
bars_age = ax6.bar(range(len(stress_by_age)), stress_by_age.values,
                   color=colors_age, edgecolor='black', linewidth=1.5, alpha=0.8)
ax6.set_xticks(range(len(stress_by_age)))
ax6.set_xticklabels(stress_by_age.index, fontsize=10)
ax6.set_xlabel('Age Group', fontsize=10, fontweight='bold')
ax6.set_ylabel('Average Stress Level', fontsize=10, fontweight='bold')
ax6.set_title('👥 Average Stress by Age Group', fontsize=11, fontweight='bold', pad=10)
ax6.grid(axis='y', alpha=0.3)

for bar, val in zip(bars_age, stress_by_age.values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 7. Sleep Impact on Stress
ax7 = fig.add_subplot(gs[2, 1])
sleep_bins = pd.cut(X_train_df['sleep_health_index'], bins=[0, 50, 65, 80, 100],
                    labels=['Poor\n(<50)', 'Fair\n(50-65)', 'Good\n(65-80)', 'Excellent\n(80+)'])
stress_by_sleep = X_train_df.groupby(sleep_bins)['Stress_Level'].mean()

colors_sleep = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71']
bars_sleep = ax7.bar(range(len(stress_by_sleep)), stress_by_sleep.values,
                     color=colors_sleep, edgecolor='black', linewidth=1.5, alpha=0.8)
ax7.set_xticks(range(len(stress_by_sleep)))
ax7.set_xticklabels(stress_by_sleep.index, fontsize=9)
ax7.set_xlabel('Sleep Health Level', fontsize=10, fontweight='bold')
ax7.set_ylabel('Average Stress Level', fontsize=10, fontweight='bold')
ax7.set_title('😴 Average Stress by Sleep Quality', fontsize=11, fontweight='bold', pad=10)
ax7.grid(axis='y', alpha=0.3)

for bar, val in zip(bars_sleep, stress_by_sleep.values):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 8. Overall Wellness Impact on Stress
ax8 = fig.add_subplot(gs[2, 2])
wellness_bins = pd.cut(X_train_df['overall_wellness'], bins=[0, 50, 65, 80, 100],
                       labels=['Poor\n(<50)', 'Fair\n(50-65)', 'Good\n(65-80)', 'Excellent\n(80+)'])
stress_by_wellness = X_train_df.groupby(wellness_bins)['Stress_Level'].mean()

colors_wellness = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71']
bars_wellness = ax8.bar(range(len(stress_by_wellness)), stress_by_wellness.values,
                        color=colors_wellness, edgecolor='black', linewidth=1.5, alpha=0.8)
ax8.set_xticks(range(len(stress_by_wellness)))
ax8.set_xticklabels(stress_by_wellness.index, fontsize=9)
ax8.set_xlabel('Overall Wellness Level', fontsize=10, fontweight='bold')
ax8.set_ylabel('Average Stress Level', fontsize=10, fontweight='bold')
ax8.set_title('🌟 Average Stress by Wellness', fontsize=11, fontweight='bold', pad=10)
ax8.grid(axis='y', alpha=0.3)

for bar, val in zip(bars_wellness, stress_by_wellness.values):
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.suptitle('🔍 COMPREHENSIVE STRESS FACTORS ANALYSIS', 
             fontsize=18, fontweight='bold', y=0.995)
plt.show()

print("✅ Stress factors analysis completed!")

# Print key insights
print("\n" + "="*80)
print("💡 KEY INSIGHTS - FACTORS AFFECTING STRESS")
print("="*80)

# Top positive correlations (increase stress)
print("\n🔺 TOP FACTORS THAT INCREASE STRESS:")
top_positive = corr_with_stress_full[corr_with_stress_full > 0].sort_values(ascending=False)[:5]
for i, (feat, corr) in enumerate(top_positive.items(), 1):
    print(f"   {i}. {feat}: +{corr:.3f} correlation")

# Top negative correlations (reduce stress)  
print("\n🔻 TOP FACTORS THAT REDUCE STRESS:")
top_negative = corr_with_stress_full[corr_with_stress_full < 0].sort_values()[:5]
for i, (feat, corr) in enumerate(top_negative.items(), 1):
    print(f"   {i}. {feat}: {corr:.3f} correlation")

print("\n" + "="*80)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Feature Importance Bar Chart
colors = ['#e74c3c' if f in selected_features[11:] else '#3498db' 
          for f in [selected_features[i] for i in indices]]
axes[0].barh(range(len(selected_features)), importances[indices], color=colors, 
             edgecolor='black', linewidth=1.5)
axes[0].set_yticks(range(len(selected_features)))
axes[0].set_yticklabels([selected_features[i] for i in indices], fontsize=9)
axes[0].set_xlabel('Importance Score', fontsize=12, fontweight='bold')
axes[0].set_title('📊 Feature Importance (🔴 = Mental Health Features)', 
                  fontsize=13, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# Mental Health Features Focus
mental_indices = [i for i, f in enumerate(selected_features) if f in selected_features[11:]]
mental_importances = importances[mental_indices]
mental_names = [selected_features[i] for i in mental_indices]

axes[1].bar(range(len(mental_names)), mental_importances, 
            color='#e74c3c', edgecolor='black', linewidth=1.5)
axes[1].set_xticks(range(len(mental_names)))
axes[1].set_xticklabels(mental_names, rotation=45, ha='right', fontsize=10)
axes[1].set_ylabel('Importance Score', fontsize=12, fontweight='bold')
axes[1].set_title('💚 Mental Health Features Importance', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

for i, v in enumerate(mental_importances):
    axes[1].text(i, v + 0.005, f'{v*100:.1f}%', ha='center', 
                fontweight='bold', fontsize=10)

plt.tight_layout()
plt.show()

# =============================================================================
# BƯỚC 12: MENTAL HEALTH ANALYSIS BY CLUSTER
# =============================================================================
print("\n" + "="*80)
print("💚 BƯỚC 12: MENTAL HEALTH ANALYSIS BY CLUSTER")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

mental_metrics = [
    ('health_score', 'Health Score (Mental Health)', '#e74c3c'),
    ('overall_wellness', 'Overall Wellness Score', '#2ecc71'),
    ('sleep_health_index', 'Sleep Health Index', '#9b59b6'),
    ('emotional_balance', 'Emotional Balance', '#f39c12'),
    ('digital_stress_score', 'Digital Stress Score', '#e67e22'),
    ('work_life_balance', 'Work-Life Balance', '#1abc9c')
]

for idx, (metric, title, color) in enumerate(mental_metrics):
    ax = axes[idx]
    
    cluster_values = [X_train_df[X_train_df['Cluster']==i][metric].mean() 
                     for i in range(optimal_k)]
    
    bars = ax.bar(range(optimal_k), cluster_values, color=color,
                  edgecolor='black', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Cluster ID', fontsize=11, fontweight='bold')
    ax.set_ylabel(title, fontsize=11, fontweight='bold')
    ax.set_title(f'📊 {title} by Cluster', fontsize=12, fontweight='bold')
    ax.set_xticks(range(optimal_k))
    ax.set_xticklabels([f'C{i}' for i in range(optimal_k)])
    ax.grid(axis='y', alpha=0.3)
    
    # Thêm giá trị lên bar
    for i, v in enumerate(cluster_values):
        ax.text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold', fontsize=10)
    
    # Thêm threshold line
    if metric != 'digital_stress_score':  # Cao là tốt
        ax.axhline(y=70, color='green', linestyle='--', linewidth=1.5, 
                  alpha=0.5, label='Good (>70)')
        ax.axhline(y=50, color='orange', linestyle='--', linewidth=1.5, 
                  alpha=0.5, label='Moderate (50-70)')
    else:  # Thấp là tốt
        ax.axhline(y=30, color='green', linestyle='--', linewidth=1.5, 
                  alpha=0.5, label='Good (<30)')
        ax.axhline(y=50, color='orange', linestyle='--', linewidth=1.5, 
                  alpha=0.5, label='Moderate (30-50)')
    
    ax.legend(fontsize=9)

plt.tight_layout()
plt.show()

# Radar Chart cho Mental Health Metrics
print("\n📊 Tạo Radar Chart cho Mental Health...")

fig = go.Figure()

for i in range(optimal_k):
    cluster_data = X_train_df[X_train_df['Cluster'] == i]
    
    values = [
        cluster_data['health_score'].mean(),
        cluster_data['overall_wellness'].mean(),
        cluster_data['sleep_health_index'].mean(),
        cluster_data['emotional_balance'].mean(),
        100 - cluster_data['digital_stress_score'].mean(),  # Invert digital stress
        cluster_data['work_life_balance'].mean()
    ]
    values.append(values[0])  # Close the polygon
    
    categories = ['Health<br>Score', 'Overall<br>Wellness', 'Sleep<br>Health', 
                 'Emotional<br>Balance', 'Digital<br>Wellness', 'Work-Life<br>Balance']
    categories.append(categories[0])
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        name=cluster_names[i],
        fill='toself',
        line=dict(width=2)
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100],
            tickfont=dict(size=10)
        ),
        angularaxis=dict(
            tickfont=dict(size=12, family='Arial Black')
        )
    ),
    title={
        'text': '🎯 Mental Health Profile by Cluster (Radar Chart)',
        'font': {'size': 18, 'family': 'Arial Black'}
    },
    showlegend=True,
    width=1000,
    height=800,
    font=dict(family='Arial', size=12)
)

fig.show()

# =============================================================================
# BƯỚC 13: EXPORT
# =============================================================================
print("\n" + "="*80)
print("💾 BƯỚC 13: EXPORT MODELS & DATA")
print("="*80)

folder_name = 'models_mental_health_analysis'
if os.path.exists(folder_name):
    shutil.rmtree(folder_name)
os.makedirs(folder_name)

print("\n📦 Đang lưu các file...")

# Save models
joblib.dump(dt_model, f'{folder_name}/decision_tree.pkl')
print("   ✅ decision_tree.pkl")
joblib.dump(rf_model, f'{folder_name}/random_forest.pkl')
print("   ✅ random_forest.pkl")
joblib.dump(kmeans, f'{folder_name}/kmeans_model.pkl')
print("   ✅ kmeans_model.pkl")
joblib.dump(scaler, f'{folder_name}/scaler.pkl')
print("   ✅ scaler.pkl")
joblib.dump(selected_features, f'{folder_name}/features.pkl')
print("   ✅ features.pkl (16 features bao gồm 5 Mental Health)")
joblib.dump(cluster_to_stress, f'{folder_name}/cluster_to_stress.pkl')
print("   ✅ cluster_to_stress.pkl")

cluster_info = {
    'optimal_k': optimal_k,
    'cluster_names': cluster_names,
    'cluster_profiles': cluster_profiles.to_dict(),
    'silhouette_score': max(silhouette_scores_list),
    'features_used': selected_features,
    'cluster_to_stress_mapping': cluster_to_stress,
    'mental_health_features': selected_features[11:]
}
joblib.dump(cluster_info, f'{folder_name}/cluster_info.pkl')
print("   ✅ cluster_info.pkl")

# Excel report
print("\n📄 Đang tạo báo cáo Excel...")
with pd.ExcelWriter(f'{folder_name}/analysis_report.xlsx', engine='openpyxl') as writer:
    # Sheet 1: Cluster Profiles (detailed)
    cluster_profiles.to_excel(writer, sheet_name='Cluster Profiles', index=False)
    
    # Sheet 2: Cluster Characteristics (readable summary)
    cluster_chars = pd.DataFrame([
        {
            'Cluster': i+1,
            'Name': cluster_names[i].split(': ')[1],
            'Size': cluster_profiles.iloc[i]['Size'],
            'Percentage': cluster_profiles.iloc[i]['Percentage'],
            'Overall_Wellness': f"{cluster_profiles.iloc[i]['Overall_Wellness_Avg']:.1f}",
            'Health_Score': f"{cluster_profiles.iloc[i]['Health_Score_Avg']:.1f}",
            'Digital_Stress': f"{cluster_profiles.iloc[i]['Digital_Stress_Avg']:.1f}",
            'Work_Life_Balance': f"{cluster_profiles.iloc[i]['Work_Life_Balance_Avg']:.1f}",
            'Sleep_Health': f"{cluster_profiles.iloc[i]['Sleep_Health_Avg']:.1f}",
            'Screen_Time_hrs': f"{cluster_profiles.iloc[i]['Screen_Time_Avg']:.1f}",
            'Avg_Age': f"{cluster_profiles.iloc[i]['Age_Avg']:.0f}",
            'Dominant_Stress': target_names[cluster_to_stress[i]]
        }
        for i in range(optimal_k)
    ])
    cluster_chars.to_excel(writer, sheet_name='Cluster Characteristics', index=False)
    
    # Sheet 3: Mental Health Summary
    mental_summary = pd.DataFrame([
        {
            'Cluster': i,
            'Name': cluster_names[i].split(': ')[1].split(' (')[0][:50],
            'Size': len(X_train_df[X_train_df['Cluster']==i]),
            'Health_Score': f"{X_train_df[X_train_df['Cluster']==i]['health_score'].mean():.1f}",
            'Overall_Wellness': f"{X_train_df[X_train_df['Cluster']==i]['overall_wellness'].mean():.1f}",
            'Sleep_Health': f"{X_train_df[X_train_df['Cluster']==i]['sleep_health_index'].mean():.1f}",
            'Emotional_Balance': f"{X_train_df[X_train_df['Cluster']==i]['emotional_balance'].mean():.1f}",
            'Digital_Stress': f"{X_train_df[X_train_df['Cluster']==i]['digital_stress_score'].mean():.1f}",
            'Work_Life_Balance': f"{X_train_df[X_train_df['Cluster']==i]['work_life_balance'].mean():.1f}",
            'Dominant_Stress': target_names[cluster_to_stress[i]]
        }
        for i in range(optimal_k)
    ])
    mental_summary.to_excel(writer, sheet_name='Mental Health Summary', index=False)
    
    # Sheet 4: Model Performance Summary
    model_perf = pd.DataFrame({
        'Model': ['K-Means', 'Decision Tree', 'Random Forest'],
        'Accuracy': [f"{kmeans_acc*100:.2f}%", f"{dt_acc*100:.2f}%", f"{rf_acc*100:.2f}%"],
        'Accuracy_Raw': [kmeans_acc, dt_acc, rf_acc]
    })
    model_perf.to_excel(writer, sheet_name='Model Performance', index=False)
    
    # Sheet 5: Classification Reports (K-Means)
    report_kmeans = classification_report(y_test, y_pred_kmeans, target_names=target_names, output_dict=True)
    report_kmeans_df = pd.DataFrame(report_kmeans).transpose()
    report_kmeans_df.to_excel(writer, sheet_name='KMeans Classification')
    
    # Sheet 6: Classification Reports (Decision Tree)
    report_dt = classification_report(y_test, y_pred_dt, target_names=target_names, output_dict=True)
    report_dt_df = pd.DataFrame(report_dt).transpose()
    report_dt_df.to_excel(writer, sheet_name='DTree Classification')
    
    # Sheet 7: Classification Reports (Random Forest)
    report_rf = classification_report(y_test, y_pred_rf, target_names=target_names, output_dict=True)
    report_rf_df = pd.DataFrame(report_rf).transpose()
    report_rf_df.to_excel(writer, sheet_name='RF Classification')
    
    # Sheet 8: Feature Importance
    feat_imp = pd.DataFrame({
        'Feature': selected_features,
        'Importance': rf_model.feature_importances_,
        'Category': ['Mental Health' if f in selected_features[11:] or f == 'health_score' 
                    else 'Behavioral' 
                    for f in selected_features]
    }).sort_values('Importance', ascending=False)
    feat_imp.to_excel(writer, sheet_name='Feature Importance', index=False)
    
    # Sheet 9: Stress Factors Correlation
    stress_factors = pd.DataFrame({
        'Feature': corr_with_stress_full.index,
        'Correlation': corr_with_stress_full.values,
        'Impact': ['Increases Stress' if x > 0 else 'Reduces Stress' 
                  for x in corr_with_stress_full.values]
    }).sort_values('Correlation', ascending=False)
    stress_factors.to_excel(writer, sheet_name='Stress Factors', index=False)
    
    # Sheet 10: Mental Health Formulas
    formulas = pd.DataFrame({
        'Metric': [
            'health_score',
            'sleep_health_index',
            'emotional_balance',
            'overall_wellness',
            'digital_stress_score',
            'work_life_balance'
        ],
        'Formula': [
            'mental_health_score (0-100)',
            '(sleep_quality/5 * 50 + sleep_duration/10 * 50)',
            'mood_rating * 10 (0-100)',
            '(health_score + sleep_health_index + emotional_balance) / 3',
            '(screen_time/24 * 40 + social_media/10 * 30 + phone/10 * 30)',
            '100 - (work_hours/16 * 100)'
        ]
    })
    formulas.to_excel(writer, sheet_name='Mental Health Formulas', index=False)

print("   ✅ analysis_report.xlsx (10 sheets)")

# README
with open(f'{folder_name}/README.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("STRESS ANALYSIS - COMPREHENSIVE MENTAL HEALTH & MODEL EVALUATION\n")
    f.write("="*80 + "\n\n")
    f.write("📋 16 FEATURES (11 Original + 5 Mental Health):\n\n")
    f.write("🔹 Original Features:\n")
    for i, feat in enumerate(selected_features[:11], 1):
        if feat == 'health_score':
            f.write(f"   {i}. {feat} ⭐ (= mental_health_score)\n")
        else:
            f.write(f"   {i}. {feat}\n")
    f.write("\n💚 Mental Health Features (NEW):\n")
    for i, feat in enumerate(selected_features[11:], 12):
        f.write(f"   {i}. {feat} 🆕\n")
    f.write("\n💡 MENTAL HEALTH FORMULAS:\n")
    f.write("   - health_score: mental_health_score (0-100)\n")
    for _, row in formulas.iterrows():
        if row['Metric'] != 'health_score':
            f.write(f"   - {row['Metric']}: {row['Formula']}\n")
    f.write(f"\n🎯 RESULTS:\n")
    f.write(f"   - Optimal Clusters: {optimal_k}\n")
    f.write(f"   - Best Model: Random Forest ({rf_acc*100:.2f}%)\n")
    f.write(f"   - Decision Tree: {dt_acc*100:.2f}%\n")
    f.write(f"   - K-Means: {kmeans_acc*100:.2f}%\n")
    f.write(f"   - Silhouette Score: {max(silhouette_scores_list):.4f}\n\n")
    f.write("🏷️ INTELLIGENT CLUSTER NAMING:\n")
    f.write("   Clusters are named based on their characteristics:\n")
    f.write("   - Overall Wellness Level (Excellent/Good/Moderate/Fair/Poor)\n")
    f.write("   - Digital Behavior (Heavy/Minimal Tech Users, High Screen Time)\n")
    f.write("   - Work-Life Balance (Well-balanced/Overworked)\n")
    f.write("   - Health Status (Healthy/Health Concerns)\n")
    f.write("   - Sleep Quality (Good Sleep/Sleep Issues)\n")
    f.write("   - Age Category (Youth/Young Adults/Middle-aged/Seniors)\n")
    f.write("   - Stress Level (Low/Medium/High)\n\n")
    f.write("📊 CLUSTER NAMES:\n")
    for i, name in enumerate(cluster_names):
        f.write(f"   {i+1}. {name}\n")
    f.write("\n📈 MODEL EVALUATION:\n")
    f.write("   ✅ Confusion Matrix for all 3 models\n")
    f.write("   ✅ Classification Report (Precision, Recall, F1-Score)\n")
    f.write("   ✅ Per-class performance comparison\n")
    f.write("   ✅ Accuracy comparison visualization\n\n")
    f.write("🔍 STRESS FACTORS ANALYSIS:\n")
    f.write("   ✅ Top 12 features affecting stress\n")
    f.write("   ✅ Positive vs Negative correlations\n")
    f.write("   ✅ Mental Health factors impact\n")
    f.write("   ✅ Digital usage impact (screen time, phone, social media)\n")
    f.write("   ✅ Work-life balance impact\n")
    f.write("   ✅ Age group stress analysis\n")
    f.write("   ✅ Sleep quality impact\n")
    f.write("   ✅ Overall wellness impact\n\n")
    f.write("📊 VISUALIZATION:\n")
    f.write("   - 3D Interactive Scatter Plots\n")
    f.write("   - 9-panel Cluster Characteristics\n")
    f.write("   - Comprehensive Model Evaluation Dashboard (9 panels)\n")
    f.write("   - Stress Factors Analysis Dashboard (8 panels)\n")
    f.write("   - Mental Health Radar Charts\n")
    f.write("   - Feature Importance Analysis\n")
    f.write("   - Cluster Profile Analysis\n\n")
    f.write("📦 FILES:\n")
    f.write("   - *.pkl: Models and preprocessing objects\n")
    f.write("   - analysis_report.xlsx: Comprehensive report (10 sheets)\n")
    f.write("     • Cluster Profiles: Detailed statistics\n")
    f.write("     • Cluster Characteristics: Readable summary with names\n")
    f.write("     • Mental Health Summary: Mental health metrics\n")
    f.write("     • Model Performance: Accuracy comparison\n")
    f.write("     • KMeans Classification: Precision, Recall, F1-Score\n")
    f.write("     • DTree Classification: Precision, Recall, F1-Score\n")
    f.write("     • RF Classification: Precision, Recall, F1-Score\n")
    f.write("     • Feature Importance: Feature ranking\n")
    f.write("     • Stress Factors: Correlation with stress\n")
    f.write("     • Mental Health Formulas: Calculation formulas\n")
    f.write("   - README.txt: This file\n")

print("   ✅ README.txt")

# Zip and download
print("\n📦 Đang nén file...")
shutil.make_archive(folder_name, 'zip', folder_name)
files.download(f'{folder_name}.zip')

print("\n" + "="*80)
print("🎉 HOÀN THÀNH!")
print("="*80)
print(f"\n✨ KẾT QUẢ PHÂN TÍCH:")
print(f"   📊 Tổng số người dùng:     {len(df)}")
print(f"   🎯 Số nhóm tối ưu:         {optimal_k}")
print(f"   📈 Silhouette Score:       {max(silhouette_scores_list):.4f}")
print(f"   🔧 Số features:            {len(selected_features)} (11 original + 5 mental health)")
print(f"\n🏆 ĐỘ CHÍNH XÁC:")
print(f"   Random Forest:             {rf_acc*100:.2f}% ⭐")
print(f"   Decision Tree:             {dt_acc*100:.2f}%")
print(f"   K-Means:                   {kmeans_acc*100:.2f}%")
print(f"\n💚 MENTAL HEALTH FEATURES:")
print(f"   health_score = mental_health_score (không cần tính toán)")
for feat in selected_features[11:]:
    idx = selected_features.index(feat)
    print(f"   {feat}: {importances[idx]*100:.2f}% importance")
print(f"\n🎨 VISUALIZATION:")
print(f"   3D Interactive Plots:      ✅")
print(f"   Radar Charts:              ✅")
print(f"   Feature Importance:        ✅")
print(f"\n📦 File đã tải xuống: {folder_name}.zip")
print("="*80)
