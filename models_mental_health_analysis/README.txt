================================================================================
STRESS ANALYSIS - COMPREHENSIVE MENTAL HEALTH & MODEL EVALUATION
================================================================================

📋 16 FEATURES (11 Original + 5 Mental Health):

🔹 Original Features:
   1. age
   2. gender
   3. daily_screen_time_hours
   4. sleep_duration_hours
   5. social_media_hours
   6. work_related_hours
   7. gaming_hours
   8. phone_usage_hours
   9. laptop_usage_hours
   10. sleep_quality
   11. health_score ⭐ (= mental_health_score)

💚 Mental Health Features (NEW):
   12. sleep_health_index 🆕
   13. emotional_balance 🆕
   14. overall_wellness 🆕
   15. digital_stress_score 🆕
   16. work_life_balance 🆕

💡 MENTAL HEALTH FORMULAS:
   - health_score: mental_health_score (0-100)
   - sleep_health_index: (sleep_quality/5 * 50 + sleep_duration/10 * 50)
   - emotional_balance: mood_rating * 10 (0-100)
   - overall_wellness: (health_score + sleep_health_index + emotional_balance) / 3
   - digital_stress_score: (screen_time/24 * 40 + social_media/10 * 30 + phone/10 * 30)
   - work_life_balance: 100 - (work_hours/16 * 100)

🎯 RESULTS:
   - Optimal Clusters: 3
   - Best Model: Random Forest (96.80%)
   - Decision Tree: 94.70%
   - K-Means: 86.40%
   - Silhouette Score: 0.2601

🏷️ INTELLIGENT CLUSTER NAMING:
   Clusters are named based on their characteristics:
   - Overall Wellness Level (Excellent/Good/Moderate/Fair/Poor)
   - Digital Behavior (Heavy/Minimal Tech Users, High Screen Time)
   - Work-Life Balance (Well-balanced/Overworked)
   - Health Status (Healthy/Health Concerns)
   - Sleep Quality (Good Sleep/Sleep Issues)
   - Age Category (Youth/Young Adults/Middle-aged/Seniors)
   - Stress Level (Low/Medium/High)

📊 CLUSTER NAMES:
   1. Cluster 1: 🟡 Fair - High Screen Time - Well-balanced (Young Adults, High Stress)
   2. Cluster 2: 🌟 Excellent - Minimal Tech Users - Well-balanced - Healthy - Good Sleep (Middle-aged, Low Stress)
   3. Cluster 3: ✅ Good - Well-balanced - Good Sleep (Middle-aged, Medium Stress)

📈 MODEL EVALUATION:
   ✅ Confusion Matrix for all 3 models
   ✅ Classification Report (Precision, Recall, F1-Score)
   ✅ Per-class performance comparison
   ✅ Accuracy comparison visualization

🔍 STRESS FACTORS ANALYSIS:
   ✅ Top 12 features affecting stress
   ✅ Positive vs Negative correlations
   ✅ Mental Health factors impact
   ✅ Digital usage impact (screen time, phone, social media)
   ✅ Work-life balance impact
   ✅ Age group stress analysis
   ✅ Sleep quality impact
   ✅ Overall wellness impact

📊 VISUALIZATION:
   - 3D Interactive Scatter Plots
   - 9-panel Cluster Characteristics
   - Comprehensive Model Evaluation Dashboard (9 panels)
   - Stress Factors Analysis Dashboard (8 panels)
   - Mental Health Radar Charts
   - Feature Importance Analysis
   - Cluster Profile Analysis

📦 FILES:
   - *.pkl: Models and preprocessing objects
   - analysis_report.xlsx: Comprehensive report (10 sheets)
     • Cluster Profiles: Detailed statistics
     • Cluster Characteristics: Readable summary with names
     • Mental Health Summary: Mental health metrics
     • Model Performance: Accuracy comparison
     • KMeans Classification: Precision, Recall, F1-Score
     • DTree Classification: Precision, Recall, F1-Score
     • RF Classification: Precision, Recall, F1-Score
     • Feature Importance: Feature ranking
     • Stress Factors: Correlation with stress
     • Mental Health Formulas: Calculation formulas
   - README.txt: This file
