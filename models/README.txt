================================================================================
STRESS ANALYSIS - MENTAL HEALTH ANALYSIS WITH 3D VISUALIZATION
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
   - Random Forest Accuracy: 96.80%
   - Silhouette Score: 0.2601

📊 VISUALIZATION:
   - 3D Interactive Scatter Plots
   - Mental Health Radar Charts
   - Feature Importance Analysis
   - Cluster Profile Analysis

📦 FILES:
   - *.pkl: Models and preprocessing objects
   - analysis_report.xlsx: Detailed report (5 sheets)
   - README.txt: This file
