================================================================================
STRESS ANALYSIS - MODELS WITH HEALTH_SCORE
================================================================================

📋 11 FEATURES SỬ DỤNG:
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
   11. health_score ⭐ (Calculated)

💡 CÔNG THỨC HEALTH_SCORE:
   health_score = (sleep_quality*20 + mood_rating*10 + mental_health_score)/3

🎯 KẾT QUẢ:
   - Số cụm tối ưu: 3
   - Random Forest Accuracy: 96.30%
   - Silhouette Score: 0.1862

📦 FILES:
   - *.pkl: Models và preprocessing objects
   - cluster_analysis_report.xlsx: Báo cáo chi tiết (5 sheets)
   - README.txt: File này
