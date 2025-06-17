from django.db import models

# Các models có thể cần cho tương lai
class PredictionHistory(models.Model):
    """Lưu lịch sử dự đoán (tùy chọn cho tương lai)"""
    age = models.IntegerField(verbose_name="Tuổi")
    workclass = models.CharField(max_length=50, verbose_name="Lớp công việc")
    education = models.CharField(max_length=50, verbose_name="Trình độ học vấn") 
    marital_status = models.CharField(max_length=50, verbose_name="Tình trạng hôn nhân")
    occupation = models.CharField(max_length=50, verbose_name="Nghề nghiệp")
    relationship = models.CharField(max_length=50, verbose_name="Mối quan hệ")
    race = models.CharField(max_length=50, verbose_name="Chủng tộc")
    gender = models.CharField(max_length=10, verbose_name="Giới tính")
    capital_gain = models.IntegerField(verbose_name="Thu nhập từ vốn")
    capital_loss = models.IntegerField(verbose_name="Lỗ vốn")
    hours_per_week = models.IntegerField(verbose_name="Số giờ làm việc/tuần")
    native_country = models.CharField(max_length=50, verbose_name="Quốc gia")
    
    prediction_probability = models.FloatField(verbose_name="Xác suất dự đoán")
    prediction_result = models.BooleanField(verbose_name="Kết quả (>50K)")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Thời gian dự đoán")
    
    class Meta:
        verbose_name = "Lịch sử dự đoán"
        verbose_name_plural = "Lịch sử dự đoán"
        ordering = ['-created_at']