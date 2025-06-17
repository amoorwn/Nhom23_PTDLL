from django.contrib import admin
from .models import PredictionHistory

# Register your models here.
@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ['created_at', 'age', 'occupation', 'education', 'prediction_probability', 'prediction_result']
    list_filter = ['prediction_result', 'created_at', 'occupation', 'education']
    search_fields = ['occupation', 'education']
    ordering = ['-created_at']