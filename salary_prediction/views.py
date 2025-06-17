from django.shortcuts import render
from django.http import JsonResponse
from .forms import SalaryPredictionForm
from .utils.prediction import predict_income
import json
from .utils.visualization import (
    plot_income_pie, plot_income_by_gender, plot_income_by_hours,
    plot_income_by_age, plot_income_by_country
)
import pandas as pd


def index(request):
    """Trang chủ với form dự đoán"""
    if request.method == 'POST':
        form = SalaryPredictionForm(request.POST)
        if form.is_valid():
            try:
                # Lấy dữ liệu từ form
                user_data = form.cleaned_data
                
                # Dự đoán
                result = predict_income(user_data)
                
                # Debug: In kết quả dự đoán
                print(f"Prediction result: {result}")
                
                # Nếu là AJAX request, trả về JSON
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse(result)
                
                # Nếu không, render lại trang với kết quả
                context = {
                    'form': form,
                    'result': result,
                    'show_result': True
                }
                return render(request, 'salary_prediction/index.html', context)
                
            except Exception as e:
                # Xử lý lỗi
                print(f"Prediction error: {e}")
                context = {
                    'form': form,
                    'error_message': 'Có lỗi xảy ra trong quá trình dự đoán. Vui lòng thử lại.',
                    'show_result': False
                }
                return render(request, 'salary_prediction/index.html', context)
    else:
        form = SalaryPredictionForm()
    
    context = {
        'form': form,
        'show_result': False
    }
    return render(request, 'salary_prediction/index.html', context)

def about(request):
    """Trang giới thiệu về dự án"""
    return render(request, 'salary_prediction/about.html')

def statistics(request):
    """Trang thống kê dữ liệu"""
    df = pd.read_csv('salary_prediction/data/adult.csv')

    # Biểu đồ
    context = {
        'income_pie': plot_income_pie(df),
        'income_gender': plot_income_by_gender(df),
        'income_hours': plot_income_by_hours(df),
        'income_age': plot_income_by_age(df),
        'income_country': plot_income_by_country(df),
    }

    # Lọc cột số để mô tả thống kê
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Tính thống kê
    desc = df[numeric_cols].describe().T
    desc['median'] = df[numeric_cols].median()
    desc = desc[['min', 'max', 'mean', 'median', 'std']].round(2).reset_index()
    desc.columns = ['feature', 'min', 'max', 'mean', 'median', 'std']

    # Thêm vào context để render bảng
    context['describe_data'] = desc.to_dict(orient='records')

    return render(request, 'salary_prediction/statistics.html', context)
