import pandas as pd
import os
from django.conf import settings
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_data():
    """Load dữ liệu từ file CSV"""
    data_path = os.path.join(settings.BASE_DIR, 'salary_prediction', 'data', 'adult.csv')
    df = pd.read_csv(data_path)
    return df

def detect_outliers(data, column):
    """Phát hiện ngoại lai bằng phương pháp IQR"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(Q1 - 1.5 * IQR, 0)
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def remove_outliers_iqr(df, column):
    """Loại bỏ ngoại lai bằng phương pháp IQR"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def preprocess_data():
    """Tiền xử lý dữ liệu hoàn chỉnh"""
    # Load dữ liệu
    df = load_data()
    
    # Loại bỏ khoảng trắng thừa
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    
    # Chuyển đổi dấu "?" thành NaN
    df = df.replace("?", pd.NA)
    
    # Xử lý giá trị thiếu cho các biến phân loại
    categorical_columns = ['workclass', 'occupation', 'native-country']
    for col in categorical_columns:
        if df[col].isnull().sum() > 0:
            # Tính mode
            mode_value = df[col].mode()[0]
            # Điền giá trị thiếu bằng mode
            df[col].fillna(mode_value, inplace=True)
    
    # Xóa bản ghi nếu có biến số thiếu
    df = df.dropna()
    
    # Áp dụng xử lý ngoại lai cho fnlwgt và capital-gain
    df = df.pipe(remove_outliers_iqr, 'fnlwgt').pipe(remove_outliers_iqr, 'capital-gain')
    
    # Loại bỏ các cột không cần thiết
    cols_to_drop = ['capital-gain', 'fnlwgt']
    df = df.drop(columns=cols_to_drop)
    
    # Loại bỏ các trường hợp không phù hợp
    df = df.drop(df[df['workclass'] == 'Without-pay'].index)
    df = df.drop(df[df['workclass'] == 'Never-worked'].index)
    
    # Loại bỏ cột educational-num vì trùng lặp với education
    df.drop(['educational-num'], axis=1, inplace=True)
    
    return df

def encode_features(df_clean):
    """Mã hóa các đặc trưng"""
    le = LabelEncoder()
    df_encoded = df_clean.copy()
    
    # Encode income (biến mục tiêu): <=50K -> 0, >50K -> 1
    if 'income' in df_encoded.columns:
        df_encoded['income'] = le.fit_transform(df_encoded['income'])
    
    # One-Hot Encoding cho các biến phân loại
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'gender', 'native-country']
    
    # Thực hiện One-Hot Encoding
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, prefix=categorical_cols)
    
    return df_encoded, le

def prepare_user_input(user_data):
    """Chuẩn bị dữ liệu người dùng nhập vào để dự đoán"""
    # Tạo DataFrame từ dữ liệu người dùng
    df = pd.DataFrame([user_data])
    
    # Đổi tên cột để phù hợp với dataset gốc
    df = df.rename(columns={
        'marital_status': 'marital-status',
        'capital_gain': 'capital-gain', 
        'capital_loss': 'capital-loss',
        'hours_per_week': 'hours-per-week',
        'native_country': 'native-country'
    })
    
    # Chuẩn hóa dữ liệu như trong preprocessing
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    
    return df