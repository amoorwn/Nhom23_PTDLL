from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
from django.conf import settings
from .preprocessing import preprocess_data, encode_features

def load_preprocessed_data():
    """Đọc và tiền xử lý dữ liệu hoàn chỉnh"""
    # Tiền xử lý dữ liệu theo đúng logic gốc
    df_clean = preprocess_data()
    
    # Mã hóa các đặc trưng
    df_encoded, le = encode_features(df_clean)
    
    # Tách biến độc lập và biến phụ thuộc
    X = df_encoded.drop('income', axis=1)
    y = df_encoded['income']
    
    return X, y

def train_model():
    """Huấn luyện mô hình"""
    # Load dữ liệu đã tiền xử lý
    X, y = load_preprocessed_data()
    
    # Chia train và test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Khởi tạo và huấn luyện mô hình Logistic Regression
    model = LogisticRegression(
        solver='liblinear',
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)

    # Đánh giá mô hình
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    return model, scaler, X.columns

def save_model(model, scaler, feature_columns):
    """Lưu mô hình và scaler"""
    # Tạo thư mục models nếu chưa tồn tại
    models_dir = os.path.join(settings.BASE_DIR, 'salary_prediction', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Lưu mô hình, scaler và danh sách cột
    joblib.dump(model, os.path.join(models_dir, 'logistic_model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(list(feature_columns), os.path.join(models_dir, 'feature_columns.pkl'))
    
def load_model():
    """Load mô hình đã lưu"""
    models_dir = os.path.join(settings.BASE_DIR, 'salary_prediction', 'models')
    
    try:
        model = joblib.load(os.path.join(models_dir, 'logistic_model.pkl'))
        scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        feature_columns = joblib.load(os.path.join(models_dir, 'feature_columns.pkl'))
        return model, scaler, feature_columns
    except:
        # Nếu chưa có mô hình, train mới
        print("Training new model...")
        model, scaler, feature_columns = train_model()
        save_model(model, scaler, feature_columns)
        return model, scaler, feature_columns

def predict_income(user_data):
    """Dự đoán thu nhập dựa trên dữ liệu người dùng"""
    from .preprocessing import prepare_user_input
    import numpy as np
    
    try:
        # Load mô hình
        model, scaler, feature_columns = load_model()
        
        # Chuẩn bị dữ liệu người dùng
        user_df = prepare_user_input(user_data)
        
        # One-hot encoding cho dữ liệu người dùng
        user_df_encoded = pd.get_dummies(user_df)
        
        # Đảm bảo có đủ các cột như khi train
        for col in feature_columns:
            if col not in user_df_encoded.columns:
                user_df_encoded[col] = 0
        
        # Chỉ giữ lại các cột có trong feature_columns và sắp xếp theo thứ tự
        user_df_encoded = user_df_encoded.reindex(columns=feature_columns, fill_value=0)
        
        # Chuẩn hóa dữ liệu
        user_scaled = scaler.transform(user_df_encoded)
        
        # Dự đoán
        probability = model.predict_proba(user_scaled)[0, 1]
        prediction = model.predict(user_scaled)[0]
        
        return {
            'probability': float(probability),
            'prediction': bool(prediction),
            'probability_percent': round(probability * 100, 2)
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        # Trả về kết quả mặc định nếu có lỗi
        return {
            'probability': 0.5,
            'prediction': False,
            'probability_percent': 50.0
        }