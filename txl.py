import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu từ file CSV
df = pd.read_csv('salary_prediction/data/adult.csv')

# Loại bỏ khoảng trắng thừa
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

# Kiểm tra và xóa các dữ liệu trùng lặp
print(f"Số bản ghi trước khi xử lí: {len(df)}")
df = df.drop_duplicates()
print(f"Số bản ghi sau khi xử lí: {len(df)}")
print(f"Số bản ghi trùng lặp : {52}")  # hoặc len(df_ban_dau) - len(df)

# Chuyển dấu '?' thành NaN
df = df.replace("?", pd.NA)

# Xử lý giá trị thiếu cho các biến phân loại
categorical_columns = ['workclass', 'occupation', 'native-country']
for col in categorical_columns:
    if df[col].isnull().sum() > 0:
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value)  # Sửa inplace cảnh báo
        print(f"Đã điền {col} bằng '{mode_value}'")

# Label Encoding cho biến mục tiêu
le = LabelEncoder()
df['income'] = le.fit_transform(df['income'])
print(f"Income mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Tạo thư mục nếu chưa có
output_dir = os.path.join('salary_prediction', 'data')
os.makedirs(output_dir, exist_ok=True)

# Lưu dữ liệu đã xử lý
df.to_csv('salary_prediction/data/adult1.csv', index=False)
print("\nDữ liệu đã được lưu vào 'adult1.csv'")
