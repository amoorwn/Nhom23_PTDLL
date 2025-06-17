# Dự án phân tích và dự đoán mức lương

## Giới thiệu
Dự án này xây dựng một ứng dụng web để dự đoán xác suất thu nhập trên 50,000 USD/năm dựa trên mô hình hồi quy logistic, sử dụng Adult Income Dataset từ Kaggle.

## Tính năng
- Form nhập liệu thân thiện với người dùng
- Dự đoán xác suất thu nhập > 50K USD/năm
- Hiển thị kết quả trực quan với thanh tiến trình
- Trang thống kê về dữ liệu
- Giao diện responsive với Bootstrap

## Cấu trúc dự án
```
PTDLL/
├── salary_prediction_project/     # Cấu hình Django project
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── salary_prediction/             # Ứng dụng Django
│   ├── data/                     # Dữ liệu
│   │   └── adult.csv
│   ├── models/                   # Mô hình đã train
│   ├── static/                   # File tĩnh (CSS, JS)
│   ├── templates/                # Templates HTML
│   ├── utils/                    # Các module xử lý
│   │   ├── preprocessing.py      # Tiền xử lý dữ liệu
│   │   ├── analysis.py          # Phân tích mô tả
│   │   └── prediction.py        # Logic dự đoán
│   ├── models.py
│   ├── views.py
│   ├── forms.py
│   └── urls.py
├── manage.py
├── requirements.txt
├── setup_and_run.bat            # Script cài đặt và chạy
└── run_server.bat               # Script chạy server

```

## Yêu cầu hệ thống
- Python 3.8 trở lên
- Windows OS (cho file .bat)
- 1GB RAM tối thiểu

## Cài đặt và sử dụng

### Cách 1: Khởi động nhanh (Windows)
1. **`start_server.bat`** (KHUYẾN NGHỊ) - Đơn giản và ổn định nhất
2. **`fix_issues.bat`** - Nếu gặp lỗi, dùng để khắc phục tự động  
3. `quick_start.bat` - Phiên bản cơ bản nhất
4. `setup_and_run.bat` - Phiên bản đầy đủ với môi trường ảo

### Cách 2: Nếu gặp lỗi cài đặt
```bash
# Cài đặt từng thư viện một cách thủ công
pip install Django
pip install pandas
pip install scikit-learn
pip install joblib

# Sau đó chạy:
python manage.py runserver
```

### Cách 3: Cài đặt thủ công đầy đủ
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo (Windows)
venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt

# Chạy migration
python manage.py migrate

# Khởi động server
python manage.py runserver
```

### Khắc phục lỗi thường gặp

**Lỗi: "No module named 'django'"**
```bash
pip install Django
```

**Lỗi: "No module named 'pandas'"**
```bash
pip install pandas scikit-learn joblib
```

**Lỗi cài đặt numpy trên Windows:**
- Cập nhật pip: `python -m pip install --upgrade pip`
- Hoặc sử dụng: `pip install numpy --only-binary=numpy`

## Sử dụng
1. Mở trình duyệt và truy cập: http://127.0.0.1:8000
2. Điền thông tin cá nhân vào form
3. Nhấn nút "Dự đoán" để xem kết quả
4. Khám phá các trang:
   - Trang chủ: Form dự đoán
   - Giới thiệu: Thông tin về dự án
   - Thống kê: Các số liệu về dataset

## Các thông tin cần nhập
- **Tuổi**: 17-90
- **Lớp công việc**: Tư nhân, Chính phủ, Tự kinh doanh, v.v.
- **Trình độ học vấn**: Từ mầm non đến Tiến sĩ
- **Tình trạng hôn nhân**: Đã kết hôn, Độc thân, Ly hôn, v.v.
- **Nghề nghiệp**: 14 loại nghề nghiệp khác nhau
- **Mối quan hệ**: Vợ, Chồng, Con, v.v.
- **Chủng tộc**: 5 nhóm chủng tộc
- **Giới tính**: Nam/Nữ
- **Thu nhập từ vốn**: Số tiền thu nhập từ đầu tư
- **Lỗ vốn**: Số tiền lỗ từ đầu tư
- **Số giờ làm việc/tuần**: 1-99 giờ
- **Quốc gia**: 41 quốc gia khác nhau

## Thư viện sử dụng
- Django 5.0.1: Framework web
- pandas 2.1.4: Xử lý dữ liệu
- scikit-learn 1.4.0: Machine learning
- numpy 1.26.3: Tính toán số học
- matplotlib & seaborn: Visualization
- joblib: Lưu/load mô hình

## Lưu ý
- Lần đầu chạy, hệ thống sẽ tự động train mô hình từ dữ liệu
- Mô hình được lưu cache để tăng tốc độ dự đoán
- Dữ liệu đã được tiền xử lý và chuẩn hóa

## Tác giả
Sinh viên thực hiện bài tập lớn môn học

## License
Dự án học tập - Không sử dụng cho mục đích thương mại