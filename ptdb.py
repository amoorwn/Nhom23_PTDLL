import os
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib



# # Đọc dữ liệu đã tiền xử lý
df = pd.read_csv('adult2.csv')
# Chuyển income thành 0/1 để dùng cho mô hình
df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

df = pd.get_dummies(df, drop_first=True)  # loại bỏ 1 biến dummy để tránh đa cộng tuyến

# # Tách biến độc lập và biến phụ thuộc
X = df.drop('income', axis=1)
y = df['income']

# 3. Chia train và test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Khởi tạo và huấn luyện mô hình Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(
    solver='saga', 
    max_iter=1000, 
    random_state=42)
model.fit(X_train_scaled, y_train)


# from sklearn.metrics import roc_curve, roc_auc_score, auc

# import matplotlib.pyplot as plt

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]


# # Dự đoán xác suất (tuỳ chọn nếu cần vẽ ROC hoặc AUC)

# fpr, tpr, thresholds = roc_curve(y_test, y_proba)
# roc_auc = auc(fpr, tpr)
# #
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # đường chéo
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC)')
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.show()
# 6. Tạo thư mục nếu chưa có
# os.makedirs('../models', exist_ok=True)
# 7. Lưu mô hình và scaler
import joblib
joblib.dump(model, '../models/logistic_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')
print("Đã lưu mô hình vào ../models/logistic_model.pkl")
print("Đã lưu scaler vào ../models/scaler.pkl")

# 8. In kết quả cơ bản
# print("Độ chính xác:", accuracy_score(y_test, y_pred))
# print("Báo cáo phân loại:")
# print(classification_report(y_test, y_pred))
# print("Ma trận nhầm lẫn:")
# print(confusion_matrix(y_test, y_pred))




# print(f"Tập huấn luyện: {X_train.shape[0]} mẫu")
# print(f"Tập kiểm tra: {X_test.shape[0]} mẫu")
# print(f"Tỷ lệ phân chia: {X_train.shape[0]/(X_train.shape[0]+X_test.shape[0])*100:.1f}% : {X_test.shape[0]/(X_train.shape[0]+X_test.shape[0])*100:.1f}%")
#
# # Kiểm tra phân phối biến mục tiêu
# print("\nPhân phối biến mục tiêu trong tập huấn luyện:")
# print(y_train.value_counts(normalize=True))
# print("\nPhân phối biến mục tiêu trong tập kiểm tra:")
# print(y_test.value_counts(normalize=True))
#
# from imblearn.over_sampling import RandomOverSampler
# from collections import Counter
#
# # Kiểm tra phân phối ban đầu
# print("Phân phối ban đầu của tập huấn luyện:")
# print(Counter(y_train))
#
# # Áp dụng Random OverSampling
# ros = RandomOverSampler(random_state=42)
# X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)
#
# print("\nPhân phối sau khi cân bằng:")
# print(Counter(y_train_balanced))
#
# # Tính toán thay đổi
# original_samples = len(y_train)
# balanced_samples = len(y_train_balanced)
# print(f"\nSố mẫu tăng từ {original_samples} lên {balanced_samples}")
# print(f"Tỷ lệ tăng: {(balanced_samples/original_samples-1)*100:.1f}%")
#
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import pandas as pd
#
# # Đọc lại dữ liệu gốc đã làm sạch
# df_clean = pd.read_csv('adult_income_processed.csv')
# #
# # # Bước 1: Loại bỏ các biến không cần thiết
# columns_to_drop = ['fnlwgt', 'education-num']  # Dựa trên phân tích từ Chương 2
# df_selected = df_clean.drop(columns=columns_to_drop, errors='ignore')
#
# print(f"Số đặc trưng sau khi loại bỏ: {len(df_selected.columns)-1}")
#
# # Bước 2: Mã hóa biến phân loại
# categorical_columns = df_selected.select_dtypes(include=['object']).columns.tolist()
# if 'income' in categorical_columns:
#     categorical_columns.remove('income')
#
# print(f"Các biến phân loại cần mã hóa: {categorical_columns}")
#
# # One-Hot Encoding
# df_encoded = pd.get_dummies(df_selected, columns=categorical_columns, prefix=categorical_columns)
# #
# # Bước 3: Chuẩn hóa dữ liệu số
# numerical_cols = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
# scaler = StandardScaler()
# df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
#
# print(f"Số đặc trưng cuối cùng: {len(df_encoded.columns)-1}")
# print(f"Kích thước dữ liệu: {df_encoded.shape}")
#
# # Hiển thị một số đặc trưng được tạo
# feature_names = [col for col in df_encoded.columns if col != 'income']
# print("Ví dụ các đặc trưng:")
# for col in X.columns[:10]:  # in 10 cột đầu tiên
#     print("-", col)
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# import time
#
# # Chuẩn bị dữ liệu đã được xử lý
# X = df_encoded.drop('income', axis=1)
# y = df_encoded['income']
#
# # Chia tách dữ liệu
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )
#
# # Khởi tạo mô hình hồi quy logistic cơ bản
# logistic_model = LogisticRegression(
#     solver='liblinear',
#     max_iter=1000,
#     random_state=42,
#     class_weight='balanced'
# )
#
# print("Thông số mô hình:")
# print(f"Solver: {logistic_model.solver}")
# print(f"Max iterations: {logistic_model.max_iter}")
# print(f"Class weight: {logistic_model.class_weight}")
# print(f"Regularization: L2 (mặc định)")
# print(f"Regularization strength (C): {logistic_model.C}")
#
# import time
# from sklearn.metrics import accuracy_score, classification_report
#
# # Ghi nhận thời gian bắt đầu
# start_time = time.time()
#
# # Huấn luyện mô hình
# print("Bắt đầu huấn luyện mô hình...")
# logistic_model.fit(X_train, y_train)
#
# # Tính thời gian huấn luyện
# training_time = time.time() - start_time
#
# print(f"Thời gian huấn luyện: {training_time:.2f} giây")
# print(f"Số vòng lặp hội tụ: {logistic_model.n_iter_[0]}")
#
# # Đánh giá nhanh trên tập huấn luyện
# y_train_pred = logistic_model.predict(X_train)
# train_accuracy = accuracy_score(y_train, y_train_pred)
# print(f"Độ chính xác trên tập huấn luyện: {train_accuracy:.4f}")
#
# # Đánh giá nhanh trên tập kiểm tra
# y_test_pred = logistic_model.predict(X_test)
# test_accuracy = accuracy_score(y_test, y_test_pred)
# print(f"Độ chính xác trên tập kiểm tra: {test_accuracy:.4f}")
#
# print("\nMô hình đã được huấn luyện thành công!")
# print("\n")
#
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer, f1_score
# import numpy as np
#
# # Định nghĩa không gian tìm kiếm
# param_grid = {
#     'C': [0.1, 1.0, 10.0, 100.0],
#     'penalty': ['l1', 'l2'],
#     'solver': ['liblinear', 'saga']
# }
#
# # Tạo scorer F1 cho lớp dương (>50K)
# f1_scorer = make_scorer(f1_score, pos_label=1)
#
# # Khởi tạo GridSearchCV
# grid_search = GridSearchCV(
#     estimator=LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
#     param_grid=param_grid,
#     cv=5,
#     scoring=f1_scorer,
#     n_jobs=-1,
#     verbose=1
# )
#
# print("Bắt đầu tối ưu hóa siêu tham số...")
# start_time = time.time()
#
# # Thực hiện grid search
# grid_search.fit(X_train, y_train)
#
# optimization_time = time.time() - start_time
#
# print(f"Thời gian tối ưu hóa: {optimization_time:.2f} giây")
# print(f"Tham số tốt nhất: {grid_search.best_params_}")
# print(f"F1-score tốt nhất: {grid_search.best_score_:.4f}")
#
# # Lấy mô hình tối ưu
# best_model = grid_search.best_estimator_
#
# # Đánh giá mô hình tối ưu
# y_train_pred_best = best_model.predict(X_train)
# y_test_pred_best = best_model.predict(X_test)
#
# train_accuracy_best = accuracy_score(y_train, y_train_pred_best)
# test_accuracy_best = accuracy_score(y_test, y_test_pred_best)
#
# print(f"\nHiệu suất mô hình tối ưu:")
# print(f"Độ chính xác trên tập huấn luyện: {train_accuracy_best:.4f}")
# print(f"Độ chính xác trên tập kiểm tra: {test_accuracy_best:.4f}")
#
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# from sklearn.metrics import precision_score, recall_score, f1_score
# import numpy as np
#
# # Dự đoán trên tập kiểm tra với mô hình tối ưu
# y_pred = best_model.predict(X_test)
# y_pred_proba = best_model.predict_proba(X_test)[:, 1]
#
# # Tính toán confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# tn, fp, fn, tp = cm.ravel()
#
# print("Confusion Matrix:")
# print(f"True Negative (TN): {tn}")
# print(f"False Positive (FP): {fp}")
# print(f"False Negative (FN): {fn}")
# print(f"True Positive (TP): {tp}")
#
# # Tính toán các chỉ số đánh giá

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)
# specificity = tn / (tn + fp)
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)
#
# print(f"\n=== CHỈ SỐ ĐÁNH GIÁ CƠ BẢN ===")
# print(f"Accuracy (Độ chính xác tổng thể): {accuracy:.4f}")
# print(f"Precision (Độ chính xác dương): {precision:.4f}")
# print(f"Recall/Sensitivity (Độ nhạy): {recall:.4f}")
# print(f"Specificity (Độ đặc hiệu): {specificity:.4f}")
# print(f"F1-Score: {f1:.4f}")
#
# # Báo cáo phân loại chi tiết
# print(f"\n=== BÁO CÁO PHÂN LOẠI CHI TIẾT ===")
# print(classification_report(y_test, y_pred, target_names=['≤50K', '>50K']))
#
# # Tính toán thêm một số chỉ số
# balanced_accuracy = (recall + specificity) / 2
# print(f"\nBalanced Accuracy: {balanced_accuracy:.4f}")
#
# # Phân tích ý nghĩa
# print(f"\n=== PHÂN TÍCH Ý NGHĨA ===")
# print(f"• Trong {len(y_test)} trường hợp kiểm tra:")
# print(f"  - Dự đoán đúng: {tp + tn} ({(tp + tn)/len(y_test)*100:.1f}%)")
# print(f"  - Dự đoán sai: {fp + fn} ({(fp + fn)/len(y_test)*100:.1f}%)")
# print(f"• Với lớp >50K (mục tiêu quan tâm):")
# print(f"  - Phát hiện đúng: {tp}/{tp + fn} ({recall*100:.1f}%)")
# print(f"  - Độ chính xác khi dự đoán: {tp}/{tp + fp} ({precision*100:.1f}%)")
