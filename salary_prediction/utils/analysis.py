import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('adult1.csv')

## Phân tích phân phối của biến thu nhập
# rcParams['figure.figsize'] = 15, 8
# df[['income']].hist()
# plt.show()


# #phân phối các biến số boxplot

# rcParams['figure.figsize'] = 15, 8
# df[['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']].boxplot()
# plt.show()


# #phân phối các biến số
# rcParams['figure.figsize'] = 15, 8
# df[['age','fnlwgt','educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']].hist()
# plt.suptitle("Phân phối các biến số")  # (tuỳ chọn tiêu đề)
# plt.tight_layout()
# plt.show()

# xử lí giá trị ngoại lai
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Áp dụng hàm xử lý cho fnlwgt và capital-gain
df = df.pipe(remove_outliers_iqr, 'fnlwgt').pipe(remove_outliers_iqr, 'capital-gain')

# #phân phối của các biến phân loại
# def plot_categorical_distributions(df):
#     categorical_cols = df.select_dtypes(include='object').columns.tolist()
#     n = len(categorical_cols)
#     rows = (n + 2) // 3
#
#     plt.figure(figsize=(18, 4 * rows))
#     for i, col in enumerate(categorical_cols):
#         plt.subplot(rows, 3, i + 1)
#         sns.countplot(y=col, data=df, order=df[col].value_counts().index)
#         plt.title(f'Phân phối: {col}')
#     plt.tight_layout()
#     plt.show()
# plot_categorical_distributions(df)

# print((df['fnlwgt']==0).value_counts())
# #mối quan hệ giữa biến đầu vào và thu nhập
# num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
# # Đảm bảo cột income có trong num_cols
# if 'income' not in num_cols:
#     print("Cột 'income' không phải biến số hoặc không tồn tại.")
# else:
#     # Tính ma trận tương quan chỉ giữa các biến số
#     corr = df[num_cols].corr()
#
#     # Lấy cột tương quan với income
#     corr_income = corr[['income']].sort_values(by='income', ascending=False)
#
#     # Vẽ heatmap
#     plt.figure(figsize=(8, 12))
#     sns.heatmap(corr_income, annot=True, cmap='coolwarm', center=0)
#     plt.title('Heatmap correlation of numerical features with Income')
#     plt.show()

cols_to_drop = ['capital-gain', 'fnlwgt']
df = df.drop(columns=cols_to_drop)

# ##tác động của ca yếu tố chính đến thu nhập
# print(df.columns)

# # df['age'] = pd.cut(df['age'], bins = [0, 25, 50, 100], labels = ['Young', 'Adult', 'Old'])
# # sns.countplot(x = 'age', hue = 'income', data = df)
# #
# # df['hours-per-week'] = pd.cut(df['hours-per-week'],
# #                                    bins = [0, 30, 40, 100],
#                                    labels = ['Lesser Hours', 'Normal Hours', 'Extra Hours'])
# sns.countplot(x = 'hours-per-week', hue = 'income', data = df)

# sns.countplot(x = 'workclass', hue = 'income', data = df)
df = df.drop(df[df['workclass'] == ' Without-pay'].index)
df = df.drop(df[df['workclass'] == ' Never-worked'].index)


# sns.countplot(x = 'educational-num', hue = 'income', data = df)

# education_classes = df['education'].unique()
# for edu_class in education_classes:
#     print("Trình độ {} có số năm học tập là {}"
#           .format(edu_class, df[df['education'] == edu_class]['educational-num'].unique()))

# sns.countplot(x = 'education', hue = 'income', data = df)
# education_classes = df['education'].unique()
# for edu_class in education_classes:
#     print("Trình độ {} có số năm học tập là {}"
#           .format(edu_class, df[df['education'] == edu_class]['education_num'].unique()))

df.drop(['educational-num'], axis = 1, inplace = True)

# plt.xticks(rotation = 45)
# sns.countplot(x = 'occupation', hue = 'income', data = df)

# df['race'].unique()
# df['race'] = df['race'].replace(['Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], 'Other')
# sns.countplot(x = 'race', hue = 'income', data = df)

# sns.countplot(x = 'gender', hue = 'income', data = df)
import matplotlib.pyplot as plt

# Đếm số người theo quốc gia
# country_counts = df['native-country'].value_counts()
#
# # Vẽ biểu đồ
# plt.figure(figsize=(12, 6))
# plt.bar(country_counts.index, country_counts.values)
# plt.xticks(rotation=90)
# plt.xlabel('Quốc gia')
# plt.ylabel('Số người')
# plt.title('Tổng số người lớn từ mỗi quốc gia')
# plt.tight_layout()
#
# df_plot = df.copy()
#
# # Bước 2: Gộp tất cả các quốc gia != 'United-States' thành 'Other'
# df_plot['native-country'] = df_plot['native-country'].apply(lambda x: x if x == 'United-States' else 'Other')
#
# # Bước 3: Vẽ biểu đồ countplot với hue = income
# plt.figure(figsize=(6, 5))
# sns.countplot(x='native-country', hue='income', data=df_plot)
# plt.xlabel('Quốc gia')
# plt.ylabel('Số lượng')
# plt.title('Phân phối thu nhập theo quốc gia (Mỹ vs Khác)')
# plt.tight_layout()
# plt.show()
#
# plt.show()
df.to_csv('adult2.csv', index=False)


