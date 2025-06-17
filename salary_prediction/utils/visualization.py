import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import pandas as pd
colors = ['#ff9999', '#66b3ff']  # đỏ hồng cho <=50K, xanh cho >50K

def plot_to_base64():
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return image_base64

# Biểu đồ pie
def plot_income_pie(df):
    income_counts = df['income'].value_counts()
    labels = ['Thu nhập <=50K', 'Thu nhập >50K']  # Thay đổi nhãn thành dễ hiểu
    sizes = income_counts.values
    colors = ['#ff9999', '#66b3ff']  # Màu tương tự biểu đồ cột

    plt.figure(figsize=(6, 6), facecolor='#f4f0fa')  # Background màu tím nhạt
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140,
            colors=colors, explode=(0.05, 0), shadow=True)
    plt.axis('equal')

    return plot_to_base64()

# Biểu đồ cột phân phối thu nhập theo giới tính
def plot_income_by_gender(df):
    plt.figure(figsize=(6, 4), facecolor='#f4f0fa')  # Background màu tím nhạt
    sns.countplot(x='gender', hue='income', data=df, palette=colors)  # Sử dụng cùng bộ màu
    plt.title('Phân phối thu nhập theo giới tính')
    return plot_to_base64()

# Biểu đồ cột thu nhập theo giờ làm việc
def plot_income_by_hours(df):
    df['hours-per-week'] = pd.cut(df['hours-per-week'],
                               bins=[0, 30, 40, 100],
                               labels=['Less', 'Normal', 'Extra'])
    plt.figure(figsize=(6, 4), facecolor='#f4f0fa')  # Background màu tím nhạt
    sns.countplot(x='hours-per-week', hue='income', data=df, palette=colors)  # Sử dụng cùng bộ màu
    plt.title('Thu nhập theo giờ làm việc')
    return plot_to_base64()

# Biểu đồ cột thu nhập theo nhóm tuổi
def plot_income_by_age(df):
    df['age'] = pd.cut(df['age'], bins=[0, 25, 50, 100],
                             labels=['Young', 'Adult', 'Old'])
    plt.figure(figsize=(6, 4), facecolor='#f4f0fa')  # Background màu tím nhạt
    sns.countplot(x='age', hue='income', data=df, palette=colors)  # Sử dụng cùng bộ màu
    plt.title('Thu nhập theo nhóm tuổi')
    return plot_to_base64()

# Biểu đồ cột thu nhập theo quốc gia
def plot_income_by_country(df):
    df['native-country'] = df['native-country'].apply(lambda x: x if x == 'United-States' else 'Other')
    plt.figure(figsize=(6, 4), facecolor='#f4f0fa')  # Background màu tím nhạt
    sns.countplot(x='native-country', hue='income', data=df, palette=colors)  # Sử dụng cùng bộ màu
    plt.title('Thu nhập theo quốc gia')
    return plot_to_base64()