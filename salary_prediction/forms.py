from django import forms

class SalaryPredictionForm(forms.Form):
    # Tuổi
    age = forms.IntegerField(
        label='Tuổi',
        min_value=17,
        max_value=90,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Nhập tuổi của bạn'
        })
    )
    
    # Lớp công việc
    WORKCLASS_CHOICES = [
        ('Private', 'Tư nhân'),
        ('Self-emp-not-inc', 'Tự kinh doanh (không hợp nhất)'),
        ('Self-emp-inc', 'Tự kinh doanh (hợp nhất)'),
        ('Federal-gov', 'Chính phủ liên bang'),
        ('Local-gov', 'Chính quyền địa phương'),
        ('State-gov', 'Chính phủ tiểu bang'),
        ('Without-pay', 'Không lương'),
        ('Never-worked', 'Chưa từng làm việc'),
    ]
    workclass = forms.ChoiceField(
        label='Lớp công việc',
        choices=WORKCLASS_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    # Trình độ học vấn
    EDUCATION_CHOICES = [
        ('Bachelors', 'Cử nhân'),
        ('Some-college', 'Cao đẳng'),
        ('11th', 'Lớp 11'),
        ('HS-grad', 'Tốt nghiệp phổ thông'),
        ('Prof-school', 'Trường nghề'),
        ('Assoc-acdm', 'Cao đẳng học thuật'),
        ('Assoc-voc', 'Cao đẳng nghề'),
        ('9th', 'Lớp 9'),
        ('7th-8th', 'Lớp 7-8'),
        ('12th', 'Lớp 12'),
        ('Masters', 'Thạc sĩ'),
        ('1st-4th', 'Lớp 1-4'),
        ('10th', 'Lớp 10'),
        ('Doctorate', 'Tiến sĩ'),
        ('5th-6th', 'Lớp 5-6'),
        ('Preschool', 'Mầm non'),
    ]
    education = forms.ChoiceField(
        label='Trình độ học vấn',
        choices=EDUCATION_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    # Tình trạng hôn nhân
    MARITAL_CHOICES = [
        ('Married-civ-spouse', 'Đã kết hôn (sống chung)'),
        ('Divorced', 'Đã ly hôn'),
        ('Never-married', 'Chưa kết hôn'),
        ('Separated', 'Ly thân'),
        ('Widowed', 'Góa'),
        ('Married-spouse-absent', 'Đã kết hôn (sống xa)'),
        ('Married-AF-spouse', 'Đã kết hôn (vợ/chồng trong quân đội)'),
    ]
    marital_status = forms.ChoiceField(
        label='Tình trạng hôn nhân',
        choices=MARITAL_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    # Nghề nghiệp
    OCCUPATION_CHOICES = [
        ('Tech-support', 'Hỗ trợ kỹ thuật'),
        ('Craft-repair', 'Thợ thủ công'),
        ('Other-service', 'Dịch vụ khác'),
        ('Sales', 'Bán hàng'),
        ('Exec-managerial', 'Quản lý điều hành'),
        ('Prof-specialty', 'Chuyên môn nghiệp vụ'),
        ('Handlers-cleaners', 'Xử lý - vệ sinh'),
        ('Machine-op-inspct', 'Vận hành máy - kiểm tra'),
        ('Adm-clerical', 'Hành chính - văn phòng'),
        ('Farming-fishing', 'Nông nghiệp - đánh cá'),
        ('Transport-moving', 'Vận tải - chuyển hàng'),
        ('Priv-house-serv', 'Giúp việc nhà'),
        ('Protective-serv', 'Dịch vụ bảo vệ'),
        ('Armed-Forces', 'Lực lượng vũ trang'),
    ]
    occupation = forms.ChoiceField(
        label='Nghề nghiệp',
        choices=OCCUPATION_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    # Mối quan hệ
    RELATIONSHIP_CHOICES = [
        ('Wife', 'Vợ'),
        ('Own-child', 'Con ruột'),
        ('Husband', 'Chồng'),
        ('Not-in-family', 'Không trong gia đình'),
        ('Other-relative', 'Họ hàng khác'),
        ('Unmarried', 'Chưa kết hôn'),
    ]
    relationship = forms.ChoiceField(
        label='Mối quan hệ',
        choices=RELATIONSHIP_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    # Chủng tộc
    RACE_CHOICES = [
        ('White', 'Da trắng'),
        ('Asian-Pac-Islander', 'Châu Á - Thái Bình Dương'),
        ('Amer-Indian-Eskimo', 'Thổ dân Mỹ - Eskimo'),
        ('Other', 'Khác'),
        ('Black', 'Da đen'),
    ]
    race = forms.ChoiceField(
        label='Chủng tộc',
        choices=RACE_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    # Giới tính
    GENDER_CHOICES = [
        ('Female', 'Nữ'),
        ('Male', 'Nam'),
    ]
    gender = forms.ChoiceField(
        label='Giới tính',
        choices=GENDER_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )
    
    # Thu nhập vốn
    capital_gain = forms.IntegerField(
        label='Thu nhập từ vốn',
        min_value=0,
        initial=0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Nhập thu nhập từ vốn (USD)'
        })
    )
    
    # Lỗ vốn
    capital_loss = forms.IntegerField(
        label='Lỗ vốn',
        min_value=0,
        initial=0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Nhập lỗ vốn (USD)'
        })
    )
    
    # Số giờ làm việc mỗi tuần
    hours_per_week = forms.IntegerField(
        label='Số giờ làm việc mỗi tuần',
        min_value=1,
        max_value=99,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Nhập số giờ làm việc mỗi tuần'
        })
    )
    
    # Quốc gia
    COUNTRY_CHOICES = [
        ('United-States', 'Hoa Kỳ'),
        ('Cambodia', 'Campuchia'),
        ('England', 'Anh'),
        ('Puerto-Rico', 'Puerto Rico'),
        ('Canada', 'Canada'),
        ('Germany', 'Đức'),
        ('Outlying-US(Guam-USVI-etc)', 'Vùng lãnh thổ Mỹ'),
        ('India', 'Ấn Độ'),
        ('Japan', 'Nhật Bản'),
        ('Greece', 'Hy Lạp'),
        ('South', 'Nam'),
        ('China', 'Trung Quốc'),
        ('Cuba', 'Cuba'),
        ('Iran', 'Iran'),
        ('Honduras', 'Honduras'),
        ('Philippines', 'Philippines'),
        ('Italy', 'Ý'),
        ('Poland', 'Ba Lan'),
        ('Jamaica', 'Jamaica'),
        ('Vietnam', 'Việt Nam'),
        ('Mexico', 'Mexico'),
        ('Portugal', 'Bồ Đào Nha'),
        ('Ireland', 'Ireland'),
        ('France', 'Pháp'),
        ('Dominican-Republic', 'Cộng hòa Dominica'),
        ('Laos', 'Lào'),
        ('Ecuador', 'Ecuador'),
        ('Taiwan', 'Đài Loan'),
        ('Haiti', 'Haiti'),
        ('Columbia', 'Colombia'),
        ('Hungary', 'Hungary'),
        ('Guatemala', 'Guatemala'),
        ('Nicaragua', 'Nicaragua'),
        ('Scotland', 'Scotland'),
        ('Thailand', 'Thái Lan'),
        ('Yugoslavia', 'Nam Tư'),
        ('El-Salvador', 'El Salvador'),
        ('Trinadad&Tobago', 'Trinidad và Tobago'),
        ('Peru', 'Peru'),
        ('Hong', 'Hong Kong'),
        ('Holand-Netherlands', 'Hà Lan'),
    ]
    native_country = forms.ChoiceField(
        label='Quốc gia',
        choices=COUNTRY_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )