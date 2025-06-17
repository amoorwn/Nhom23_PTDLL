@echo off
echo =====================================
echo DU AN DU DOAN MUC LUONG
echo =====================================
echo.

REM Kiem tra Python
echo [1/6] Kiem tra Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo LOI: Python chua duoc cai dat!
    echo Vui long cai dat Python tu https://www.python.org/
    pause
    exit /b 1
)
echo Python da san sang!
echo.

REM Tao moi truong ao
echo [2/6] Tao moi truong ao...
if not exist "venv" (
    python -m venv venv
    echo Moi truong ao da duoc tao!
) else (
    echo Moi truong ao da ton tai!
)
echo.

REM Kich hoat moi truong ao
echo [3/6] Kich hoat moi truong ao...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo LOI: Khong the kich hoat moi truong ao!
    echo Thu chay truc tiep...
    goto :install_global
)
echo Moi truong ao da kich hoat!
echo.

REM Cai dat thu vien trong venv
echo [4/6] Cai dat cac thu vien can thiet...
venv\Scripts\python.exe -m pip install --upgrade pip
echo Cai dat Django truoc...
venv\Scripts\python.exe -m pip install Django
if %errorlevel% neq 0 (
    echo LOI: Khong the cai dat Django trong venv!
    echo Thu cai dat global...
    goto :install_global
)
echo Cai dat cac thu vien con lai...
venv\Scripts\python.exe -m pip install pandas scikit-learn joblib
echo Cac thu vien da duoc cai dat!
set USE_VENV=1
goto :continue

:install_global
echo Cai dat global...
python -m pip install Django pandas scikit-learn joblib
if %errorlevel% neq 0 (
    echo LOI: Khong the cai dat thu vien!
    pause
    exit /b 1
)
echo Cac thu vien da duoc cai dat!
set USE_VENV=0

:continue
echo.

REM Chay migration
echo [5/6] Khoi tao co so du lieu...
if "%USE_VENV%"=="1" (
    venv\Scripts\python.exe manage.py makemigrations
    venv\Scripts\python.exe manage.py migrate
) else (
    python manage.py makemigrations
    python manage.py migrate
)
echo Co so du lieu da san sang!
echo.

REM Tao superuser (tuy chon)
echo Ban co muon tao tai khoan admin? (y/n)
set /p create_admin=
if /i "%create_admin%"=="y" (
    if "%USE_VENV%"=="1" (
        venv\Scripts\python.exe manage.py createsuperuser
    ) else (
        python manage.py createsuperuser
    )
)
echo.

REM Khoi dong server
echo [6/6] Khoi dong server...
echo.
echo =====================================
echo Server dang chay tai: http://127.0.0.1:8000
echo Nhan Ctrl+C de dung server
echo =====================================
echo.
if "%USE_VENV%"=="1" (
    venv\Scripts\python.exe manage.py runserver
) else (
    python manage.py runserver
)

pause