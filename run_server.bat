@echo off
echo =====================================
echo DU AN DU DOAN MUC LUONG
echo =====================================
echo.

REM Kiem tra moi truong ao
if not exist "venv" (
    echo LOI: Moi truong ao chua duoc tao!
    echo Vui long chay setup_and_run.bat truoc!
    pause
    exit /b 1
)

REM Kich hoat moi truong ao
echo Kich hoat moi truong ao...
call venv\Scripts\activate.bat
echo.

REM Khoi dong server
echo Khoi dong server...
echo.
echo =====================================
echo Server dang chay tai: http://127.0.0.1:8000
echo Nhan Ctrl+C de dung server
echo =====================================
echo.
python manage.py runserver

pause