@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo Changchun High-Tech Streamlit Dashboard
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found. Please install Python or add it to PATH.
    pause
    exit /b 1
)

echo [1/3] Checking project dependencies...
python -c "import streamlit, pandas, akshare, openpyxl, sklearn" >nul 2>&1
if errorlevel 1 (
    echo [2/3] Installing dependencies from requirements.txt...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
) else (
    echo [2/3] Dependencies already available.
)

echo [3/3] Starting Streamlit...
echo.
echo Open this URL if the browser does not open automatically:
echo http://localhost:8501
echo.
python -m streamlit run app.py --server.address localhost --server.port 8501

echo.
echo Streamlit has stopped.
pause
