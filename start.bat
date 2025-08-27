@echo off
echo.
echo ================================
echo  OSME - Odor Search Molecule Engine
echo ================================
echo.
echo Starting protected application...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ and try again
    pause
    exit /b 1
)

REM Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip is not available
    echo Please ensure pip is installed
    pause
    exit /b 1
)

REM Install dependencies if requirements.txt exists
if exist requirements.txt (
    echo Installing dependencies...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
    echo.
)

REM Check if data files exist
if not exist "data\embeddings.npy" (
    echo WARNING: embeddings.npy not found in data folder
    echo Please ensure all data files are present
)

if not exist "data\molecules.csv" (
    echo WARNING: molecules.csv not found in data folder
    echo Please ensure all data files are present
)

echo Starting OSME application...
echo.
echo Available login credentials:
echo - admin / admin
echo - user / user  
echo - osme / osme
echo.
echo The application will open in your browser automatically.
echo Press Ctrl+C to stop the application.
echo.

REM Start Streamlit app
streamlit run streamlit_app.py --server.headless false --server.runOnSave true --browser.gatherUsageStats false

echo.
echo Application stopped.
pause