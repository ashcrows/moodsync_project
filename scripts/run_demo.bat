@echo off
REM MoodSync demo — Windows.
cd /d "%~dp0\.."

set OS_ARG=%1
if "%OS_ARG%"=="" set OS_ARG=auto

echo ==> Creating virtual environment (.venv)
python -m venv .venv
call .venv\Scripts\activate.bat

echo ==> Installing light requirements
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ==> Running MoodSync demo (target OS: %OS_ARG%)
python -m moodsync.cli demo --os %OS_ARG%

echo.
echo Done. Try the UI:   python -m moodsync.cli serve-app --os %OS_ARG%
echo Or the API:         python -m moodsync.cli serve-api --os %OS_ARG%
