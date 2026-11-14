@echo off
title ProBad Dataset Reviewer

cd /d "%~dp0"

echo ========================================
echo       ProBad Dataset Reviewer
echo ========================================
echo.

streamlit run app/review_app.py

echo.
echo ProBad has stopped.
pause