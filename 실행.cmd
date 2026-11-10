@echo off
cd /d "%~dp0"
"..\.venv-battery\Scripts\python.exe" app.py
if errorlevel 1 pause
