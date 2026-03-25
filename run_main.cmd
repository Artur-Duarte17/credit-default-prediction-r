@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0run_main.ps1" %*
exit /b %errorlevel%
