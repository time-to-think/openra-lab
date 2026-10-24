@echo off
rem Apply the execution policy only to this child PowerShell process.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
exit /b %errorlevel%
