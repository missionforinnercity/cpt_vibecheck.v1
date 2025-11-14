@echo off
REM Batch file to run sentiment agent via Task Scheduler
REM Ensures the PowerShell script executes with proper permissions

cd /d c:\Users\STEFA\Sentiment_mfic
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "c:\Users\STEFA\Sentiment_mfic\run_sentiment_agent.ps1"
