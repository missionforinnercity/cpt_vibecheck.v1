# PowerShell script to run sentiment agent with proper error handling and logging

$pythonExe = "C:\Users\STEFA\AppData\Local\Programs\Python\Python312\python.exe"
$scriptPath = "c:\Users\STEFA\Sentiment_mfic\sentiment_agent.py"
$logPath = "c:\Users\STEFA\Sentiment_mfic\logs\sentiment_agent.log"
$logDir = Split-Path -Path $logPath
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# Create logs directory if it doesn't exist
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

# If a local virtual environment exists at .venv, prefer its python executable
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonExe = $venvPython
    Write-Output "Using virtual environment python: $pythonExe" | Tee-Object -FilePath $logPath -Append
}

# Run the sentiment agent with sample reviews
$args = @(
    $scriptPath,
    "--local-review-csv",
    "sample_reviews.csv",
    "--output",
    "data\cape_town_sentiment.csv",
    "--log-level",
    "INFO"
)

# Execute and capture output
try {
    Write-Output "[$timestamp] Starting sentiment agent..." | Tee-Object -FilePath $logPath -Append
    & $pythonExe @args 2>&1 | Tee-Object -FilePath $logPath -Append
    Write-Output "[$timestamp] Sentiment agent completed successfully" | Tee-Object -FilePath $logPath -Append
}
catch {
    Write-Output "[$timestamp] ERROR: $($_.Exception.Message)" | Tee-Object -FilePath $logPath -Append
    exit 1
}
