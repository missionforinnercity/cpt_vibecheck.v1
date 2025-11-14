# Sentiment Agent Deployment Guide

## Setup Instructions for Windows Task Scheduler

### Option 1: Manual Setup (Recommended)

1. **Open Task Scheduler** (search "Task Scheduler" in Windows)

2. **Create a new task:**
   - Right-click "Task Scheduler Library" → "Create Basic Task"
   - Name: `Sentiment Agent Daily`
   - Description: `Monitor Cape Town CBD sentiment from reviews`

3. **Set trigger:**
   - Click "Trigger" tab → New
   - Choose "Daily"
   - Set time: 9:00 AM (or your preferred time)
   - Click OK

4. **Set action:**
   - Click "Action" tab → New
   - Program/script: `c:\Users\STEFA\Sentiment_mfic\run_sentiment_agent.bat`
   - Start in (Optional): `c:\Users\STEFA\Sentiment_mfic`
   - Click OK

5. **Configure settings:**
   - "General" tab: Check "Run whether user is logged in or not"
   - "Conditions" tab: Uncheck "Start the task only if the computer is on AC power"
   - Click OK

---

### Option 2: PowerShell Script Setup

Run this PowerShell command as Administrator:

```powershell
$taskName = "Sentiment Agent Daily"
$scriptPath = "c:\Users\STEFA\Sentiment_mfic\run_sentiment_agent.bat"
$trigger = New-ScheduledTaskTrigger -Daily -At 9:00AM
$action = New-ScheduledTaskAction -Execute $scriptPath
Register-ScheduledTask -TaskName $taskName -Trigger $trigger -Action $action -RunLevel Highest
```

---

## Data Sources Configuration

### Currently Configured:
- **Local CSV Reviews**: `sample_reviews.csv`
- **Output**: `data/cape_town_sentiment.csv`

### To Enable Twitter Scraping:

1. Find a newer snscrape fork compatible with Python 3.12, or
2. Downgrade to Python 3.11:
   ```
   - Uninstall current Python 3.12
   - Install Python 3.11 from python.org
   - Reinstall snscrape
   ```

3. Then modify the batch file to include Twitter queries:
```batch
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "run_sentiment_agent.ps1" -TwitterEnabled $true
```

---

## Monitoring

- **Logs**: Check `logs/sentiment_agent.log` for execution details
- **Output**: `data/cape_town_sentiment.csv` contains all analyzed records
- **Check Task History**: Task Scheduler → Select task → History tab

---

## Troubleshooting

### Task doesn't run:
- Verify the full paths are correct in the batch file
- Check that Python is in the PATH or use full executable path
- Review Task Scheduler → Event Viewer for error details

### Module not found errors:
- Ensure nltk is installed: `pip install nltk`
- For Twitter scraping: `pip install snscrape` (requires Python 3.11 or compatible fork)

### Permission denied:
- Run Task Scheduler as Administrator
- Check file permissions on the script directory

---

## Test Run

To verify everything works before scheduling:

```powershell
cd c:\Users\STEFA\Sentiment_mfic
.\run_sentiment_agent.bat
```

Check `logs/sentiment_agent.log` for output.
