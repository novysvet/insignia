$ErrorActionPreference = "Stop"
$python = "E:\coding\python-envs\insignia-win\Scripts\python.exe"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$log = Join-Path $root "server.log"

if (-not (Test-Path $python)) { throw "Missing E:-backed Python environment: $python" }
Start-Process -FilePath $python -ArgumentList @((Join-Path $root "server.py"), "--host", "127.0.0.1", "--port", "8000") -WorkingDirectory $root -RedirectStandardOutput $log -RedirectStandardError (Join-Path $root "server-error.log") -WindowStyle Hidden
Write-Host "Insignia UI started at http://127.0.0.1:8000"
