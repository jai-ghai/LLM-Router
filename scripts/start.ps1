# Starts the Gradist API (port 8000) and the UI (port 5174) in two windows.
# Reads KEY=VALUE lines from .env into the API's environment; main.py itself reads os.environ only.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $root ".env"
if (-not (Test-Path $envFile)) { throw "Missing $envFile (needs at least GRADIST_TIERS and GRADIST_SESSION_SECRET)." }

$assignments = Get-Content $envFile | Where-Object { $_ -match '^\s*[A-Za-z_][A-Za-z0-9_]*\s*=' } | ForEach-Object {
    $key, $value = $_ -split '=', 2
    $value = $value.Trim().Trim('"').Trim("'").Replace("'", "''")
    "`$env:$($key.Trim()) = '$value'"
}
$apiCommand = ($assignments + "Set-Location '$root'", ".\.venv\Scripts\python.exe -m uvicorn gradist.service.main:create --factory --port 8000") -join "; "

Start-Process powershell -ArgumentList "-NoExit", "-Command", $apiCommand
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root\ui'; npm run dev"
Write-Host "API: http://localhost:8000/docs   UI: http://localhost:5174"
