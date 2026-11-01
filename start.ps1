# Sobe backend (FastAPI :8000) e frontend (Vite :5173) em janelas separadas.
# Uso:  powershell -ExecutionPolicy Bypass -File .\start.ps1
$root = $PSScriptRoot

if (-not (Test-Path "$root\backend\.venv")) {
    Write-Host "Criando ambiente Python (backend/.venv)..."
    python -m venv --system-site-packages "$root\backend\.venv"
    & "$root\backend\.venv\Scripts\python.exe" -m pip install -r "$root\backend\requirements.txt"
}
if (-not (Test-Path "$root\frontend\node_modules")) {
    Write-Host "Instalando dependências do frontend..."
    Push-Location "$root\frontend"; npm install; Pop-Location
}

$busy = foreach ($port in 8000, 5173) {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1 | ForEach-Object {
            $p = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
            "  porta $port em uso por PID $($_.OwningProcess) ($($p.ProcessName))"
        }
}
if ($busy) {
    Write-Host "Ja existe algo rodando nas portas do laboratorio:" -ForegroundColor Yellow
    $busy | ForEach-Object { Write-Host $_ -ForegroundColor Yellow }
    Write-Host "Feche as janelas antigas do backend/frontend (ou encerre esses processos) e rode de novo." -ForegroundColor Yellow
    Write-Host "Se o laboratorio ja estiver no ar, basta abrir http://localhost:5173"
    exit 1
}

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\backend'; .\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm run dev -- --port 5173 --strictPort"
Start-Sleep -Seconds 4
Start-Process "http://localhost:5173"
