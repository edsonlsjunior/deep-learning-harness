# Windows wrapper for the shared cross-platform launcher.
Set-Location $PSScriptRoot
if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 "$PSScriptRoot/start.py" @args
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python "$PSScriptRoot/start.py" @args
} else {
    Write-Error "Instale Python 3.11+ e Node.js LTS e abra novamente o terminal."
    exit 1
}
exit $LASTEXITCODE
