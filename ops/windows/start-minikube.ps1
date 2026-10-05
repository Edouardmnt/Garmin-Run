# Démarre Docker Desktop puis minikube automatiquement à l'ouverture de session Windows.
# Lancé par la tâche planifiée "Garmin-Run minikube" (voir install-autostart.ps1).
# Journal : %LOCALAPPDATA%\garmin-run\start-minikube.log

$log = Join-Path $env:LOCALAPPDATA "garmin-run\start-minikube.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
Start-Transcript -Path $log -Append | Out-Null
Write-Output "=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') Démarrage"

function Test-Docker {
    docker info *> $null
    return ($LASTEXITCODE -eq 0)
}

# 1. Lancer Docker Desktop s'il ne tourne pas
if (-not (Test-Docker)) {
    Write-Output "Docker n'est pas prêt : lancement de Docker Desktop"
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
}

# 2. Attendre que Docker soit prêt (5 minutes maximum)
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    if (Test-Docker) { $ready = $true; break }
    Start-Sleep -Seconds 5
}
if (-not $ready) {
    Write-Output "Docker toujours indisponible après 5 minutes : abandon"
    Stop-Transcript | Out-Null
    exit 1
}
Write-Output "Docker est prêt"

# 3. Démarrer minikube s'il n'est pas déjà démarré
minikube status *> $null
if ($LASTEXITCODE -eq 0) {
    Write-Output "minikube tourne déjà"
} else {
    Write-Output "Démarrage de minikube"
    minikube start
}

# 4. Ouvrir l'accès à Foulée sur http://localhost:8501, une seule fois, en arrière-plan
$running = Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" |
    Where-Object { $_.CommandLine -like "*port-forward.ps1*" }
if ($running) {
    Write-Output "L'accès à Foulée est déjà ouvert"
} else {
    Write-Output "Attente du tableau de bord dans le cluster"
    kubectl -n garmin-run rollout status deployment/garmin-dashboard --timeout=300s
    $forward = Join-Path $PSScriptRoot "port-forward.ps1"
    Start-Process powershell.exe -WindowStyle Hidden `
        -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$forward`""
    Write-Output "Accès à Foulée ouvert : http://localhost:8501"
}

Write-Output "=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') Terminé"
Stop-Transcript | Out-Null
