\xef\xbb\xbf# Rend Foulée accessible sur http://localhost:8501 en permanence.
# kubectl port-forward s'arrête quand le Pod est remplacé (redéploiement du matin) : la boucle le relance.
# Lancé en arrière-plan par start-minikube.ps1. Journal : %LOCALAPPDATA%\garmin-run\port-forward.log

$log = Join-Path $env:LOCALAPPDATA "garmin-run\port-forward.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

while ($true) {
    Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') Ouverture de l'accès à Foulée (localhost:8501)"
    kubectl -n garmin-run port-forward svc/garmin-dashboard 8501:80 *>> $log
    Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') Accès coupé, nouvelle tentative dans 10 s"
    Start-Sleep -Seconds 10
}
