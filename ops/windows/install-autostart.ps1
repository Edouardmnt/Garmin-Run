# Installe (ou réinstalle) la tâche planifiée qui démarre minikube à l'ouverture de session.
# Usage, depuis la racine du dépôt : powershell -ExecutionPolicy Bypass -File ops\windows\install-autostart.ps1

$taskName = "Garmin-Run minikube"
$script = Join-Path $PSScriptRoot "start-minikube.ps1"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`""

# Déclenchement à l'ouverture de session, avec 1 minute de délai pour laisser Windows démarrer
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$trigger.Delay = "PT1M"

# Ne pas s'arrêter sur batterie, relancer si l'ordinateur sort de veille trop tard
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Démarre Docker Desktop et minikube pour la synchronisation Garmin-Run" -Force | Out-Null

Write-Output "Tâche '$taskName' installée. Script lancé : $script"
