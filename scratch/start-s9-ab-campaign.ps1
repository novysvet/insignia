$ErrorActionPreference = "Stop"

Start-Process -FilePath "$env:SystemRoot\System32\wsl.exe" `
    -ArgumentList @("-d", "Arch", "--", "bash", "/var/lib/insignia/s9-ab-campaign.sh") `
    -WindowStyle Hidden
