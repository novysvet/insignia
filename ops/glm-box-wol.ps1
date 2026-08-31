#Requires -Version 5.1
<#
Run locally from an elevated PowerShell:
  glm-box:  .\ops\glm-box-wol.ps1 -Mode ConfigureTarget
  dev PC:   .\ops\glm-box-wol.ps1 -Mode ConfigureRelay -Mac AA-BB-CC-DD-EE-FF
Wake later, locally or over SSH/Tailscale:
  .\ops\glm-box-wol.ps1 -Mode Wake
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet("ConfigureTarget", "ConfigureRelay", "Wake", "SelfTest")]
    [string]$Mode,
    [string]$Mac,
    [string]$Broadcast = "255.255.255.255",
    [string]$TargetUser = "PC"
)

$ErrorActionPreference = "Stop"
$configDirectory = Join-Path $env:ProgramData "Insignia"
$configPath = Join-Path $configDirectory "glm-box-wol.json"

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Require-Administrator {
    if (-not (Test-Administrator)) {
        throw "Run this action from PowerShell opened with 'Run as administrator'."
    }
}

function Normalize-Mac([string]$Value) {
    $hex = $Value -replace "[^0-9A-Fa-f]", ""
    if ($hex.Length -ne 12) { throw "Invalid MAC address: $Value" }
    if ($hex -notmatch "^[0-9A-Fa-f]{12}$") { throw "Invalid MAC address: $Value" }
    $hex.ToUpperInvariant()
}

function Test-IPv4([string]$Value) {
    $address = $null
    [Net.IPAddress]::TryParse($Value, [ref]$address) -and
        $address.AddressFamily -eq [Net.Sockets.AddressFamily]::InterNetwork
}

function New-WakePacket([string]$TargetMac) {
    $hex = Normalize-Mac $TargetMac
    [byte[]]$address = 0..5 | ForEach-Object { [Convert]::ToByte($hex.Substring($_ * 2, 2), 16) }
    [byte[]]((,[byte]0xFF * 6) + ($address * 16))
}

function Send-WakePacket([string]$TargetMac, [string]$TargetBroadcast) {
    [byte[]]$packet = New-WakePacket $TargetMac
    $udp = [Net.Sockets.UdpClient]::new()
    try {
        $udp.EnableBroadcast = $true
        1..3 | ForEach-Object {
            [void]$udp.Send($packet, $packet.Length, $TargetBroadcast, 9)
            Start-Sleep -Milliseconds 80
        }
    } finally {
        $udp.Dispose()
    }
}

function Enable-TailscaleUnattended {
    $tailscale = Join-Path $env:ProgramFiles "Tailscale\tailscale.exe"
    if (-not (Test-Path $tailscale)) { throw "Tailscale is not installed." }
    & $tailscale up --unattended=true
    if ($LASTEXITCODE) { throw "tailscale up failed with status $LASTEXITCODE" }
}

function Merge-AdministratorSshKeys([string]$UserName) {
    Get-LocalUser -Name $UserName | Out-Null
    $userKeys = "C:\Users\$UserName\.ssh\authorized_keys"
    $adminKeys = Join-Path $env:ProgramData "ssh\administrators_authorized_keys"
    $keys = @()
    if (Test-Path $adminKeys) { $keys += Get-Content $adminKeys }
    if (Test-Path $userKeys) { $keys += Get-Content $userKeys }
    $keys = @($keys | Where-Object { $_.Trim() } | Sort-Object -Unique)
    if (-not $keys.Count) {
        throw "No SSH public key exists in $userKeys or $adminKeys; refusing to risk lockout."
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $adminKeys) | Out-Null
    $keys | Set-Content -Encoding ascii $adminKeys
    & icacls.exe $adminKeys /inheritance:r /grant "*S-1-5-32-544:F" /grant "SYSTEM:F" | Out-Null
    if ($LASTEXITCODE) { throw "icacls failed with status $LASTEXITCODE" }

    $administrators = Get-LocalGroup -SID "S-1-5-32-544"
    $account = "$env:COMPUTERNAME\$UserName"
    if (-not (Get-LocalGroupMember -Group $administrators | Where-Object Name -EQ $account)) {
        Add-LocalGroupMember -Group $administrators -Member $account
    }
    Set-Service sshd -StartupType Automatic
    Restart-Service sshd
}

switch ($Mode) {
    "ConfigureTarget" {
        Require-Administrator
        Write-Host "ASUS UEFI must be configured manually:" -ForegroundColor Yellow
        Write-Host "  Delete during boot -> F7 Advanced Mode -> Advanced -> APM Configuration"
        Write-Host "  ErP Ready = Disabled"
        Write-Host "  Power On By PCI-E = Enabled"
        Write-Host "  F10 -> Save and reset"
        if ((Read-Host "Type YES after those settings are saved") -ne "YES") { exit 1 }

        & powercfg.exe /hibernate on
        Set-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Power" `
            -Name HiberbootEnabled -Type DWord -Value 0

        $adapters = @(Get-NetAdapter -Physical | Where-Object HardwareInterface)
        if (-not $adapters.Count) { throw "No physical network adapter found." }
        $wired = @($adapters | Where-Object { $_.MediaType -eq "802.3" -or $_.PhysicalMediaType -eq "802.3" })
        if ($wired.Count -eq 1) {
            $adapter = $wired[0]
        } else {
            $adapters | Format-Table ifIndex, Name, InterfaceDescription, Status, MacAddress
            $index = [int](Read-Host "Enter the ifIndex of the wired Ethernet adapter")
            $adapter = $adapters | Where-Object ifIndex -EQ $index
            if (-not $adapter) { throw "No adapter has ifIndex $index." }
        }

        try {
            Set-NetAdapterPowerManagement -Name $adapter.Name -WakeOnMagicPacket Enabled -WakeOnPattern Enabled
        } catch {
            Write-Warning "Enable Magic Packet wake manually in Device Manager for '$($adapter.Name)'."
        }
        Get-NetAdapterAdvancedProperty -Name $adapter.Name -ErrorAction SilentlyContinue |
            Where-Object DisplayName -Match "Wake.*Magic|Shutdown Wake|Enable PME" |
            ForEach-Object {
                try {
                    Set-NetAdapterAdvancedProperty -Name $adapter.Name `
                        -DisplayName $_.DisplayName -DisplayValue "Enabled" -ErrorAction Stop
                } catch {
                    Write-Warning "Set '$($_.DisplayName)' to Enabled manually in Device Manager."
                }
            }

        Enable-TailscaleUnattended
        Merge-AdministratorSshKeys $TargetUser
        Write-Host "glm-box configured. Wired MAC: $($adapter.MacAddress)" -ForegroundColor Green
        Write-Host "Keep this console open until a fresh 'ssh glm-box whoami' connection succeeds."
    }

    "ConfigureRelay" {
        Require-Administrator
        if (-not $Mac) { $Mac = Read-Host "glm-box wired Ethernet MAC address" }
        $Mac = Normalize-Mac $Mac
        if (-not (Test-IPv4 $Broadcast)) { throw "Invalid IPv4 broadcast address: $Broadcast" }
        New-Item -ItemType Directory -Force -Path $configDirectory | Out-Null
        @{ Mac = $Mac; Broadcast = $Broadcast } |
            ConvertTo-Json | Set-Content -Encoding ascii $configPath
        Enable-TailscaleUnattended

        $capability = Get-WindowsCapability -Online -Name "OpenSSH.Server~~~~0.0.1.0"
        if ($capability.State -ne "Installed") {
            Add-WindowsCapability -Online -Name "OpenSSH.Server~~~~0.0.1.0" | Out-Null
        }
        Set-Service sshd -StartupType Automatic
        Start-Service sshd
        $firewall = Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue
        if ($firewall) {
            $firewall | Get-NetFirewallAddressFilter |
                Set-NetFirewallAddressFilter -RemoteAddress "100.64.0.0/10", "fd7a:115c:a1e0::/48" | Out-Null
        }
        Write-Host "Relay configured. Test now with:" -ForegroundColor Green
        Write-Host "  powershell -File $PSCommandPath -Mode Wake"
        Write-Host "From another tailnet computer, run that command over SSH on this relay."
    }

    "Wake" {
        if ((-not $Mac) -and (Test-Path $configPath)) {
            $config = Get-Content $configPath -Raw | ConvertFrom-Json
            $Mac = $config.Mac
            $Broadcast = $config.Broadcast
        }
        if (-not $Mac) { throw "No MAC configured. Run -Mode ConfigureRelay first or pass -Mac." }
        if (-not (Test-IPv4 $Broadcast)) { throw "Invalid IPv4 broadcast address: $Broadcast" }
        Send-WakePacket $Mac $Broadcast
        Write-Host "Sent three Wake-on-LAN packets to $Mac via ${Broadcast}:9." -ForegroundColor Green
    }

    "SelfTest" {
        $packet = New-WakePacket "01:23:45:67:89:ab"
        if ($packet.Length -ne 102 -or ($packet[0..5] | Where-Object { $_ -ne 255 })) {
            throw "Wake packet self-test failed."
        }
        if (-not (Test-IPv4 "255.255.255.255") -or (Test-IPv4 "999.1.1.1")) {
            throw "IPv4 validation self-test failed."
        }
        Write-Host "glm-box-wol.ps1 self-test passed."
    }
}
