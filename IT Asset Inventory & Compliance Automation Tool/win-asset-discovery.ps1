# scripts/discovery/win-asset-discovery.ps1
$computer = Get-ComputerInfo
$software = Get-ItemProperty HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*

$assetData = @{
    hostname = $computer.CSName
    os_version = $computer.WindowsVersion
    serial_number = (Get-CimInstance Win32_BIOS).SerialNumber
    installed_software = $software.DisplayName
}

Invoke-RestMethod -Uri https://api/asset-discovery -Method POST -Body ($assetData | ConvertTo-Json)