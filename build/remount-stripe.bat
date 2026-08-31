@echo off
setlocal EnableExtensions EnableDelayedExpansion
rem Attach the E:-hosted stripe VHDX and let the Arch guest verify LABEL/UUID.
set "STRIPE_VHD=E:\stripe\stripe.vhdx"
set "STRIPE_UUID=%~1"
if not exist "%STRIPE_VHD%" (
  echo Missing stripe VHDX: %STRIPE_VHD% 1>&2
  exit /b 1
)

rem Query this exact E: image instead of trusting some other attached disk
rem that happens to carry LABEL=stripe.
set "ATTACHED=False"
for /f "usebackq delims=" %%A in (`powershell.exe -NoProfile -Command "(Get-DiskImage -ImagePath '%STRIPE_VHD%').Attached"`) do set "ATTACHED=%%A"
if /I not "!ATTACHED!"=="True" (
  wsl --mount "%STRIPE_VHD%" --vhd --bare --name insigstripe
  rem WSL has returned success-looking output for failed attaches before.  The
  rem authoritative check is whether Windows now reports this exact image as
  rem attached, so query it again regardless of wsl.exe's exit code.
  set "ATTACHED=False"
  for /f "usebackq delims=" %%A in (`powershell.exe -NoProfile -Command "(Get-DiskImage -ImagePath '%STRIPE_VHD%').Attached"`) do set "ATTACHED=%%A"
  if /I not "!ATTACHED!"=="True" (
    echo Failed to attach stripe VHDX: %STRIPE_VHD%. Run this script from an elevated terminal. 1>&2
    exit /b 1
  )
)

if defined STRIPE_UUID (
  wsl -d Arch -e bash /mnt/e/coding/Insignia/build/mount-stripe.sh "%STRIPE_UUID%"
) else (
  wsl -d Arch -e bash /mnt/e/coding/Insignia/build/mount-stripe.sh
)
if errorlevel 1 exit /b 1
exit /b 0
