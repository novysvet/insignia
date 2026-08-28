@echo off
rem Re-attach the stripe vhdx and mount it in the Arch guest after a WSL recycle.
rem Run from Windows:  build\remount-stripe.bat
wsl --mount "E:\stripe\stripe.vhdx" --vhd --bare --name insigstripe
wsl -d Arch -e bash -lc "bash /mnt/e/coding/Insignia/build/mount-stripe.sh 100"
