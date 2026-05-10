@echo off
SET KEY="C:\Users\zpwkg\Documents\WasherCRM\AWS_accesskey\WhiteOn-Key.pem"
SET USER_IP=ubuntu@13.124.100.75

echo [AWS] V35 Trading Bot STOPPING...
ssh -i %KEY% -o StrictHostKeyChecking=no %USER_IP% "pkill -f v35_live.py"
echo.
echo Bot stopped.
pause
