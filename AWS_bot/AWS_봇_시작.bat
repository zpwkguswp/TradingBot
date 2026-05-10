@echo off
SET KEY="C:\Users\zpwkg\Documents\WasherCRM\AWS_accesskey\WhiteOn-Key.pem"
SET USER_IP=ubuntu@13.124.100.75

echo [AWS] V35 Trading Bot STARTING...
ssh -i %KEY% -o StrictHostKeyChecking=no %USER_IP% "/home/ubuntu/trading_bot/start_v35_live.sh"
echo.
echo Done.
pause
