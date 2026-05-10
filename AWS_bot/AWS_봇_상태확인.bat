@echo off
SET KEY="C:\Users\zpwkg\Documents\WasherCRM\AWS_accesskey\WhiteOn-Key.pem"
SET USER_IP=ubuntu@13.124.100.75

echo [AWS] Checking Bot Process Status...
ssh -i %KEY% -o StrictHostKeyChecking=no %USER_IP% "ps aux | grep v35_live.py | grep -v grep"
echo.
echo [AWS] Recent 10 lines of Logs:
ssh -i %KEY% -o StrictHostKeyChecking=no %USER_IP% "tail -n 10 /home/ubuntu/trading_bot/bot_live.log"
echo.
pause
