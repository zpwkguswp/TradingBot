#!/bin/bash
# V35 Trading Bot Startup Script for AWS (Linux)

BOT_DIR="/home/ubuntu/trading_bot"
LOG_FILE="$BOT_DIR/bot_live.log"

echo "-------------------------------------------------------"
echo "  V35 Long/Short Sniper Live Engine Starting (AWS)"
echo "  Time: $(date)"
echo "-------------------------------------------------------"

cd $BOT_DIR

# 1. Ensure no other instance is running
pkill -f v35_live.py

# 2. Activate virtual environment and run
# Using nohup to keep it running after logout
nohup ./venv/bin/python v35_live.py >> $LOG_FILE 2>&1 &

echo "Bot started in background."
echo "Check logs with: tail -f $LOG_FILE"
