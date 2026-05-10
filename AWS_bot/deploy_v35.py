import subprocess
import os
import sys

# AWS Configuration
SERVER_IP = '13.124.100.75'
KEY_PATH = r'C:\Users\zpwkg\Documents\WasherCRM\AWS_accesskey\WhiteOn-Key.pem'
USERNAME = 'ubuntu'
REMOTE_DIR = '/home/ubuntu/trading_bot'

# Files to sync
FILES_TO_SYNC = [
    'v35_live.py',
    'v30_train.py',
    'config.py',
    'exchange.py',
    'telegram_bot.py',
    'v29_env.py',
    'requirements.txt',
    'start_v35_live.sh'
]

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(result.stdout)
    return result

def deploy():
    print("Starting V35 Deployment to AWS...")
    
    # 1. Upload files
    for file in FILES_TO_SYNC:
        if os.path.exists(file):
            print(f"Uploading {file}...")
            scp_cmd = f'scp -i "{KEY_PATH}" -o StrictHostKeyChecking=no "{file}" {USERNAME}@{SERVER_IP}:{REMOTE_DIR}/{file}'
            run_cmd(scp_cmd)
        else:
            print(f"Warning: {file} not found locally.")

    # 2. Set permissions and restart
    print("Restarting Bot on AWS...")
    ssh_cmd = f'ssh -i "{KEY_PATH}" -o StrictHostKeyChecking=no {USERNAME}@{SERVER_IP} "chmod +x {REMOTE_DIR}/start_v35_live.sh && {REMOTE_DIR}/start_v35_live.sh"'
    run_cmd(ssh_cmd)
    
    print("\nDeployment Complete!")
    print(f"Monitor logs: ssh -i \"{KEY_PATH}\" {USERNAME}@{SERVER_IP} \"tail -f {REMOTE_DIR}/bot_live.log\"")

if __name__ == "__main__":
    deploy()
