import json
import pickle
import os
from datetime import datetime

LOG_FILE = "v29_experience_dataset.pkl"
STATE_FILE = "v29_live_state.json"
DASHBOARD_FILE = "v29_dashboard.html"

def generate_dashboard():
    # 1. Load Data
    active_positions = {}
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            active_positions = json.load(f).get("positions", {})
    
    past_logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            past_logs = pickle.load(f)
    
    # 2. Calculate Stats
    total_pnl = sum(log.get('performance', {}).get('pnl_pct', 0.0) for log in past_logs)
    win_count = sum(1 for log in past_logs if log.get('performance', {}).get('pnl_pct', 0.0) > 0)
    win_rate = (win_count / len(past_logs)) * 100.0 if past_logs else 0
    
    # 3. Build HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>V29 Universal Alpha - Command Center</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg: #050505;
            --panel: rgba(20, 20, 25, 0.8);
            --accent: #00f2ff;
            --accent-glow: rgba(0, 242, 255, 0.3);
            --success: #00ff88;
            --danger: #ff0055;
            --text: #e0e0e0;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: 'Inter', sans-serif;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 20% 20%, rgba(0, 242, 255, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 80% 80%, rgba(255, 0, 85, 0.05) 0%, transparent 40%);
        }}

        header {{
            padding: 2rem;
            text-align: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(10px);
            position: sticky;
            top: 0;
            z-index: 100;
        }}

        h1 {{
            font-family: 'Orbitron', sans-serif;
            font-size: 2.5rem;
            letter-spacing: 4px;
            background: linear-gradient(90deg, #00f2ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 20px var(--accent-glow);
        }}

        .container {{
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 1rem;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }}

        .card {{
            background: var(--panel);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 1.5rem;
            backdrop-filter: blur(15px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            transition: transform 0.3s ease;
        }}

        .card:hover {{
            transform: translateY(-5px);
            border-color: var(--accent);
        }}

        h2 {{
            font-family: 'Orbitron', sans-serif;
            font-size: 1.2rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--accent);
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .stat-item {{
            text-align: center;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 12px;
        }}

        .stat-value {{
            font-size: 1.5rem;
            font-weight: 700;
            margin-top: 5px;
        }}

        .position-list, .history-list {{
            list-style: none;
        }}

        .item {{
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            background: rgba(255, 255, 255, 0.03);
            border-left: 4px solid var(--accent);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .item.win {{ border-left-color: var(--success); }}
        .item.loss {{ border-left-color: var(--danger); }}

        .ticker {{ font-weight: 600; font-family: 'Orbitron', sans-serif; }}
        .side {{ font-size: 0.8rem; opacity: 0.7; }}
        .pnl {{ font-weight: 700; font-size: 1.2rem; }}

        .pnl.plus {{ color: var(--success); }}
        .pnl.minus {{ color: var(--danger); }}

        .details {{ font-size: 0.8rem; opacity: 0.6; margin-top: 5px; }}

        .chart-container {{
            margin-top: 2rem;
            height: 300px;
        }}

        @media (max-width: 900px) {{
            .container {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>V29 ALPHA COMMAND</h1>
        <p style="opacity: 0.5; font-size: 0.9rem; margin-top: 5px;">Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </header>

    <div class="container">
        <!-- Dashboard Stats -->
        <div class="card" style="grid-column: span 2;">
            <h2>📊 BATTLE OVERVIEW</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <p>Total LifeCycles</p>
                    <p class="stat-value">{len(past_logs)}</p>
                </div>
                <div class="stat-item">
                    <p>Accumulated PnL</p>
                    <p class="stat-value" style="color: {'var(--success)' if total_pnl > 0 else 'var(--danger)'}">{total_pnl:+.2f}%</p>
                </div>
                <div class="stat-item">
                    <p>Win Rate</p>
                    <p class="stat-value">{win_rate:.1f}%</p>
                </div>
            </div>
            <div class="chart-container">
                <canvas id="equityChart"></canvas>
            </div>
        </div>

        <!-- Active Positions -->
        <div class="card">
            <h2>⚔️ ACTIVE SOLDIERS (Positions)</h2>
            <ul class="position-list">
    """
    
    if not active_positions:
        html_content += """<li style="text-align:center; opacity:0.5; padding: 2rem;">No active positions. All units at base.</li>"""
    else:
        for ticker, pos in active_positions.items():
            pnl_curr = pos.get('mfe', 0.0) * 100.0 # Just for visualization
            side = pos.get('side', '').upper()
            html_content += f"""
                <li class="item">
                    <div>
                        <div class="ticker">{ticker} <span class="side">{side}</span></div>
                        <div class="details">Entry: {pos.get('entry_price', 0.0):.4f}</div>
                    </div>
                    <div class="pnl plus">+{pnl_curr:.2f}% <span style="font-size:0.7rem; color:var(--text); opacity:0.5;">(MFE)</span></div>
                </li>
            """

    html_content += """
            </ul>
        </div>

        <!-- Recent Archive -->
        <div class="card">
            <h2>📜 EXPERIENCE ARCHIVE (History)</h2>
            <ul class="history-list">
    """
    
    if not past_logs:
        html_content += """<li style="text-align:center; opacity:0.5; padding: 2rem;">No tactical records yet.</li>"""
    else:
        for log in reversed(past_logs[-10:]):
            pnl = log.get('performance', {}).get('pnl_pct', 0.0)
            ticker = log.get('ticker', 'N/A')
            status_class = "win" if pnl > 0 else "loss"
            pnl_class = "plus" if pnl > 0 else "minus"
            html_content += f"""
                <li class="item {status_class}">
                    <div>
                        <div class="ticker">{ticker}</div>
                        <div class="details">{log.get('exit', {}).get('reason', 'N/A')}</div>
                    </div>
                    <div class="pnl {pnl_class}">{pnl:+.2f}%</div>
                </li>
            """

    # Equity Curve Data
    equity_curve = [0.0]
    total = 0.0
    for log in past_logs:
        total += log.get('performance', {}).get('pnl_pct', 0.0)
        equity_curve.append(total)

    html_content += f"""
            </ul>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('equityChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {list(range(len(equity_curve)))},
                datasets: [{{
                    label: 'Cumulative PnL (%)',
                    data: {equity_curve},
                    borderColor: '#00f2ff',
                    backgroundColor: 'rgba(0, 242, 255, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 2
                }}]
            }},
            options: {{
                responsive: True,
                maintainAspectRatio: False,
                scales: {{
                    y: {{ grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#888' }} }},
                    x: {{ grid: {{ display: False }}, ticks: {{ color: '#888' }} }}
                }},
                plugins: {{
                    legend: {{ display: False }}
                }}
            }}
        }});
    </script>
</body>
</html>
    """
    
    with open(DASHBOARD_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✨ Dashboard generated: {DASHBOARD_FILE}")

if __name__ == "__main__":
    generate_dashboard()
