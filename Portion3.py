# ---------------- Portion 3: Real-Time Risk Control ----------------
import MetaTrader5 as mt5
import logging
from datetime import datetime, time as dt_time
import requests

# ---------------- CONFIG ----------------
MAX_DRAWDOWN_PCT = 5.0        # Max account drawdown allowed (%)
ACTIVE_SESSIONS = [           # Active trading sessions (UTC)
    ("00:00", "08:00"),       # Asia
    ("08:00", "16:00"),       # Europe
    ("16:00", "23:59")        # US
]
RAPIDAPI_HOST = "forex-api2.p.rapidapi.com"
RAPIDAPI_KEY = "568f2b2373msh742cabe0ad35154p16a6ecjsn40bc7e5787c8"
NEWS_LOOKBACK_MINUTES = 120  # Look at last 2 hours

# ---------------- Helper Functions ----------------
def is_high_impact_news(symbol):
    """
    Check for recent high-impact news that could affect trading.
    Uses Forex news API; returns True if news detected.
    """
    url = f"https://forexnewsapi.com/api/v1?currencypair={symbol}&items=20&token={NEWS_API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        now = datetime.utcnow()
        for item in data:
            dt = datetime.fromisoformat(item.get("date")[:19])
            impact = item.get("impact", "").lower()
            if (now - dt).total_seconds() / 60 <= NEWS_LOOKBACK_MINUTES and impact in ("high", "major"):
                logging.info(f"High-impact news detected for {symbol}: {item['title']}")
                return True
        return False
    except Exception as e:
        logging.warning(f"News API failed: {e}")
        return False

def check_max_drawdown(max_drawdown_pct=MAX_DRAWDOWN_PCT):
    """
    Stops trading if equity drops below threshold.
    """
    account_info = mt5.account_info()
    if account_info is None:
        logging.warning("Unable to fetch account info.")
        return False
    balance = account_info.balance
    equity = account_info.equity
    drawdown_pct = (balance - equity) / balance * 100
    if drawdown_pct >= max_drawdown_pct:
        logging.warning(f"Max drawdown reached: {drawdown_pct:.2f}%. Trading paused.")
        return True
    return False

def is_active_session(active_sessions=ACTIVE_SESSIONS):
    """
    Only trade during active sessions.
    """
    now_utc = datetime.utcnow().time()
    for start_str, end_str in active_sessions:
        start = dt_time(*map(int, start_str.split(":")))
        end = dt_time(*map(int, end_str.split(":")))
        if start <= now_utc <= end:
            return True
    logging.info(f"Outside active trading sessions ({now_utc}). No trades executed.")
    return False

def check_correlation(symbols):
    """
    Avoid opening trades if positions are highly correlated.
    """
    positions = mt5.positions_get()
    open_symbols = [pos.symbol for pos in positions] if positions else []
    for s in symbols:
        if s in open_symbols:
            logging.warning(f"Skipping trade due to correlation with open position: {s}")
            return True
    return False

# ---------------- Real-Time Risk Control Wrapper ----------------
def risk_control_pass(symbol, correlated_symbols=None):
    """
    Returns True if all risk checks pass.
    """
    if is_high_impact_news(symbol):
        return False
    if check_max_drawdown():
        return False
    if not is_active_session():
        return False
    if correlated_symbols and check_correlation(correlated_symbols):
        return False
    return True

# ---------------- Example Integration ----------------
# In the trading loop (Portion 2), wrap trade preparation:
# if risk_control_pass(symbol, correlated_symbols=['EURUSD', 'GBPUSD']):
#     trade_request = prepare_trade_request_dynamic(signal, df_features, symbol)
#     execute_trade_request(trade_request)
# else:
#     logging.info("Trade skipped due to risk control checks.")