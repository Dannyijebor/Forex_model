# bot/phase6_bot.py
import os
import sys
import logging
import time
from datetime import datetime
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import pickle

from bot.data_collector import connect_mt5, disconnect_mt5, fetch_candles, check_symbol

# ---------------- CONFIG ----------------
LOGIN = 297307412
PASSWORD = "Paul_mark2003"
SERVER = "Exness-MT5Trial9"
SYMBOL = "USDJPYm"
TIMEFRAME = mt5.TIMEFRAME_M5
CANDLES_N = 500
DRY_RUN = False
RISK_PERCENT = 0.5
LOT_MIN = 0.01
LOG_CSV = "trade_log.csv"
import os
import joblib

MODEL_PATH = "models/model_latest.joblib"
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
else:
    print(f"No model found at {MODEL_PATH}, running in dynamic mode.")
NEWS_CHECK = True
import requests

RAPIDAPI_KEY = "YOUR_RAPIDAPI_KEY"
NEWS_URL = "https://forex-api2.p.rapidapi.com/economic-calendar"

def get_latest_news():
    headers = {
        "x-rapidapi-host": "forex-api2.p.rapidapi.com",
        "x-rapidapi-key": RAPIDAPI_KEY
    }
    params = {
        "includeVolatilities": "NONE;LOW;MEDIUM;HIGH"
    }
    response = requests.get(NEWS_URL, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        # Example: return highest volatility event or None
        events = data.get("events", [])
        if events:
            return events[0]  # or filter by volatility
        return None
    else:
        return None

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------------- INDICATORS ----------------
def add_indicators(df):
    df = df.copy()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['ATR14'] = (df['High'] - df['Low']).rolling(14).mean()
    df['Volatility'] = df['Close'].rolling(20).std()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Close_lag1'] = df['Close'].shift(1)
    df['Close_lag2'] = df['Close'].shift(2)
    df['Close_lag3'] = df['Close'].shift(3)
    return df.dropna()

# ---------------- MODEL ----------------
def load_model(path=MODEL_PATH):
    if os.path.exists(path) and os.path.getsize(path) > 100:
        with open(path, "rb") as f:
            model = pickle.load(f)
        logging.info(f"Loaded model from {path}")
        return model
    logging.warning(f"Model not found or empty at {path}")
    return None

# ---------------- SIGNAL ----------------
def generate_signal(latest, model=None):
    if model:
        features = ['Open','High','Low','Close','Volume','MA10','MA20','EMA10','EMA20','Volatility','ATR14','RSI','Close_lag1','Close_lag2','Close_lag3']
        X = latest[features].iloc[-1:].values
        pred = model.predict(X)[0]
        return int(pred)
    else:
        signal = 0
        if latest['MA10'].iloc[-1] > latest['MA20'].iloc[-1] and latest['EMA10'].iloc[-1] > latest['EMA20'].iloc[-1]:
            signal = 1
        elif latest['MA10'].iloc[-1] < latest['MA20'].iloc[-1] and latest['EMA10'].iloc[-1] < latest['EMA20'].iloc[-1]:
            signal = -1
        if latest['RSI'].iloc[-1] > 70:
            signal = -1
        elif latest['RSI'].iloc[-1] < 30:
            signal = 1
        return signal

# ---------------- POSITION SIZING ----------------
def compute_lots(symbol, sl_pips):
    account = mt5.account_info()
    if account is None:
        raise RuntimeError("Account info not found")
    equity = account.equity
    symbol_info = mt5.symbol_info(symbol)
    contract_size = symbol_info.trade_contract_size or 100000
    point = symbol_info.point
    pip_value_per_lot = contract_size * (point * 10)
    lots = equity * (RISK_PERCENT / 100) / (sl_pips * pip_value_per_lot)
    return max(round(lots,2), LOT_MIN)

# ---------------- ORDER EXECUTION ----------------
def send_order(symbol, action, lots, sl_pips=None, tp_pips=None):
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if action=="BUY" else tick.bid
    point = mt5.symbol_info(symbol).point
    sl = tp = 0
    if sl_pips and tp_pips:
        if action=="BUY":
            sl = price - sl_pips*point*10
            tp = price + tp_pips*point*10
        else:
            sl = price + sl_pips*point*10
            tp = price - tp_pips*point*10
    if DRY_RUN:
        logging.info(f"[DRY_RUN] Prepared order: {action} {lots} lots, SL: {sl}, TP: {tp}")
        return {"retcode": 0, "comment": "DRY_RUN"}
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lots,
        "type": mt5.ORDER_TYPE_BUY if action=="BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 234000,
        "comment": "Phase6_Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    result = mt5.order_send(request)
    logging.info(f"Order result: {result}")
    return result

# ---------------- NEWS CHECK ----------------
def check_news():
    if not NEWS_CHECK:
        return True
    logging.info(f"News check: {NEWS_OPTION} → no major news detected (placeholder)")
    return True

# ---------------- LOGGING ----------------
def log_trade(row):
    df = pd.DataFrame([row])
    header = not os.path.exists(LOG_CSV)
    df.to_csv(LOG_CSV, mode='a', index=False, header=header)

# ---------------- MAIN LOOP ----------------
def main_loop():
    if not connect_mt5(LOGIN, PASSWORD, SERVER):
        return
    if not check_symbol(SYMBOL):
        disconnect_mt5()
        return

    model = load_model()

    try:
        while True:
            df = fetch_candles(SYMBOL, TIMEFRAME, CANDLES_N)
            df_ind = add_indicators(df)
            latest = df_ind.iloc[-1:]

            if not check_news():
                logging.info("Skipping trade due to news")
                time.sleep(60)
                continue

            signal = generate_signal(df_ind, model)
            action = "BUY" if signal==1 else "SELL" if signal==-1 else None

            log_row = {
                "time": datetime.utcnow(),
                "symbol": SYMBOL,
                "signal": signal,
                "action": action,
                "price": float(latest['Close'].iloc[0])
            }

            if action:
                sl_pips = int(df_ind['ATR14'].iloc[-1]) or 20
                tp_pips = sl_pips * 2
                lots = compute_lots(SYMBOL, sl_pips)
                log_row.update({"lots": lots, "sl_pips": sl_pips, "tp_pips": tp_pips})
                res = send_order(SYMBOL, action, lots, sl_pips, tp_pips)
                log_row.update({"order_result": str(res)})

            log_trade(log_row)
            logging.info(f"Step completed: {log_row}")
            time.sleep(30)

    finally:
        disconnect_mt5()