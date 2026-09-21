"""
Phase 4: Exness MT5 bot integrating Phase 3 signals.

Usage:
- Edit LOGIN, PASSWORD, SERVER, SYMBOL if needed.
- Start MT5 terminal (optional) and ensure network connectivity.
- Run: python login_exness_demo.py
"""

import time
import math
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# -------- USER CONFIG --------
LOGIN = 297307412              # your Exness demo account number (int)
PASSWORD = "Paul_mark2003"     # your Exness demo trading password (careful with sharing)
SERVER = "Exness-MT5Trial9"    # exact server shown in MT5
SYMBOL = "USDJPYm"             # symbol to trade; match broker symbol
TIMEFRAME = mt5.TIMEFRAME_M5   # timeframe to fetch (M5)
CANDLES_N = 500                # number of candles to fetch to compute indicators
DRY_RUN = False                 # True = don't send real orders; False = place orders
RISK_PERCENT = 0.5             # percent of account equity to risk per trade (e.g., 0.5%)
SL_PIPS = 40                   # Stop Loss in pips
TP_PIPS = 2                   # Take Profit in pips
LOT_MIN = 0.01                 # minimum lot for account (use broker limits)
MODEL_PATH = "model.pkl"       # optional: if you have a trained ML model, put it here

LOG_CSV = "trade_log.csv"
# ------------------------------

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ------------------ MT5 Utilities ------------------
def connect_mt5():
    """Initialize MetaTrader5 and login to account"""
    if not mt5.initialize():
        logging.error("mt5.initialize() failed: %s", mt5.last_error())
        return False
    logged_in = mt5.login(LOGIN, password=PASSWORD, server=SERVER)
    if not logged_in:
        logging.error("mt5.login failed, error: %s", mt5.last_error())
        mt5.shutdown()
        return False
    logging.info("Connected to MT5 (account: %s)", LOGIN)
    return True

def disconnect_mt5():
    mt5.shutdown()
    logging.info("Disconnected MT5")

# ------------------ Market Data ------------------
def fetch_candles(symbol, timeframe, n):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) == 0:
        raise RuntimeError("Failed to fetch rates for symbol: " + symbol)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'time': 'Date', 'tick_volume': 'Volume'}, inplace=True)
    df = df[['Date', 'open', 'high', 'low', 'close', 'Volume']]
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    return df

def add_indicators(df):
    df = df.copy()

    # Moving Averages
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # Volatility
    df['Volatility'] = df['Close'].rolling(20).std()

    # ATR
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR14'] = df['TR'].rolling(14).mean()

    # Price Range Features
    df['Range'] = df['High'] - df['Low']
    df['Body'] = abs(df['Close'] - df['Open'])
    df['UpperWick'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['LowerWick'] = df[['Close', 'Open']].min(axis=1) - df['Low']
    df['Close_pct_change'] = df['Close'].pct_change()

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # Stochastic
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low14) / (high14 - low14)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    df['RSI'] = 100 - (100 / (1 + rs))

    # Lagged Closes
    df['Close_lag1'] = df['Close'].shift(1)
    df['Close_lag2'] = df['Close'].shift(2)
    df['Close_lag3'] = df['Close'].shift(3)

    # Candle Patterns
    df['Doji'] = (df['Body'] <= (df['Range'] * 0.1)).astype(int)
    df['PinBar'] = ((df['UpperWick'] > df['Body']*2) | (df['LowerWick'] > df['Body']*2)).astype(int)
    df['BullishEngulf'] = ((df['Close'].shift(1) < df['Open'].shift(1)) & 
                           (df['Close'] > df['Open']) &
                           (df['Open'] <= df['Close'].shift(1)) &
                           (df['Close'] >= df['Open'].shift(1))).astype(int)
    df['BearishEngulf'] = ((df['Close'].shift(1) > df['Open'].shift(1)) &
                           (df['Close'] < df['Open']) &
                           (df['Open'] >= df['Close'].shift(1)) &
                           (df['Close'] <= df['Open'].shift(1))).astype(int)

    # Time Features
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Session'] = df['Hour'].apply(lambda h: 1 if 0 <= h < 8 else 2 if 8 <= h < 16 else 3)

    # Market Regime
    df['MA_slope'] = df['MA20'] - df['MA50']
    df['Range_vs_ATR'] = df['Range'] / df['ATR14']

    return df.dropna()

# ------------------ Signal Generation ------------------
def generate_signals_rules(row):
    signal = 0
    if row['MA10'] > row['MA20'] and row['EMA10'] > row['EMA20']:
        signal = 1
    elif row['MA10'] < row['MA20'] and row['EMA10'] < row['EMA20']:
        signal = -1
    if row['RSI'] > 70:
        signal = -1
    elif row['RSI'] < 30:
        signal = 1
    return signal

# ------------------ Position Sizing ------------------
def compute_position_size(symbol, sl_pips):
    account_info = mt5.account_info()
    if account_info is None:
        raise RuntimeError("Failed to get account info for position sizing")
    equity = account_info.equity
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        raise RuntimeError("Failed to get symbol_info")
    contract_size = symbol_info.trade_contract_size if symbol_info.trade_contract_size else 100000
    point = symbol_info.point
    pip_value_per_lot = 10.0
    if point and contract_size:
        pip_value_per_lot = contract_size * (point * 10)
        if pip_value_per_lot == 0:
            pip_value_per_lot = 10.0
    lots = equity * (RISK_PERCENT / 100.0) / (sl_pips * pip_value_per_lot)
    lots = max(round(lots, 2), LOT_MIN)
    if symbol_info.volume_max is not None:
        lots = min(lots, symbol_info.volume_max)
    return lots

# ------------------ Order Execution ------------------
def send_order(symbol, action, lots, sl_pips, tp_pips):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError("Failed to get tick for symbol: " + symbol)
    price = tick.ask if action == 'BUY' else tick.bid
    point = mt5.symbol_info(symbol).point
    if action == 'BUY':
        sl = price - sl_pips * point * 10
        tp = price + tp_pips * point * 10
        order_type = mt5.ORDER_TYPE_BUY
    else:
        sl = price + sl_pips * point * 10
        tp = price - tp_pips * point * 10
        order_type = mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lots,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 234000,
        "comment": "Phase4_Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    if DRY_RUN:
        logging.info(f"[DRY_RUN] Order prepared: {request}")
        return {"retcode": 0, "comment": "DRY_RUN"}
    result = mt5.order_send(request)
    logging.info(f"Order send result: {result}")
    return result

# ------------------ Trade Logging ------------------
def log_trade(row):
    df = pd.DataFrame([row])
    header = not os.path.exists(LOG_CSV)
    df.to_csv(LOG_CSV, mode='a', index=False, header=header)

# ------------------ ML Model ------------------
def load_ml_model(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            logging.info("Loaded ML model from " + path)
            return model
        except Exception as e:
            logging.error("Failed to load ML model: " + str(e))
    return None

# ------------------ Trade Management ------------------
def close_order(position):
    if position is None:
        logging.warning("No position provided to close_order()")
        return False
    symbol = position.symbol
    lot = position.volume
    order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = mt5.symbol_info_tick(symbol).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": 234000,
        "comment": "AutoClose_Phase4",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    result = mt5.order_send(request)
    if result.retcode == 10009:
        logging.info(f"✅ Successfully closed trade {position.ticket}, P/L: {position.profit:.2f}")
        return True
    else:
        logging.error(f"❌ Failed to close trade {position.ticket}, retcode: {result.retcode}")
        return False

def monitor_trade(symbol, order_ticket, tp_threshold=0.15, time_limit=300, check_interval=5):
    logging.info(f"⏳ Monitoring trade {order_ticket} for {symbol}...")
    open_time = time.time()
    while True:
        positions = mt5.positions_get(symbol=symbol)
        pos = next((p for p in positions if p.ticket == order_ticket), None) if positions else None

        if pos is None:
            # Trade might be closed
            history = mt5.history_orders_get(time_from=0, time_to=int(time.time()), symbol=symbol)
            closed_order = next((o for o in history if o.ticket == order_ticket), None) if history else None
            if closed_order:
                logging.info(f"✅ Trade already closed! Ticket: {order_ticket}, "
                             f"P/L: {closed_order.profit}, Closed at: {closed_order.price_closed}")
            else:
                logging.warning(f"Trade {order_ticket} closed but not found in history.")
            break

        elapsed = time.time() - open_time
        profit = pos.profit

        # Close trade if 5 minutes elapsed AND profit >= threshold
        if elapsed >= time_limit and profit >= tp_threshold:
            logging.info(f"⏳ 5 minutes reached and profit ${profit:.2f} ≥ {tp_threshold} → Closing trade")
            close_order(pos)
            break
        elif profit >= tp_threshold and elapsed < time_limit:
            logging.info(f"Profit ${profit:.2f} before 5 minutes → waiting for timer ({int(time_limit - elapsed)}s left)")
        elif elapsed >= time_limit and profit < tp_threshold:
            logging.info(f"⏳ 5 minutes passed but profit ${profit:.2f} < {tp_threshold} → waiting for profit")
        else:
            logging.info(f"Trade open. Elapsed: {int(elapsed)}s, Unrealized P/L: ${profit:.2f}")

        time.sleep(check_interval)

# ------------------ Main Loop ------------------
def main_loop():
    if not connect_mt5():
        return
    if not mt5.symbol_select(SYMBOL, True):
        logging.error("Failed to select symbol %s", SYMBOL)
        disconnect_mt5()
        return

    model = load_ml_model(MODEL_PATH)

    try:
        while True:
            try:
                df = fetch_candles(SYMBOL, TIMEFRAME, CANDLES_N)
                df_ind = add_indicators(df)
                latest = df_ind.iloc[-1]

                features = ['Open','High','Low','Close','Volume','MA10','MA20','MA50','EMA10','EMA20',
                            'Volatility','ATR14','TR','Range','Body','UpperWick','LowerWick','Close_pct_change',
                            'MACD','MACD_signal','MACD_hist','Stoch_K','Stoch_D','RSI','Close_lag1','Close_lag2',
                            'Close_lag3','Doji','PinBar','BullishEngulf','BearishEngulf','Hour','DayOfWeek',
                            'Session','MA_slope','Range_vs_ATR']

                if model:
                    X = df_ind[features].iloc[-1:].values
                    pred = model.predict(X)[0]
                    signal = int(pred)
                    logging.info("ML signal: %s", signal)
                else:
                    signal = generate_signals_rules(latest)
                    logging.info("Rule-based signal: %s", signal)

                action = "BUY" if signal == 1 else "SELL" if signal == -1 else None
                account = mt5.account_info()
                equity = account.equity if account else None
                log_row = {"time": datetime.utcnow(), "symbol": SYMBOL, "signal": signal,
                           "action": action, "price": float(latest['Close']), "equity": equity}

                if action:
                    try:
                        lots = compute_position_size(SYMBOL, SL_PIPS)
                    except Exception as e:
                        logging.error("Position sizing failed: %s", str(e))
                        lots = LOT_MIN
                    log_row.update({"lots": lots, "sl_pips": SL_PIPS, "tp_pips": TP_PIPS})
                    res = send_order(SYMBOL, action, lots, SL_PIPS, TP_PIPS)
                    log_row.update({"order_result": str(res)})

                    # Monitor trade
                    if not DRY_RUN and res and 'order' in res and res['order']:
                        monitor_trade(SYMBOL, res['order'])
                else:
                    log_row.update({"lots": 0, "sl_pips": 0, "tp_pips": 0, "order_result": None})

                log_trade(log_row)
                logging.info("Logged step: %s", log_row)
                time.sleep(30)

            except Exception as e:
                logging.exception("Main loop error: %s", str(e))
                time.sleep(10)

    finally:
        disconnect_mt5()

# ------------------ Run ------------------
if __name__ == "__main__":
    main_loop()