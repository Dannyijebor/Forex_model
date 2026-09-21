# bot/data_collector.py
import MetaTrader5 as mt5
import pandas as pd
import logging
import os
from datetime import datetime

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------------- MT5 CONNECTION ----------------
def connect_mt5(login, password, server):
    if not mt5.initialize():
        logging.error(f"MT5 initialize failed: {mt5.last_error()}")
        return False
    if not mt5.login(login, password=password, server=server):
        logging.error(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False
    logging.info(f"Connected to MT5 account {login}")
    return True

def disconnect_mt5():
    mt5.shutdown()
    logging.info("Disconnected MT5")

# ---------------- MARKET DATA ----------------
def fetch_candles(symbol, timeframe, n):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Failed to fetch candles for {symbol}")
    
    df = pd.DataFrame(rates)
    df['Date'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','tick_volume':'Volume'}, inplace=True)
    df = df[['Date','Open','High','Low','Close','Volume']]
    return df

def save_live_data(df, symbol, folder="data/live"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, f"{symbol}_live.csv")
    header = not os.path.exists(file_path)
    df.to_csv(file_path, mode='a', index=False, header=header)
    logging.info(f"Saved {len(df)} rows to {file_path}")

# ---------------- SYMBOL CHECK ----------------
def check_symbol(symbol):
    if not mt5.symbol_select(symbol, True):
        logging.error(f"Symbol {symbol} cannot be selected in MT5")
        return False
    return True

# ---------------- EXAMPLE USAGE ----------------
if __name__ == "__main__":
    LOGIN = 297307412
    PASSWORD = "Paul_mark2003"
    SERVER = "Exness-MT5Trial9"
    SYMBOL = "USDJPYm"
    TIMEFRAME = mt5.TIMEFRAME_M5
    CANDLES_N = 100

    if connect_mt5(LOGIN, PASSWORD, SERVER):
        if check_symbol(SYMBOL):
            df = fetch_candles(SYMBOL, TIMEFRAME, CANDLES_N)
            save_live_data(df, SYMBOL)
        disconnect_mt5()