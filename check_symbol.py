# check_symbol.py
import MetaTrader5 as mt5

LOGIN = 297307412
PASSWORD = "Paul_mark2003"
SERVER = "Exness-MT5Trial9"

def list_symbols():
    if not mt5.initialize():
        print("MT5 initialize() failed:", mt5.last_error())
        return

    if not mt5.login(LOGIN, password=PASSWORD, server=SERVER):
        print("MT5 login failed:", mt5.last_error())
        mt5.shutdown()
        return

    print("Connected to MT5!")
    
    # Get all available symbols
    symbols = mt5.symbols_get()
    for s in symbols:
        print(s.name)  # prints exact symbol name

    mt5.shutdown()
    print("Disconnected MT5")

if __name__ == "__main__":
    list_symbols()