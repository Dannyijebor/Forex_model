# Portion2.py - Complete Dynamic Trade Management with Risk Integration
import MetaTrader5 as mt5
import logging
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

# ==================== CONFIGURATION ====================
class Portion2Config:
    # Risk Parameters
    RISK_PERCENT = 1.0               # % of account equity to risk per trade
    TRAILING_STOP_PIPS = 15          # Trailing stop distance in pips
    PARTIAL_CLOSE_PCT = 0.5          # Close 50% of position at partial TP
    MAX_SPREAD_PIPS = 3              # Maximum spread allowed for trade execution
    MAX_SLIPPAGE_PIPS = 5            # Maximum slippage allowed
    
    # Position Sizing
    MIN_LOT_SIZE = 0.01              # Minimum lot size
    MAX_LOT_SIZE = 10.0              # Maximum lot size
    LOT_STEP = 0.01                  # Lot size increment
    
    # Stop Loss & Take Profit
    SL_MULTIPLIER = 1.0              # ATR multiplier for SL
    TP_MULTIPLIER = 2.0              # ATR multiplier for TP
    MIN_SL_PIPS = 10                 # Minimum SL distance in pips
    MAX_SL_PIPS = 50                 # Maximum SL distance in pips
    
    # Trade Management
    ENABLE_TRAILING_STOP = True
    ENABLE_PARTIAL_CLOSE = True
    ENABLE_BREAKEVEN_STOP = True
    BREAKEVEN_TRIGGER_PIPS = 20      # Move to breakeven after X pips profit
    
    # Risk Integration
    USE_RECOVERY_MODE = True         # Reduce risk after losses
    RECOVERY_RISK_MULTIPLIER = 0.5   # Reduce risk to 50% in recovery mode
    
    # Order Settings
    DEVIATION = 20                   # Maximum price deviation
    MAGIC_NUMBER = 234000            # Base magic number for orders
    ORDER_COMMENT = "Phase6_Bot"     # Order comment prefix

config = Portion2Config()

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | Portion2 | %(message)s"
)
logger = logging.getLogger(_name_)

# ==================== POSITION SIZING ====================
def calculate_lot_size(account_balance, atr, sl_pips, risk_percent=None):
    """
    Calculate position size based on risk % and SL distance in pips.
    Enhanced with recovery mode and safety limits.
    """
    if risk_percent is None:
        risk_percent = config.RISK_PERCENT
    
    # Apply recovery mode if enabled
    risk_multiplier = 1.0
    if config.USE_RECOVERY_MODE:
        try:
            from Portion3 import get_recovery_mode_risk_multiplier
            risk_multiplier = get_recovery_mode_risk_multiplier()
        except ImportError:
            logger.warning("Portion3 not available, using full risk")
    
    # Adjust risk based on recovery mode
    adjusted_risk_percent = risk_percent * risk_multiplier
    
    # Calculate risk amount
    risk_amount = account_balance * (adjusted_risk_percent / 100.0)
    
    # Get symbol point value (for pip calculation)
    symbol = "USDJPYm"  # Default, should be passed in production
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        # Fallback approximation for JPY pairs
        pip_value = 0.0001
    else:
        # Calculate pip value based on symbol
        pip_value = 10 ** -symbol_info.digits  # 0.001 for JPY, 0.00001 for others
    
    # Calculate lots: risk_amount / (sl_pips * pip_value * contract_size)
    contract_size = 100000  # Standard forex contract
    lots = risk_amount / (sl_pips * pip_value * contract_size)
    
    # Apply limits
    lots = max(config.MIN_LOT_SIZE, min(config.MAX_LOT_SIZE, lots))
    
    # Round to nearest lot step
    lots = round(lots / config.LOT_STEP) * config.LOT_STEP
    
    if risk_multiplier < 1.0:
        logger.info(f"Recovery mode active: Risk reduced to {adjusted_risk_percent:.2f}%, "
                   f"Lots: {lots:.2f}")
    else:
        logger.debug(f"Lot calculation: Balance=${account_balance:.2f}, "
                    f"Risk={adjusted_risk_percent:.2f}%, SL={sl_pips}pips, Lots={lots:.2f}")
    
    return lots

def calculate_dynamic_lot_size(account_balance, atr, current_price, signal, volatility_ratio=1.0):
    """
    Dynamic lot sizing based on market volatility and account conditions.
    """
    # Base SL distance from ATR
    base_sl_pips = atr * config.SL_MULTIPLIER
    
    # Adjust based on volatility
    adjusted_sl_pips = base_sl_pips * volatility_ratio
    
    # Apply SL limits
    sl_pips = max(config.MIN_SL_PIPS, min(config.MAX_SL_PIPS, adjusted_sl_pips))
    
    # Calculate lots
    lots = calculate_lot_size(account_balance, atr, sl_pips)
    
    # Further adjustment based on market conditions
    try:
        from Portion3 import check_market_volatility
        # Additional volatility-based adjustments could go here
        pass
    except ImportError:
        pass
    
    return lots, sl_pips

# ==================== STOP LOSS & TAKE PROFIT ====================
def calculate_sl_tp(price, atr, signal, sl_multiplier=None, tp_multiplier=None):
    """
    Calculate dynamic SL and TP based on ATR and signal.
    Returns: (sl_price, tp_price)
    """
    if sl_multiplier is None:
        sl_multiplier = config.SL_MULTIPLIER
    if tp_multiplier is None:
        tp_multiplier = config.TP_MULTIPLIER
    
    # Calculate distances in price units (not pips)
    sl_distance = atr * sl_multiplier
    tp_distance = atr * tp_multiplier
    
    if signal == 1:  # BUY
        sl = price - sl_distance
        tp = price + tp_distance
    elif signal == -1:  # SELL
        sl = price + sl_distance
        tp = price - tp_distance
    else:
        sl, tp = None, None
    
    # Round to appropriate decimals
    if sl is not None and tp is not None:
        sl = round(sl, 5)
        tp = round(tp, 5)
        logger.debug(f"SL/TP Calculation: Price={price}, ATR={atr:.5f}, "
                    f"SL={sl} ({sl_distance*10000:.1f}pips), TP={tp} ({tp_distance*10000:.1f}pips)")
    
    return sl, tp

def calculate_sl_tp_pips(price, sl_pips, tp_pips, signal):
    """
    Calculate SL and TP based on fixed pip distances.
    """
    # Convert pips to price units (1 pip = 0.01 for JPY, 0.0001 for others)
    symbol_info = mt5.symbol_info("USDJPYm")
    if symbol_info:
        pip_value = 10 ** -symbol_info.digits
    else:
        pip_value = 0.01  # Default for JPY
    
    sl_distance = sl_pips * pip_value
    tp_distance = tp_pips * pip_value
    
    if signal == 1:  # BUY
        sl = price - sl_distance
        tp = price + tp_distance
    elif signal == -1:  # SELL
        sl = price + sl_distance
        tp = price - tp_distance
    else:
        sl, tp = None, None
    
    return round(sl, 5), round(tp, 5)

# ==================== MARKET CONDITIONS ====================
def get_current_spread(symbol):
    """
    Calculate current spread in pips.
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f"Unable to get tick data for {symbol}")
        return None
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Unable to get symbol info for {symbol}")
        return None
    
    spread_price = tick.ask - tick.bid
    spread_pips = spread_price / symbol_info.point
    
    logger.debug(f"Spread for {symbol}: {spread_pips:.1f} pips")
    return spread_pips

def check_spread_limit(symbol, max_spread=None):
    """
    Check if spread is within acceptable limits.
    Returns: True if spread is acceptable, False otherwise.
    """
    if max_spread is None:
        max_spread = config.MAX_SPREAD_PIPS
    
    spread = get_current_spread(symbol)
    if spread is None:
        return False
    
    if spread > max_spread:
        logger.warning(f"Spread too high for {symbol}: {spread:.1f} pips > {max_spread} pips")
        return False
    
    return True

def check_slippage(symbol, requested_price, executed_price=None):
    """
    Check if slippage is within acceptable limits.
    """
    if executed_price is None:
        # For order preparation, use current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return True  # Can't check, assume OK
        
        if requested_price > tick.ask:  # BUY order
            executed_price = tick.ask
        else:  # SELL order
            executed_price = tick.bid
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return True  # Can't check, assume OK
    
    slippage_pips = abs(executed_price - requested_price) / symbol_info.point
    
    if slippage_pips > config.MAX_SLIPPAGE_PIPS:
        logger.warning(f"Slippage too high: {slippage_pips:.1f} pips")
        return False
    
    return True

# ==================== TRADE MANAGEMENT ====================
def manage_open_positions(symbol):
    """
    Check open positions and apply trailing stop / partial close logic.
    Enhanced with breakeven stops and advanced management.
    """
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f"Unable to get tick data for {symbol}")
        return
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Unable to get symbol info for {symbol}")
        return
    
    for pos in positions:
        # Convert position to dictionary for easier handling
        pos_dict = {
            'ticket': pos.ticket,
            'type': pos.type,
            'volume': pos.volume,
            'price_open': pos.price_open,
            'sl': pos.sl,
            'tp': pos.tp,
            'profit': pos.profit,
            'symbol': pos.symbol
        }
        
        # Apply trailing stop if enabled
        if config.ENABLE_TRAILING_STOP:
            trailing_stop_applied = apply_trailing_stop(pos_dict, tick, symbol_info)
            if trailing_stop_applied:
                logger.info(f"Trailing stop updated for position {pos.ticket}")
        
        # Apply breakeven stop if enabled
        if config.ENABLE_BREAKEVEN_STOP:
            breakeven_applied = apply_breakeven_stop(pos_dict, tick, symbol_info)
            if breakeven_applied:
                logger.info(f"Breakeven stop applied for position {pos.ticket}")
        
        # Apply partial close if enabled
        if config.ENABLE_PARTIAL_CLOSE:
            partial_close_applied = apply_partial_close(pos_dict, tick, symbol_info)
            if partial_close_applied:
                logger.info(f"Partial close executed for position {pos.ticket}")
    
    logger.debug(f"Managed {len(positions)} positions for {symbol}")

def apply_trailing_stop(position, tick, symbol_info):
    """
    Apply trailing stop logic to a position.
    """
    try:
        if position['type'] == mt5.ORDER_TYPE_BUY:
            current_price = tick.bid
            profit_pips = (current_price - position['price_open']) / symbol_info.point
            
            # Check if we should trail
            if profit_pips >= config.TRAILING_STOP_PIPS:
                new_sl = current_price - (config.TRAILING_STOP_PIPS * symbol_info.point)
                
                # Only move SL up, not down
                if position['sl'] is None or new_sl > position['sl']:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position['ticket'],
                        "sl": new_sl,
                        "tp": position['tp']
                    }
                    result = mt5.order_send(request)
                    
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.debug(f"Trailing stop moved to {new_sl:.5f} for BUY position {position['ticket']}")
                        return True
        
        elif position['type'] == mt5.ORDER_TYPE_SELL:
            current_price = tick.ask
            profit_pips = (position['price_open'] - current_price) / symbol_info.point
            
            # Check if we should trail
            if profit_pips >= config.TRAILING_STOP_PIPS:
                new_sl = current_price + (config.TRAILING_STOP_PIPS * symbol_info.point)
                
                # Only move SL down, not up
                if position['sl'] is None or new_sl < position['sl']:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position['ticket'],
                        "sl": new_sl,
                        "tp": position['tp']
                    }
                    result = mt5.order_send(request)
                    
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.debug(f"Trailing stop moved to {new_sl:.5f} for SELL position {position['ticket']}")
                        return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error applying trailing stop: {e}")
        return False

def apply_breakeven_stop(position, tick, symbol_info):
    """
    Apply breakeven stop logic.
    """
    try:
        # Calculate current profit in pips
        if position['type'] == mt5.ORDER_TYPE_BUY:
            current_price = tick.bid
            profit_pips = (current_price - position['price_open']) / symbol_info.point
            
            # Check if we should move to breakeven
            if profit_pips >= config.BREAKEVEN_TRIGGER_PIPS:
                # Move SL to breakeven + spread
                new_sl = position['price_open'] + (symbol_info.spread * symbol_info.point)
                
                # Only move if new SL is better than current
                if position['sl'] is None or new_sl > position['sl']:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position['ticket'],
                        "sl": new_sl,
                        "tp": position['tp']
                    }
                    result = mt5.order_send(request)
                    
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.debug(f"Breakeven stop at {new_sl:.5f} for position {position['ticket']}")
                        return True
        
        elif position['type'] == mt5.ORDER_TYPE_SELL:
            current_price = tick.ask
            profit_pips = (position['price_open'] - current_price) / symbol_info.point
            
            # Check if we should move to breakeven
            if profit_pips >= config.BREAKEVEN_TRIGGER_PIPS:
                # Move SL to breakeven - spread
                new_sl = position['price_open'] - (symbol_info.spread * symbol_info.point)
                
                # Only move if new SL is better than current
                if position['sl'] is None or new_sl < position['sl']:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position['ticket'],
                        "sl": new_sl,
                        "tp": position['tp']
                    }
                    result = mt5.order_send(request)
                    
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.debug(f"Breakeven stop at {new_sl:.5f} for position {position['ticket']}")
                        return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error applying breakeven stop: {e}")
        return False

def apply_partial_close(position, tick, symbol_info):
    """
    Apply partial close logic.
    """
    try:
        # Check if TP is reached
        if position['tp'] is None:
            return False
        
        tp_reached = False
        if position['type'] == mt5.ORDER_TYPE_BUY:
            tp_reached = tick.bid >= position['tp']
        elif position['type'] == mt5.ORDER_TYPE_SELL:
            tp_reached = tick.ask <= position['tp']
        
        if tp_reached and position['volume'] > config.MIN_LOT_SIZE:
            # Calculate volume to close
            close_volume = position['volume'] * config.PARTIAL_CLOSE_PCT
            close_volume = round(close_volume / config.LOT_STEP) * config.LOT_STEP
            
            # Ensure we leave at least minimum lot
            if position['volume'] - close_volume < config.MIN_LOT_SIZE:
                close_volume = position['volume'] - config.MIN_LOT_SIZE
            
            if close_volume <= 0:
                return False
            
            # Prepare close request
            close_type = mt5.ORDER_TYPE_SELL if position['type'] == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": position['ticket'],
                "symbol": position['symbol'],
                "volume": close_volume,
                "type": close_type,
                "price": tick.ask if close_type == mt5.ORDER_TYPE_BUY else tick.bid,
                "deviation": config.DEVIATION,
                "magic": config.MAGIC_NUMBER,
                "comment": f"PartialClose_{position['ticket']}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Partial close executed: {close_volume:.2f} lots from position {position['ticket']}")
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error applying partial close: {e}")
        return False

# ==================== TRADE PREPARATION ====================
def prepare_trade_request_dynamic(signal, df_features, symbol="USDJPYm", custom_lots=None):
    """
    Prepare trade request considering ATR-based SL/TP, lot sizing, spread, and slippage.
    Enhanced with comprehensive risk checks.
    """
    # Check spread first
    if not check_spread_limit(symbol):
        logger.warning(f"Spread too high for {symbol}. Trade skipped.")
        return None
    
    # Get current tick data
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.warning(f"Tick data not available for {symbol}.")
        return None
    
    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.warning(f"Symbol info not available for {symbol}.")
        return None
    
    # Get account info
    account_info = mt5.account_info()
    if account_info is None:
        logger.warning("Account info not available.")
        return None
    
    # Get ATR for SL/TP calculation
    if df_features is None or 'ATR' not in df_features.columns or len(df_features) == 0:
        logger.warning("No ATR data available for SL/TP calculation.")
        atr = 0.001  # Default ATR for JPY
    else:
        atr = float(df_features['ATR'].iloc[-1])
    
    # Calculate price
    price = tick.ask if signal == 1 else tick.bid
    
    # Calculate SL and TP
    sl, tp = calculate_sl_tp(price, atr, signal)
    if sl is None or tp is None:
        logger.warning("Failed to calculate SL/TP.")
        return None
    
    # Calculate lot size
    if custom_lots is not None:
        lots = custom_lots
    else:
        # Calculate SL distance in pips for lot sizing
        sl_distance_pips = abs(price - sl) / symbol_info.point
        
        # Calculate dynamic lot size
        balance = account_info.balance
        volatility_ratio = 1.0
        
        # Get volatility ratio if available
        if df_features is not None and 'volatility_ratio' in df_features.columns:
            volatility_ratio = float(df_features['volatility_ratio'].iloc[-1])
        
        lots, _ = calculate_dynamic_lot_size(balance, atr, price, signal, volatility_ratio)
    
    # Verify lot size is valid
    if lots < config.MIN_LOT_SIZE:
        logger.warning(f"Calculated lot size too small: {lots:.2f}")
        return None
    
    # Determine trade type
    trade_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
    
    # Generate unique magic number
    magic = config.MAGIC_NUMBER + int(datetime.now().timestamp() % 1000)
    
    # Prepare trade request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lots,
        "type": trade_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": config.DEVIATION,
        "magic": magic,
        "comment": f"{config.ORDER_COMMENT}{'BUY' if signal==1 else 'SELL'}{datetime.now().strftime('%H%M')}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # Log trade details
    spread = get_current_spread(symbol) or 0
    logger.info(f"Prepared trade: {symbol} {'BUY' if signal==1 else 'SELL'} "
                f"@{price:.5f}, Lots: {lots:.2f}, "
                f"SL: {sl:.5f}, TP: {tp:.5f}, "
                f"Spread: {spread:.1f}pips")
    
    return request

def prepare_trade_request_fixed(signal, symbol="USDJPYm", lots=0.01, sl_pips=20, tp_pips=40):
    """
    Prepare trade request with fixed SL/TP in pips.
    """
    # Check spread
    if not check_spread_limit(symbol):
        return None
    
    # Get tick data
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    
    # Calculate price
    price = tick.ask if signal == 1 else tick.bid
    
    # Calculate SL and TP
    sl, tp = calculate_sl_tp_pips(price, sl_pips, tp_pips, signal)
    
    # Determine trade type
    trade_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
    
    # Generate magic number
    magic = config.MAGIC_NUMBER + int(datetime.now().timestamp() % 1000)
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lots,
        "type": trade_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": config.DEVIATION,
        "magic": magic,
        "comment": f"{config.ORDER_COMMENT}FIXED{'BUY' if signal==1 else 'SELL'}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    logger.info(f"Prepared fixed trade: {symbol} {'BUY' if signal==1 else 'SELL'} "
                f"@{price:.5f}, Lots: {lots:.2f}, "
                f"SL: {sl:.5f} ({sl_pips}pips), TP: {tp:.5f} ({tp_pips}pips)")
    
    return request

# ==================== TRADE ANALYSIS ====================
def analyze_position_health(symbol):
    """
    Analyze health of all positions for a symbol.
    Returns summary statistics.
    """
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None
    
    summary = {
        'total_positions': len(positions),
        'total_volume': 0,
        'total_profit': 0,
        'avg_profit_pips': 0,
        'buy_positions': 0,
        'sell_positions': 0,
        'positions': []
    }
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return summary
    
    tick = mt5.symbol_info_tick(symbol)
    
    for pos in positions:
        # Basic info
        pos_info = {
            'ticket': pos.ticket,
            'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
            'volume': pos.volume,
            'price_open': pos.price_open,
            'sl': pos.sl,
            'tp': pos.tp,
            'profit': pos.profit,
            'profit_pips': 0
        }
        
        # Calculate profit in pips
        if tick:
            if pos.type == mt5.ORDER_TYPE_BUY:
                current_price = tick.bid
                profit_pips = (current_price - pos.price_open) / symbol_info.point
            else:
                current_price = tick.ask
                profit_pips = (pos.price_open - current_price) / symbol_info.point
            
            pos_info['profit_pips'] = profit_pips
            pos_info['current_price'] = current_price
            
            # Calculate distance to SL and TP
            if pos.sl:
                pos_info['sl_distance_pips'] = abs(current_price - pos.sl) / symbol_info.point
            if pos.tp:
                pos_info['tp_distance_pips'] = abs(current_price - pos.tp) / symbol_info.point
        
        # Update summary
        summary['total_volume'] += pos.volume
        summary['total_profit'] += pos.profit
        summary['avg_profit_pips'] += pos_info['profit_pips']
        
        if pos.type == mt5.ORDER_TYPE_BUY:
            summary['buy_positions'] += 1
        else:
            summary['sell_positions'] += 1
        
        summary['positions'].append(pos_info)
    
    if summary['total_positions'] > 0:
        summary['avg_profit_pips'] /= summary['total_positions']
    
    return summary

def get_position_summary(symbol):
    """
    Get concise position summary for logging.
    """
    summary = analyze_position_health(symbol)
    if summary is None:
        return "No positions"
    
    return (f"Positions: {summary['total_positions']} "
            f"(B:{summary['buy_positions']}/S:{summary['sell_positions']}), "
            f"Volume: {summary['total_volume']:.2f}, "
            f"Profit: ${summary['total_profit']:.2f}, "
            f"Avg Pips: {summary['avg_profit_pips']:.1f}")

# ==================== UTILITY FUNCTIONS ====================
def close_all_positions(symbol=None, comment_filter=None):
    """
    Close all positions (optionally filtered by symbol or comment).
    """
    if symbol:
        positions = mt5.positions_get(symbol=symbol)
    else:
        positions = mt5.positions_get()
    
    if not positions:
        logger.info("No positions to close")
        return 0
    
    closed_count = 0
    for pos in positions:
        # Filter by comment if specified
        if comment_filter and comment_filter not in pos.comment:
            continue
        
        # Prepare close request
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        # Get current price
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            continue
        
        price = tick.ask if close_type == mt5.ORDER_TYPE_BUY else tick.bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": pos.ticket,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "price": price,
            "deviation": config.DEVIATION,
            "magic": pos.magic,
            "comment": f"CloseAll_{pos.ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            closed_count += 1
            logger.info(f"Closed position {pos.ticket} ({pos.symbol})")
    
    logger.info(f"Closed {closed_count}/{len(positions)} positions")
    return closed_count

def modify_position_sltp(ticket, new_sl=None, new_tp=None):
    """
    Modify SL and/or TP for an existing position.
    """
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
    }
    
    if new_sl is not None:
        request["sl"] = new_sl
    
    if new_tp is not None:
        request["tp"] = new_tp
    
    result = mt5.order_send(request)
    
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"Modified position {ticket}: SL={new_sl}, TP={new_tp}")
        return True
    else:
        logger.error(f"Failed to modify position {ticket}")
        return False

# ==================== TESTING ====================
def test_portion2():
    """Test Portion2 functionality"""
    print("Testing Portion2 - Dynamic Trade Management")
    
    # Initialize MT5 (for testing)
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return
    
    try:
        symbol = "USDJPYm"
        
        print(f"\n1. Testing spread check for {symbol}...")
        spread = get_current_spread(symbol)
        print(f"   Current spread: {spread:.1f} pips")
        print(f"   Within limit: {check_spread_limit(symbol)}")
        
        print(f"\n2. Testing lot calculation...")
        account_info = mt5.account_info()
        if account_info:
            lots = calculate_lot_size(account_info.balance, 0.001, 20)
            print(f"   Account balance: ${account_info.balance:.2f}")
            print(f"   Calculated lots: {lots:.2f}")
        
        print(f"\n3. Testing SL/TP calculation...")
        sl, tp = calculate_sl_tp(150.00, 0.001, 1)
        print(f"   BUY @150.00, ATR=0.001 -> SL={sl:.5f}, TP={tp:.5f}")
        
        print(f"\n4. Testing position management...")
        summary = get_position_summary(symbol)
        print(f"   Position summary: {summary}")
        
        print("\nPortion2 test complete!")
        
    finally:
        mt5.shutdown()

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    test_portion2()