# running_phase6pipeline.py - Complete 6-Phase Trading Pipeline
"""
COMPLETE 6-PHASE TRADING PIPELINE
=================================
Portion 1: Data Collection & Signal Generation
Portion 2: Risk Management & Position Sizing
Portion 3: Order Execution & Trade Management
Portion 4: Portfolio Management & Diversification
Portion 5: Performance Monitoring & Analytics
Portion 6: Main Orchestrator & Pipeline Controller
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import json
import time
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ==================== PHASE 6 PIPELINE CONFIGURATION ====================
class Phase6Config:
    """Configuration for complete 6-phase pipeline"""
    
    # General Settings
    VERSION = "1.0.0"
    PIPELINE_NAME = "Advanced 6-Phase Trading Pipeline"
    
    # Symbols & Timeframes
    SYMBOLS = ["USDJPYm"]  # Primary trading symbol
    TIMEFRAMES = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    
    # Portion 1: Signal Generation
    SIGNAL_MODEL_PATH = "models/model_latest.h5"
    SIGNAL_SCALER_PATH = "models/scaler.pkl"
    SIGNAL_CONFIDENCE_THRESHOLD = 0.65
    USE_MULTI_TIMEFRAME_SIGNALS = True
    
    # Portion 2: Risk Management
    RISK_PER_TRADE = 0.02  # 2% risk per trade
    MAX_PORTFOLIO_RISK = 0.10  # 10% max portfolio risk
    MAX_DRAWDOWN_LIMIT = 0.15  # 15% max drawdown
    STOP_LOSS_ATR_MULTIPLIER = 1.5
    TAKE_PROFIT_ATR_MULTIPLIER = 2.5
    
    # Portion 3: Order Execution
    ORDER_TYPE_BUY = mt5.ORDER_TYPE_BUY
    ORDER_TYPE_SELL = mt5.ORDER_TYPE_SELL
    ORDER_SLIPPAGE = 3  # Points
    ORDER_RETRY_ATTEMPTS = 3
    ORDER_RETRY_DELAY = 1  # seconds
    
    # Portion 4: Portfolio Management
    MAX_CONCURRENT_TRADES = 3
    MAX_CORRELATED_POSITIONS = 2
    PORTFOLIO_REBALANCE_HOURS = 24
    
    # Portion 5: Performance Monitoring
    PERFORMANCE_UPDATE_INTERVAL = 60  # seconds
    SAVE_PERFORMANCE_REPORTS = True
    ALERT_ON_DRAWDOWN = True
    DRAWDOWN_ALERT_THRESHOLD = 0.08  # 8%
    
    # Portion 6: Pipeline Control
    PIPELINE_RUN_INTERVAL = 60  # seconds between pipeline runs
    MAX_PIPELINE_RUNS = None  # None for infinite, or set number
    ENABLE_LIVE_TRADING = False  # Set to True for actual trading
    LOG_ALL_TRADES = True
    
    # Logging
    LOG_LEVEL = logging.INFO
    LOG_FILE = "logs/phase6_pipeline.log"
    TRADE_LOG_FILE = "logs/trade_history.jsonl"
    
    # Database/Storage
    SAVE_STATE_PATH = "state/pipeline_state.pkl"
    SAVE_STATE_INTERVAL = 300  # seconds

# ==================== PHASE 1: SIGNAL GENERATION ====================
class Portion1_SignalGenerator:
    """Portion 1: Advanced Data Collection & Signal Generation"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Portion1")
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_signal_model()
    
    def load_signal_model(self):
        """Load the trained signal generation model"""
        try:
            # In a real implementation, you'd load your actual model
            self.logger.info("Signal model loaded (placeholder)")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load signal model: {e}")
            return False
    
    def fetch_market_data(self, symbol, timeframe="H1", bars=100):
        """Fetch market data from MT5"""
        try:
            if not mt5.initialize():
                self.logger.error("MT5 not initialized")
                return None
            
            tf = self.config.TIMEFRAMES.get(timeframe, mt5.TIMEFRAME_H1)
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
            
            if rates is None:
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def generate_features(self, df):
        """Generate technical features from market data"""
        if df is None or len(df) < 20:
            return None
        
        features = df.copy()
        
        # Basic price features
        features['returns'] = features['close'].pct_change()
        features['range'] = (features['high'] - features['low']) / features['close']
        
        # Moving averages
        features['sma_20'] = features['close'].rolling(20).mean()
        features['ema_12'] = features['close'].ewm(span=12).mean()
        features['ema_26'] = features['close'].ewm(span=26).mean()
        
        # RSI
        delta = features['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        features['macd'] = features['ema_12'] - features['ema_26']
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        features['bb_middle'] = features['sma_20']
        features['bb_std'] = features['close'].rolling(20).std()
        features['bb_upper'] = features['bb_middle'] + 2 * features['bb_std']
        features['bb_lower'] = features['bb_middle'] - 2 * features['bb_std']
        
        # ATR for volatility
        high_low = features['high'] - features['low']
        high_close = np.abs(features['high'] - features['close'].shift())
        low_close = np.abs(features['low'] - features['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr'] = true_range.rolling(14).mean()
        
        return features.dropna()
    
    def generate_signal(self, symbol):
        """Generate trading signal for a symbol"""
        try:
            # Fetch recent data
            df = self.fetch_market_data(symbol, "H1", 100)
            if df is None:
                return {"signal": 0, "confidence": 0, "details": "No data"}
            
            # Generate features
            features = self.generate_features(df)
            if features is None:
                return {"signal": 0, "confidence": 0, "details": "No features"}
            
            # Latest features for signal generation
            latest = features.iloc[-1]
            
            # Simple signal logic (replace with your actual model)
            signal = 0
            confidence = 0.5
            
            # RSI-based signal
            if latest['rsi'] < 30:
                signal = 1  # Buy
                confidence = min(0.8, 0.5 + (30 - latest['rsi']) / 50)
            elif latest['rsi'] > 70:
                signal = -1  # Sell
                confidence = min(0.8, 0.5 + (latest['rsi'] - 70) / 50)
            
            # MACD confirmation
            if signal == 1 and latest['macd'] > latest['macd_signal']:
                confidence += 0.1
            elif signal == -1 and latest['macd'] < latest['macd_signal']:
                confidence += 0.1
            
            # Filter by confidence threshold
            if confidence < self.config.SIGNAL_CONFIDENCE_THRESHOLD:
                signal = 0
            
            result = {
                "signal": signal,
                "confidence": float(confidence),
                "symbol": symbol,
                "price": float(latest['close']),
                "timestamp": datetime.now().isoformat(),
                "rsi": float(latest['rsi']),
                "macd": float(latest['macd']),
                "atr": float(latest['atr']),
                "details": f"RSI: {latest['rsi']:.1f}, MACD: {latest['macd']:.4f}"
            }
            
            self.logger.info(f"Signal generated for {symbol}: {signal} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return {"signal": 0, "confidence": 0, "details": f"Error: {str(e)}"}

# ==================== PHASE 2: RISK MANAGEMENT ====================
class Portion2_RiskManager:
    """Portion 2: Risk Management & Position Sizing"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Portion2")
        self.account_info = None
        self.position_history = []
        self.update_account_info()
    
    def update_account_info(self):
        """Get current account information from MT5"""
        try:
            if mt5.initialize():
                self.account_info = mt5.account_info()
                return self.account_info is not None
            return False
        except:
            # For testing without MT5
            self.account_info = type('Account', (), {
                'balance': 10000.0,
                'equity': 10000.0,
                'margin': 0.0,
                'free_margin': 10000.0,
                'leverage': 100
            })()
            return True
    
    def calculate_position_size(self, signal, symbol, price, atr):
        """Calculate position size based on risk parameters"""
        if self.account_info is None:
            return 0
        
        # Account equity
        equity = float(self.account_info.equity)
        
        # Risk amount per trade
        risk_amount = equity * self.config.RISK_PER_TRADE
        
        # Stop loss distance in points
        stop_distance_points = atr * self.config.STOP_LOSS_ATR_MULTIPLIER
        stop_distance_price = stop_distance_points * 0.01  # Simplified for JPY pairs
        
        if stop_distance_price <= 0:
            return 0
        
        # Calculate position size
        position_size = risk_amount / stop_distance_price
        
        # Normalize to lot size (0.01 = micro lot)
        min_lot = 0.01
        position_size_lots = max(min_lot, round(position_size / 1000, 2))
        
        # Adjust for available margin
        required_margin = position_size_lots * price / float(self.account_info.leverage)
        if required_margin > float(self.account_info.free_margin) * 0.8:
            position_size_lots = (float(self.account_info.free_margin) * 0.8 * 
                                 float(self.account_info.leverage)) / price
        
        self.logger.info(f"Position size for {symbol}: {position_size_lots:.2f} lots "
                        f"(risk: ${risk_amount:.2f}, SL: {stop_distance_price:.4f})")
        
        return position_size_lots
    
    def check_portfolio_risk(self, new_position_size, symbol):
        """Check if new position exceeds portfolio risk limits"""
        # In a full implementation, check all open positions
        # For now, simple check
        if self.account_info is None:
            return False
        
        equity = float(self.account_info.equity)
        current_risk = 0  # Would calculate from open positions
        
        new_risk = new_position_size * self.config.RISK_PER_TRADE * equity
        
        if (current_risk + new_risk) / equity > self.config.MAX_PORTFOLIO_RISK:
            self.logger.warning(f"Portfolio risk limit exceeded for {symbol}")
            return False
        
        return True
    
    def calculate_stop_take(self, signal, entry_price, atr):
        """Calculate stop loss and take profit levels"""
        if signal == 1:  # Buy
            stop_loss = entry_price - (atr * self.config.STOP_LOSS_ATR_MULTIPLIER * 0.01)
            take_profit = entry_price + (atr * self.config.TAKE_PROFIT_ATR_MULTIPLIER * 0.01)
        elif signal == -1:  # Sell
            stop_loss = entry_price + (atr * self.config.STOP_LOSS_ATR_MULTIPLIER * 0.01)
            take_profit = entry_price - (atr * self.config.TAKE_PROFIT_ATR_MULTIPLIER * 0.01)
        else:
            return 0, 0
        
        return round(stop_loss, 5), round(take_profit, 5)

# ==================== PHASE 3: ORDER EXECUTION ====================
class Portion3_OrderExecutor:
    """Portion 3: Order Execution & Trade Management"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Portion3")
        self.initialized = self.initialize_mt5()
    
    def initialize_mt5(self):
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                self.logger.warning("MT5 initialization failed - running in simulation mode")
                return False
            self.logger.info("MT5 initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"MT5 initialization error: {e}")
            return False
    
    def execute_order(self, symbol, order_type, volume, price, sl, tp, comment=""):
        """Execute a trade order"""
        if not self.config.ENABLE_LIVE_TRADING:
            self.logger.info(f"[SIMULATION] Would execute: {symbol} {order_type} "
                           f"{volume} lots @ {price}, SL: {sl}, TP: {tp}")
            return self.create_simulated_order(symbol, order_type, volume, price, sl, tp, comment)
        
        if not self.initialized:
            self.logger.error("MT5 not initialized for live trading")
            return None
        
        try:
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": self.config.ORDER_SLIPPAGE,
                "magic": 234000,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {result.retcode} - {result.comment}")
                return None
            
            self.logger.info(f"Order executed: {symbol} {order_type} {volume} lots "
                           f"Order ID: {result.order}")
            
            # Log the trade
            self.log_trade(result, symbol, order_type, volume, price, sl, tp, comment)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Order execution error: {e}")
            return None
    
    def create_simulated_order(self, symbol, order_type, volume, price, sl, tp, comment):
        """Create simulated order for testing"""
        simulated_order = {
            'order': np.random.randint(100000, 999999),
            'retcode': mt5.TRADE_RETCODE_DONE,
            'volume': volume,
            'price': price,
            'bid': price,
            'ask': price,
            'comment': f"[SIM]{comment}",
            'request_id': 0,
            'retcode_external': 0
        }
        
        # Log simulated trade
        trade_log = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'type': 'BUY' if order_type == mt5.ORDER_TYPE_BUY else 'SELL',
            'volume': volume,
            'entry_price': price,
            'stop_loss': sl,
            'take_profit': tp,
            'status': 'SIMULATED',
            'comment': comment
        }
        
        if self.config.LOG_ALL_TRADES:
            try:
                with open(self.config.TRADE_LOG_FILE, 'a') as f:
                    f.write(json.dumps(trade_log) + '\n')
            except:
                pass
        
        return type('OrderResult', (), simulated_order)()
    
    def log_trade(self, result, symbol, order_type, volume, price, sl, tp, comment):
        """Log trade to file"""
        if not self.config.LOG_ALL_TRADES:
            return
        
        trade_log = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'type': 'BUY' if order_type == mt5.ORDER_TYPE_BUY else 'SELL',
            'volume': volume,
            'entry_price': price,
            'stop_loss': sl,
            'take_profit': tp,
            'order_id': result.order,
            'status': 'EXECUTED',
            'comment': comment
        }
        
        try:
            with open(self.config.TRADE_LOG_FILE, 'a') as f:
                f.write(json.dumps(trade_log) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to log trade: {e}")
    
    def close_position(self, position_id, symbol, volume):
        """Close an existing position"""
        if not self.config.ENABLE_LIVE_TRADING:
            self.logger.info(f"[SIMULATION] Would close position {position_id} for {symbol}")
            return True
        
        try:
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return False
            
            # Determine close direction (opposite of open)
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                return False
            
            position = positions[0]
            close_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": close_type,
                "position": position_id,
                "price": tick.ask if close_type == mt5.ORDER_TYPE_SELL else tick.bid,
                "deviation": self.config.ORDER_SLIPPAGE,
                "magic": 234000,
                "comment": "Close by Phase6 Pipeline",
                "type_time": mt5.ORDER_TIME_GTC,
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"Position {position_id} closed for {symbol}")
                return True
            else:
                self.logger.error(f"Failed to close position {position_id}: {result.comment}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return False

# ==================== PHASE 4: PORTFOLIO MANAGEMENT ====================
class Portion4_PortfolioManager:
    """Portion 4: Portfolio Management & Diversification"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Portion4")
        self.open_positions = {}
        self.last_rebalance = None
    
    def update_open_positions(self):
        """Update list of open positions"""
        try:
            if not mt5.initialize():
                # Simulate for testing
                return self.open_positions
            
            positions = mt5.positions_get()
            self.open_positions = {}
            
            for pos in positions:
                symbol = pos.symbol
                if symbol not in self.open_positions:
                    self.open_positions[symbol] = []
                
                self.open_positions[symbol].append({
                    'ticket': pos.ticket,
                    'type': pos.type,
                    'volume': pos.volume,
                    'open_price': pos.price_open,
                    'current_price': pos.price_current,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'open_time': pd.to_datetime(pos.time, unit='s')
                })
            
            return self.open_positions
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
            return self.open_positions
    
    def check_position_limit(self, symbol):
        """Check if we can open a new position for this symbol"""
        self.update_open_positions()
        
        # Check max concurrent trades
        total_positions = sum(len(positions) for positions in self.open_positions.values())
        if total_positions >= self.config.MAX_CONCURRENT_TRADES:
            self.logger.warning(f"Max concurrent trades reached ({self.config.MAX_CONCURRENT_TRADES})")
            return False
        
        # Check symbol-specific limits
        if symbol in self.open_positions:
            current_symbol_positions = len(self.open_positions[symbol])
            
            # Don't open same direction positions
            # In real implementation, check direction correlation
            
            if current_symbol_positions >= 2:  # Max 2 positions per symbol
                self.logger.warning(f"Max positions for {symbol} reached")
                return False
        
        return True
    
    def calculate_correlation(self, symbol1, symbol2):
        """Calculate correlation between two symbols (simplified)"""
        # In real implementation, calculate based on historical returns
        # For now, simple currency group check
        majors = ['EUR', 'GBP', 'AUD', 'NZD', 'USD', 'CAD', 'CHF', 'JPY']
        
        base1 = symbol1[:3]
        base2 = symbol2[:3]
        
        # If same base or quote currency, high correlation
        if base1 in symbol2 or base2 in symbol1:
            return 0.7
        
        return 0.3  # Default low correlation
    
    def needs_rebalance(self):
        """Check if portfolio needs rebalancing"""
        if self.last_rebalance is None:
            return True
        
        time_since_rebalance = datetime.now() - self.last_rebalance
        return time_since_rebalance.total_seconds() > (self.config.PORTFOLIO_REBALANCE_HOURS * 3600)
    
    def get_portfolio_summary(self):
        """Get portfolio summary"""
        self.update_open_positions()
        
        summary = {
            'total_positions': 0,
            'total_symbols': 0,
            'total_volume': 0,
            'total_profit': 0,
            'symbols': {}
        }
        
        for symbol, positions in self.open_positions.items():
            summary['symbols'][symbol] = {
                'positions': len(positions),
                'volume': sum(p['volume'] for p in positions),
                'profit': sum(p['profit'] for p in positions)
            }
            summary['total_positions'] += len(positions)
            summary['total_volume'] += sum(p['volume'] for p in positions)
            summary['total_profit'] += sum(p['profit'] for p in positions)
        
        summary['total_symbols'] = len(self.open_positions)
        
        return summary

# ==================== PHASE 5: PERFORMANCE MONITORING ====================
class Portion5_PerformanceMonitor:
    """Portion 5: Performance Monitoring & Analytics"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Portion5")
        self.performance_history = []
        self.start_time = datetime.now()
        self.start_equity = 10000.0  # Default, will be updated
        self.max_equity = 10000.0
        self.max_drawdown = 0.0
    
    def update_performance(self):
        """Update performance metrics"""
        try:
            # Get account info
            if mt5.initialize():
                account = mt5.account_info()
                equity = float(account.equity)
                balance = float(account.balance)
            else:
                # Simulated values for testing
                equity = 10000.0 + np.random.normal(0, 50)
                balance = 10000.0
            
            # Update max equity and drawdown
            if equity > self.max_equity:
                self.max_equity = equity
            
            current_drawdown = (self.max_equity - equity) / self.max_equity if self.max_equity > 0 else 0
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Get open positions
            if mt5.initialize():
                positions = mt5.positions_get()
                open_positions = len(positions)
                open_profit = sum(pos.profit for pos in positions)
            else:
                open_positions = 0
                open_profit = 0
            
            # Performance record
            record = {
                'timestamp': datetime.now().isoformat(),
                'equity': equity,
                'balance': balance,
                'open_positions': open_positions,
                'open_profit': open_profit,
                'max_drawdown': self.max_drawdown,
                'current_drawdown': current_drawdown,
                'return_pct': ((equity - self.start_equity) / self.start_equity * 100) 
                             if self.start_equity > 0 else 0
            }
            
            self.performance_history.append(record)
            
            # Check for drawdown alert
            if (self.config.ALERT_ON_DRAWDOWN and 
                current_drawdown > self.config.DRAWDOWN_ALERT_THRESHOLD):
                self.logger.warning(f"HIGH DRAWDOWN ALERT: {current_drawdown:.2%} "
                                  f"(Max: {self.max_drawdown:.2%})")
            
            # Save performance report periodically
            if (len(self.performance_history) % 10 == 0 and 
                self.config.SAVE_PERFORMANCE_REPORTS):
                self.save_performance_report()
            
            return record
            
        except Exception as e:
            self.logger.error(f"Error updating performance: {e}")
            return None
    
    def save_performance_report(self):
        """Save performance report to file"""
        try:
            if not self.performance_history:
                return
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'pipeline_start': self.start_time.isoformat(),
                'total_runtime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                'current_equity': self.performance_history[-1]['equity'],
                'max_drawdown': self.max_drawdown,
                'total_return_pct': self.performance_history[-1]['return_pct'],
                'performance_history': self.performance_history[-100:]  # Last 100 records
            }
            
            filename = f"logs/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Performance report saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving performance report: {e}")
    
    def get_performance_summary(self):
        """Get summary of performance metrics"""
        if not self.performance_history:
            return {}
        
        latest = self.performance_history[-1]
        
        summary = {
            'Runtime': f"{(datetime.now() - self.start_time).total_seconds() / 3600:.1f} hours",
            'Current Equity': f"${latest['equity']:.2f}",
            'Total Return': f"{latest['return_pct']:.2f}%",
            'Max Drawdown': f"{self.max_drawdown:.2%}",
            'Current Drawdown': f"{latest['current_drawdown']:.2%}",
            'Open Positions': latest['open_positions'],
            'Open Profit': f"${latest['open_profit']:.2f}"
        }
        
        return summary

# ==================== PHASE 6: MAIN PIPELINE ORCHESTRATOR ====================
class Phase6_PipelineOrchestrator:
    """Portion 6: Main Orchestrator - Controls the complete 6-phase pipeline"""
    
    def __init__(self, config_class=Phase6Config):
        self.config = config_class()
        self.logger = self._setup_logging()
        
        # Initialize all portions
        self.portion1 = None  # Signal Generation
        self.portion2 = None  # Risk Management
        self.portion3 = None  # Order Execution
        self.portion4 = None  # Portfolio Management
        self.portion5 = None  # Performance Monitoring
        
        self.initialize_pipeline()
        self.logger.info("Phase 6 Pipeline initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=self.config.LOG_LEVEL,
            format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(self.config.LOG_FILE),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger("Phase6")
    
    def initialize_pipeline(self):
        """Initialize all pipeline components"""
        try:
            self.logger.info("=" * 60)
            self.logger.info(f"Initializing {self.config.PIPELINE_NAME} v{self.config.VERSION}")
            self.logger.info("=" * 60)
            
            # Initialize MT5
            if not mt5.initialize():
                self.logger.warning("MT5 not available - running in simulation mode")
            
            # Initialize all portions
            self.portion1 = Portion1_SignalGenerator(self.config)
            self.portion2 = Portion2_RiskManager(self.config)
            self.portion3 = Portion3_OrderExecutor(self.config)
            self.portion4 = Portion4_PortfolioManager(self.config)
            self.portion5 = Portion5_PerformanceMonitor(self.config)
            
            self.logger.info("All 6 portions initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def run_pipeline_cycle(self, symbol=None):
        """Run one complete cycle of the 6-phase pipeline"""
        if symbol is None:
            symbol = self.config.SYMBOLS[0]
        
        self.logger.info(f"Starting pipeline cycle for {symbol}")
        
        try:
            # ===== PHASE 1: Generate Signal =====
            signal_data = self.portion1.generate_signal(symbol)
            
            if signal_data['signal'] == 0:
                self.logger.info(f"No trading signal for {symbol} (confidence: {signal_data['confidence']:.2f})")
                return None
            
            # ===== PHASE 2: Risk Management =====
            position_size = self.portion2.calculate_position_size(
                signal_data['signal'], 
                symbol, 
                signal_data['price'],
                signal_data.get('atr', 0.01)
            )
            
            if position_size <= 0:
                self.logger.warning(f"Position size calculation failed for {symbol}")
                return None
            
            if not self.portion2.check_portfolio_risk(position_size, symbol):
                self.logger.warning(f"Portfolio risk check failed for {symbol}")
                return None
            
            # Calculate stop loss and take profit
            sl, tp = self.portion2.calculate_stop_take(
                signal_data['signal'],
                signal_data['price'],
                signal_data.get('atr', 0.01)
            )
            
            # ===== PHASE 4: Portfolio Management Check =====
            if not self.portion4.check_position_limit(symbol):
                self.logger.warning(f"Position limit check failed for {symbol}")
                return None
            
            # ===== PHASE 3: Execute Order =====
            order_type = mt5.ORDER_TYPE_BUY if signal_data['signal'] == 1 else mt5.ORDER_TYPE_SELL
            
            comment = f"Phase6_{signal_data['signal']}_{signal_data['confidence']:.2f}"
            
            order_result = self.portion3.execute_order(
                symbol=symbol,
                order_type=order_type,
                volume=position_size,
                price=signal_data['price'],
                sl=sl,
                tp=tp,
                comment=comment
            )
            
            if order_result is None:
                self.logger.error(f"Order execution failed for {symbol}")
                return None
            
            # ===== PHASE 5: Update Performance =====
            self.portion5.update_performance()
            
            # ===== Log successful trade =====
            trade_summary = {
                'symbol': symbol,
                'signal': signal_data['signal'],
                'signal_confidence': signal_data['confidence'],
                'entry_price': signal_data['price'],
                'position_size': position_size,
                'stop_loss': sl,
                'take_profit': tp,
                'order_id': order_result.order if hasattr(order_result, 'order') else 'SIM',
                'timestamp': datetime.now().isoformat(),
                'atr': signal_data.get('atr'),
                'rsi': signal_data.get('rsi')
            }
            
            self.logger.info(f"Pipeline cycle completed for {symbol}")
            self.logger.info(f"  Trade: {symbol} {'BUY' if signal_data['signal'] == 1 else 'SELL'} "
                           f"{position_size:.2f} lots @ {signal_data['price']}")
            self.logger.info(f"  SL: {sl}, TP: {tp}, Confidence: {signal_data['confidence']:.2f}")
            
            return trade_summary
            
        except Exception as e:
            self.logger.error(f"Pipeline cycle failed: {e}", exc_info=True)
            return None
    
    def run_continuous(self, interval=None, max_runs=None):
        """Run the pipeline continuously at specified interval"""
        if interval is None:
            interval = self.config.PIPELINE_RUN_INTERVAL
        
        if max_runs is None:
            max_runs = self.config.MAX_PIPELINE_RUNS
        
        self.logger.info(f"Starting continuous pipeline execution (interval: {interval}s)")
        
        run_count = 0
        
        try:
            while True:
                if max_runs and run_count >= max_runs:
                    self.logger.info(f"Completed {max_runs} pipeline runs")
                    break
                
                run_count += 1
                self.logger.info(f"--- Pipeline Run #{run_count} ---")
                
                # Update performance monitor
                perf = self.portion5.update_performance()
                if perf:
                    self.logger.info(f"Performance: Equity=${perf['equity']:.2f}, "
                                   f"Return={perf['return_pct']:.2f}%, "
                                   f"Drawdown={perf['current_drawdown']:.2%}")
                
                # Check portfolio and rebalance if needed
                if self.portion4.needs_rebalance():
                    self.logger.info("Portfolio rebalance check triggered")
                    # In full implementation, would rebalance here
                
                # Run pipeline for each symbol
                for symbol in self.config.SYMBOLS:
                    self.run_pipeline_cycle(symbol)
                
                # Save pipeline state
                self.save_pipeline_state()
                
                # Wait for next cycle
                self.logger.info(f"Waiting {interval} seconds until next cycle...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("Pipeline stopped by user")
        except Exception as e:
            self.logger.error(f"Continuous execution failed: {e}")
        finally:
            self.shutdown()
    
    def save_pipeline_state(self):
        """Save current pipeline state to file"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'run_count': getattr(self, 'run_count', 0),
                'performance_history': self.portion5.performance_history if self.portion5 else [],
                'open_positions': self.portion4.open_positions if self.portion4 else {},
                'config': {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
            }
            
            with open(self.config.SAVE_STATE_PATH, 'wb') as f:
                pickle.dump(state, f)
            
            self.logger.debug(f"Pipeline state saved to {self.config.SAVE_STATE_PATH}")
            
        except Exception as e:
            self.logger.error(f"Failed to save pipeline state: {e}")
    
    def load_pipeline_state(self):
        """Load pipeline state from file"""
        try:
            with open(self.config.SAVE_STATE_PATH, 'rb') as f:
                state = pickle.load(f)
            
            self.logger.info(f"Pipeline state loaded from {self.config.SAVE_STATE_PATH}")
            return state
            
        except FileNotFoundError:
            self.logger.info("No saved state found, starting fresh")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load pipeline state: {e}")
            return None
    
    def get_pipeline_status(self):
        """Get current status of all pipeline components"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'pipeline': 'Running' if hasattr(self, 'portion1') else 'Stopped',
            'config': {
                'symbols': self.config.SYMBOLS,
                'live_trading': self.config.ENABLE_LIVE_TRADING,
                'risk_per_trade': self.config.RISK_PER_TRADE
            }
        }
        
        if self.portion5:
            status['performance'] = self.portion5.get_performance_summary()
        
        if self.portion4:
            status['portfolio'] = self.portion4.get_portfolio_summary()
        
        return status
    
    def shutdown(self):
        """Shutdown the pipeline gracefully"""
        self.logger.info("Shutting down Phase 6 Pipeline...")
        
        try:
            if mt5.initialize():
                mt5.shutdown()
                self.logger.info("MT5 connection closed")
            
            # Save final state
            self.save_pipeline_state()
            
            # Generate final performance report
            if self.portion5 and self.config.SAVE_PERFORMANCE_REPORTS:
                self.portion5.save_performance_report()
            
            self.logger.info("Phase 6 Pipeline shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# ==================== SIMPLIFIED INTERFACE ====================
class TradingBot:
    """Simplified interface for easy integration with notebooks"""
    
    def __init__(self, enable_live_trading=False, config_class=None):
        """Initialize TradingBot with optional live trading setting
        
        Args:
            enable_live_trading (bool): If True, executes real trades. Default False.
            config_class: Configuration class to use. Defaults to Phase6Config.
        """
        if config_class is None:
            config_class = Phase6Config
        
        self.config_class = config_class
        self.enable_live_trading = enable_live_trading
        self.pipeline = None
        self.initialize()
    
    def initialize(self):
        """Initialize the trading bot with configuration"""
        # Create a custom config class that overrides ENABLE_LIVE_TRADING
        class BotConfig(self.config_class):
            ENABLE_LIVE_TRADING = self.enable_live_trading
        
        # Initialize the full pipeline
        self.pipeline = Phase6_PipelineOrchestrator(BotConfig)
        return self.pipeline
    
    def run_single_cycle(self, symbol=None):
        """Run a single pipeline cycle
        
        Args:
            symbol (str, optional): Symbol to trade. Uses first from config if None.
        
        Returns:
            dict: Trade result or None if no trade
        """
        if self.pipeline is None:
            self.initialize()
        
        return self.pipeline.run_pipeline_cycle(symbol)
    
    def run_continuous(self, interval=None, max_runs=None):
        """Run pipeline continuously
        
        Args:
            interval (int, optional): Seconds between cycles. Uses config default if None.
            max_runs (int, optional): Maximum number of cycles. Uses config default if None.
        
        Returns:
            None: Runs until stopped or max_runs reached
        """
        if self.pipeline is None:
            self.initialize()
        
        return self.pipeline.run_continuous(interval, max_runs)
    
    def get_status(self):
        """Get current bot status
        
        Returns:
            dict: Status information
        """
        if self.pipeline is None:
            return {
                "status": "Not initialized",
                "live_trading": self.enable_live_trading,
                "initialized": False
            }
        
        status = self.pipeline.get_pipeline_status()
        status["live_trading"] = self.enable_live_trading
        return status
    
    def shutdown(self):
        """Shutdown the bot gracefully"""
        if self.pipeline:
            self.pipeline.shutdown()
            self.pipeline = None
            return True
        return False
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto shutdown"""
        self.shutdown()
    
    def __repr__(self):
        """String representation"""
        status = "initialized" if self.pipeline else "not initialized"
        return f"TradingBot(live={self.enable_live_trading}, {status})"

# ==================== EXPORT FUNCTIONS ====================
def create_trading_bot(enable_live_trading=False, config_class=None):
    """Create and return a trading bot instance
    
    Args:
        enable_live_trading (bool): If True, executes real trades
        config_class: Configuration class to use
    
    Returns:
        TradingBot: Initialized trading bot instance
    """
    return TradingBot(enable_live_trading=enable_live_trading, config_class=config_class)

def run_quick_test():
    """Run a quick test of the pipeline"""
    print("=" * 60)
    print("PHASE 6 PIPELINE QUICK TEST")
    print("=" * 60)
    
    try:
        bot = TradingBot(enable_live_trading=False)
        
        print("\n1. Running single cycle...")
        result = bot.run_single_cycle()
        
        if result:
            print(f"Trade executed for {result.get('symbol', 'N/A')}:")
            print(f"  Signal: {'BUY' if result.get('signal', 0) == 1 else 'SELL'}")
            print(f"  Size: {result.get('position_size', 0):.2f} lots")
            print(f"  Price: {result.get('entry_price', 0)}")
            print(f"  SL: {result.get('stop_loss', 0)}, TP: {result.get('take_profit', 0)}")
        else:
            print("No trade executed (no signal or checks failed)")
        
        print("\n2. Getting status...")
        status = bot.get_status()
        print(f"  Pipeline: {status.get('pipeline', 'Unknown')}")
        
        if 'config' in status:
            config = status['config']
            print(f"  Live Trading: {'ENABLED' if config.get('live_trading', False) else 'DISABLED (safe)'}")
            print(f"  Symbols: {config.get('symbols', [])}")
        
        if 'performance' in status:
            print(f"  Performance:")
            for key, value in status['performance'].items():
                print(f"    {key}: {value}")
        
        print("\n3. Shutting down...")
        bot.shutdown()
        
        print("=" * 60)
        print("QUICK TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nQUICK TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

# ==================== CONVENIENCE FUNCTIONS ====================
def run_simulation(duration_seconds=300, interval=30):
    """Run a simulation for a specified duration
    
    Args:
        duration_seconds (int): Total simulation duration in seconds
        interval (int): Seconds between trading cycles
    """
    import time
    
    print(f"Starting {duration_seconds//60} minute simulation...")
    bot = TradingBot(enable_live_trading=False)
    
    start_time = time.time()
    cycles = 0
    
    try:
        while time.time() - start_time < duration_seconds:
            cycles += 1
            print(f"\n--- Cycle {cycles} ---")
            result = bot.run_single_cycle()
            
            if result:
                print(f"Trade: {result.get('symbol')} {'BUY' if result.get('signal') == 1 else 'SELL'}")
            else:
                print("No trade")
            
            # Update performance
            status = bot.get_status()
            if 'performance' in status:
                perf = status['performance']
                print(f"Equity: {perf.get('Current Equity', 'N/A')}, "
                      f"Return: {perf.get('Total Return', 'N/A')}")
            
            # Wait for next cycle
            if time.time() - start_time < duration_seconds:
                time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        bot.shutdown()
    
    print(f"\nSimulation complete: {cycles} cycles executed")

# ==================== NOTEBOOK HELPER ====================
def get_notebook_interface():
    """Return interface functions for notebook usage"""
    return {
        'TradingBot': TradingBot,
        'create_trading_bot': create_trading_bot,
        'run_quick_test': run_quick_test,
        'run_simulation': run_simulation,
        'Phase6Config': Phase6Config,
        'Phase6_PipelineOrchestrator': Phase6_PipelineOrchestrator
    }

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 6 Complete Trading Pipeline')
    parser.add_argument('--mode', type=str, default='test',
                       choices=['test', 'continuous', 'single', 'status'],
                       help='Execution mode')
    parser.add_argument('--live', action='store_true',
                       help='Enable live trading (CAUTION)')
    parser.add_argument('--symbol', type=str, default='USDJPYm',
                       help='Trading symbol')
    parser.add_argument('--interval', type=int, default=60,
                       help='Interval for continuous mode (seconds)')
    parser.add_argument('--runs', type=int, default=None,
                       help='Max runs for continuous mode')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"PHASE 6 TRADING PIPELINE")
    print(f"Mode: {args.mode.upper()}")
    print(f"Live Trading: {'ENABLED' if args.live else 'DISABLED'}")
    print(f"{'='*60}\n")
    
    bot = create_trading_bot(enable_live_trading=args.live)
    
    try:
        if args.mode == 'test':
            run_quick_test()
        
        elif args.mode == 'single':
            result = bot.run_single_cycle(args.symbol)
            if result:
                print(f"Trade executed: {result}")
            else:
                print("No trade executed")
        
        elif args.mode == 'continuous':
            print(f"Starting continuous execution (interval: {args.interval}s)")
            bot.run_continuous(interval=args.interval, max_runs=args.runs)
        
        elif args.mode == 'status':
            status = bot.get_status()
            print(json.dumps(status, indent=2))
    
    except KeyboardInterrupt:
        print("\nPipeline stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        bot.shutdown()