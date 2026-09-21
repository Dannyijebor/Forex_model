# Portion1.py - Updated with correct volume column name
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import joblib
import pickle
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Portion1Config:
    # MT5 Configuration
    SYMBOLS = ["USDJPYm"]
    TIMEFRAMES = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    
    # Data Collection
    CANDLES_FETCH = 500  # More candles for sequence creation
    MIN_SEQUENCE_LENGTH = 60  # Match training sequence length
    
    # Model Configuration
    MODEL_TYPE = "DEEP_LEARNING"  # Options: DL (Deep Learning), HYBRID, LEGACY
    MODEL_PATH = "models/model_latest.h5"
    SCALER_PATH = "models/scaler.pkl"
    FEATURE_NAMES_PATH = "models/feature_names.pkl"
    MODEL_METADATA_PATH = "models/model_latest_metadata.pkl"
    
    # Signal Generation
    SIGNAL_CONFIDENCE_THRESHOLD = 0.65  # Higher threshold for DL
    UNCERTAINTY_THRESHOLD = 0.15  # Reject uncertain predictions
    USE_MONTE_CARLO = True  # Use dropout for uncertainty estimation
    MC_ITERATIONS = 30  # Monte Carlo iterations
    
    # Feature Engineering
    CREATE_ADVANCED_FEATURES = True
    USE_MULTI_TIMEFRAME = True
    INCLUDE_MICROSTRUCTURE = True
    
    # Risk Management
    MIN_VOLATILITY_REQUIRED = 0.0001  # Minimum volatility filter
    MAX_CORRELATION_THRESHOLD = 0.8
    
    # Logging
    LOG_LEVEL = logging.INFO

config = Portion1Config()

# ==================== INITIALIZE LOGGING ====================
logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | Portion1 | %(message)s",
    handlers=[
        logging.FileHandler("logs/portion1.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(_name_)

# ==================== DEEP LEARNING MODEL HANDLER ====================
class DLModelHandler:
    """Handles deep learning model loading, prediction, and uncertainty estimation"""
    
    def _init_(self, config):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        self.sequence_length = None
        self.load_model_artifacts()
    
    def load_model_artifacts(self):
        """Load all model artifacts"""
        try:
            # Load TensorFlow model
            self.model = keras.models.load_model(
                self.config.MODEL_PATH,
                custom_objects={'f1_score': self.f1_score_metric}
            )
            logger.info(f"Loaded DL model from {self.config.MODEL_PATH}")
            
            # Load scaler
            with open(self.config.SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"Loaded scaler from {self.config.SCALER_PATH}")
            
            # Load feature names
            with open(self.config.FEATURE_NAMES_PATH, 'rb') as f:
                self.feature_names = pickle.load(f)
            logger.info(f"Loaded {len(self.feature_names)} feature names")
            
            # Load metadata
            with open(self.config.MODEL_METADATA_PATH, 'rb') as f:
                self.metadata = pickle.load(f)
            
            self.sequence_length = self.metadata.get('sequence_length', 60)
            logger.info(f"Model sequence length: {self.sequence_length}")
            
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}")
            raise
    
    def f1_score_metric(self, y_true, y_pred):
        """Custom F1 score metric for model loading"""
        precision = keras.metrics.Precision()(y_true, y_pred)
        recall = keras.metrics.Recall()(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + keras.backend.epsilon()))
    
    def predict_with_uncertainty(self, sequence, iterations=None):
        """
        Predict using Monte Carlo dropout for uncertainty estimation
        Returns: (prediction, confidence, uncertainty)
        """
        if iterations is None:
            iterations = self.config.MC_ITERATIONS
        
        if not self.config.USE_MONTE_CARLO or not self.model.trainable:
            # Standard prediction
            predictions = self.model.predict(sequence, verbose=0)
            confidence = np.max(predictions, axis=1)
            uncertainty = np.zeros_like(confidence)
            return predictions, confidence, uncertainty
        
        # Monte Carlo dropout
        predictions = []
        for _ in range(iterations):
            # Enable dropout at inference time
            pred = self.model(sequence, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.stack(predictions)  # Shape: (iterations, batch, classes)
        
        # Mean prediction
        mean_prediction = np.mean(predictions, axis=0)
        
        # Confidence (max probability)
        confidence = np.max(mean_prediction, axis=1)
        
        # Uncertainty (variance of predictions)
        uncertainty = np.var(predictions, axis=0)
        avg_uncertainty = np.mean(np.max(uncertainty, axis=1))
        
        return mean_prediction, confidence, avg_uncertainty
    
    def decode_prediction(self, prediction_probs):
        """
        Convert prediction probabilities to trading signal
        0: Strong SELL, 1: Weak SELL, 2: HOLD, 3: Weak BUY, 4: Strong BUY
        Returns: (-2, -1, 0, 1, 2)
        """
        # Get class with highest probability
        predicted_class = np.argmax(prediction_probs, axis=1)
        
        # Map to trading signals
        signal_map = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
        signals = np.array([signal_map[c] for c in predicted_class])
        
        return signals

# ==================== ADVANCED FEATURE ENGINEERING ====================
class AdvancedFeatureEngineer:
    """Creates deep learning optimized features from MT5 data"""
    
    def _init_(self, config):
        self.config = config
        
    def create_single_timeframe_features(self, df, timeframe_name=""):
        """Create comprehensive features from a single timeframe"""
        df_features = df.copy()
        
        # Fix column names - MT5 often has spaces
        df_features.columns = df_features.columns.str.strip()
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df_features.columns:
                logger.error(f"Missing required column: {col}")
                return None
        
        # 1. BASIC PRICE FEATURES
        df_features['returns'] = df_features['close'].pct_change()
        df_features['log_returns'] = np.log(df_features['close'] / df_features['close'].shift(1))
        df_features['range'] = (df_features['high'] - df_features['low']) / df_features['close']
        df_features['body'] = abs(df_features['close'] - df_features['open']) / df_features['close']
        df_features['high_low_ratio'] = df_features['high'] / df_features['low']
        df_features['close_open_ratio'] = df_features['close'] / df_features['open']
        
        # 2. VOLATILITY FEATURES
        returns = df_features['returns'].fillna(0)
        df_features['volatility_5'] = returns.rolling(5).std()
        df_features['volatility_10'] = returns.rolling(10).std()
        df_features['volatility_20'] = returns.rolling(20).std()
        df_features['volatility_ratio'] = df_features['volatility_5'] / df_features['volatility_20'].replace(0, 1e-10)
        
        # 3. TREND FEATURES
        df_features['ema_8'] = df_features['close'].ewm(span=8, adjust=False).mean()
        df_features['ema_21'] = df_features['close'].ewm(span=21, adjust=False).mean()
        df_features['ema_55'] = df_features['close'].ewm(span=55, adjust=False).mean()
        df_features['ema_ratio'] = df_features['ema_8'] / df_features['ema_55']
        
        # Moving averages
        df_features['sma_20'] = df_features['close'].rolling(20).mean()
        df_features['sma_50'] = df_features['close'].rolling(50).mean()
        df_features['ma_distance'] = (df_features['close'] - df_features['sma_20']) / df_features['sma_20']
        
        # 4. MOMENTUM INDICATORS
        # RSI
        delta = df_features['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df_features['rsi'] = 100 - (100 / (1 + rs))
        
        # Stochastic
        low_14 = df_features['low'].rolling(14).min()
        high_14 = df_features['high'].rolling(14).max()
        df_features['stoch_k'] = 100 * (df_features['close'] - low_14) / (high_14 - low_14)
        df_features['stoch_d'] = df_features['stoch_k'].rolling(3).mean()
        
        # MACD
        exp1 = df_features['close'].ewm(span=12, adjust=False).mean()
        exp2 = df_features['close'].ewm(span=26, adjust=False).mean()
        df_features['macd'] = exp1 - exp2
        df_features['macd_signal'] = df_features['macd'].ewm(span=9, adjust=False).mean()
        df_features['macd_hist'] = df_features['macd'] - df_features['macd_signal']
        
        # 5. SUPPORT/RESISTANCE FEATURES
        df_features['atr'] = self.calculate_atr(df_features, period=14)
        df_features['bb_upper'] = df_features['sma_20'] + 2 * df_features['close'].rolling(20).std()
        df_features['bb_lower'] = df_features['sma_20'] - 2 * df_features['close'].rolling(20).std()
        df_features['bb_position'] = (df_features['close'] - df_features['bb_lower']) / (df_features['bb_upper'] - df_features['bb_lower'])
        
        # 6. STATISTICAL FEATURES
        df_features['skewness_20'] = df_features['returns'].rolling(20).skew()
        df_features['kurtosis_20'] = df_features['returns'].rolling(20).kurt()
        df_features['z_score'] = (df_features['close'] - df_features['sma_20']) / df_features['close'].rolling(20).std()
        
        # 7. TIME-BASED FEATURES
        if 'time' in df_features.columns:
            df_features['time'] = pd.to_datetime(df_features['time'], unit='s')
            df_features['hour'] = df_features['time'].dt.hour
            df_features['minute'] = df_features['time'].dt.minute
            df_features['day_of_week'] = df_features['time'].dt.dayofweek
            
            # Market session indicators
            df_features['asian_session'] = ((df_features['hour'] >= 0) & (df_features['hour'] < 8)).astype(int)
            df_features['london_session'] = ((df_features['hour'] >= 8) & (df_features['hour'] < 16)).astype(int)
            df_features['ny_session'] = ((df_features['hour'] >= 13) & (df_features['hour'] < 21)).astype(int)
        
        # 8. VOLUME FEATURES - FIXED FOR MT5
        if 'Volume ' in df_features.columns or 'volume' in df_features.columns:
            # Handle both possible column names
            vol_col = 'Volume ' if 'Volume ' in df_features.columns else 'volume'
            if vol_col == 'Volume ':
                # Strip extra spaces in values
                df_features[vol_col] = pd.to_numeric(df_features[vol_col].astype(str).str.strip(), errors='coerce')
            
            df_features['volume_ratio'] = df_features[vol_col] / df_features[vol_col].rolling(20).mean()
            df_features['volume_zscore'] = (df_features[vol_col] - df_features[vol_col].rolling(20).mean()) / df_features[vol_col].rolling(20).std()
            df_features['volume_price_corr'] = df_features['close'].rolling(20).corr(df_features[vol_col].fillna(0))
        
        # 9. BID/ASK FEATURES (if available in live data)
        if self.config.INCLUDE_MICROSTRUCTURE:
            # Check for bid/ask columns
            bid_ask_cols = [c for c in df_features.columns if 'bid' in c.lower() or 'ask' in c.lower()]
            if len(bid_ask_cols) >= 2:
                # Find bid and ask columns
                bid_col = next((c for c in df_features.columns if 'bid' in c.lower()), None)
                ask_col = next((c for c in df_features.columns if 'ask' in c.lower()), None)
                
                if bid_col and ask_col:
                    df_features['spread'] = df_features[ask_col] - df_features[bid_col]
                    df_features['spread_ratio'] = df_features['spread'] / df_features['close']
                    df_features['mid_price'] = (df_features[ask_col] + df_features[bid_col]) / 2
        
        # Add timeframe prefix if specified
        if timeframe_name:
            df_features = df_features.add_prefix(f"{timeframe_name}_")
        
        # Drop NaN values
        df_features = df_features.dropna()
        
        logger.debug(f"Created features: {df_features.shape}")
        return df_features
    
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        return atr
    
    def create_multi_timeframe_features(self, symbol, timeframes=None):
        """Combine features from multiple timeframes"""
        if timeframes is None:
            timeframes = ['M5', 'M15', 'H1']  # Optimal timeframes for USD/JPY
        
        all_features = []
        
        for tf_name in timeframes:
            if tf_name not in self.config.TIMEFRAMES:
                logger.warning(f"Timeframe {tf_name} not in config")
                continue
            
            # Fetch data for this timeframe
            df = fetch_candles_simple(symbol, self.config.TIMEFRAMES[tf_name], 
                                     self.config.CANDLES_FETCH)
            
            if df is not None and len(df) > 100:  # Ensure enough data
                # Create features
                df_features = self.create_single_timeframe_features(df, tf_name)
                if df_features is not None:
                    # Take only the latest row that has all features
                    latest_features = df_features.iloc[-1:]
                    all_features.append(latest_features)
        
        if not all_features:
            logger.error(f"No features generated for {symbol}")
            return None
        
        # Combine all timeframe features
        combined_features = pd.concat(all_features, axis=1)
        
        # Fill NaN values
        combined_features = combined_features.ffill().bfill()
        
        logger.info(f"Multi-timeframe features created: {combined_features.shape}")
        return combined_features
    
    def prepare_dl_sequence(self, df_features, sequence_length, feature_names):
        """Prepare data for deep learning model input"""
        try:
            # First, ensure all columns are properly named (strip spaces)
            df_features.columns = df_features.columns.str.strip()
            
            # Clean feature names (strip spaces)
            clean_feature_names = [f.strip() for f in feature_names]
            
            # Find available features
            available_features = [f for f in clean_feature_names if f in df_features.columns]
            
            missing_features = set(clean_feature_names) - set(available_features)
            if missing_features:
                logger.warning(f"Missing {len(missing_features)} features: {list(missing_features)[:5]}...")
            
            if len(available_features) < len(clean_feature_names) * 0.7:  # 70% threshold
                logger.warning(f"Only {len(available_features)}/{len(clean_feature_names)} features available")
                # Try to use what we have
                if len(available_features) < 10:
                    return None
            
            # Extract feature values
            feature_data = df_features[available_features].values
            
            # Ensure we have enough data for sequence
            if len(feature_data) < sequence_length:
                logger.warning(f"Insufficient data: {len(feature_data)} < {sequence_length}")
                return None
            
            # Take the most recent sequence
            sequence = feature_data[-sequence_length:]
            
            # Reshape for model input: (1, sequence_length, n_features)
            sequence = sequence.reshape(1, sequence_length, -1)
            
            logger.debug(f"Sequence prepared: {sequence.shape}")
            return sequence
            
        except Exception as e:
            logger.error(f"Error preparing sequence: {e}", exc_info=True)
            return None

# ==================== MT5 DATA COLLECTION ====================
def initialize_mt5():
    """Initialize MT5 connection"""
    if not mt5.initialize():
        logger.error("MT5 initialization failed")
        return False
    
    logger.info("MT5 initialized successfully")
    return True

def fetch_candles_simple(symbol, timeframe, n_candles):
    """Fetch candles from MT5 with error handling"""
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
        if rates is None or len(rates) == 0:
            logger.warning(f"No data for {symbol} @ {timeframe}")
            return None
        
        df = pd.DataFrame(rates)
        
        # Strip column names to handle MT5 quirks
        df.columns = df.columns.str.strip()
        
        # Ensure proper column names
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Fix volume column if it exists
        if 'Volume ' in df.columns:
            # Convert to numeric, handling any issues
            df['Volume '] = pd.to_numeric(df['Volume '].astype(str).str.strip(), errors='coerce')
        
        logger.debug(f"Fetched {len(df)} candles for {symbol} @ {timeframe}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching candles for {symbol}: {e}")
        return None

# ==================== SIGNAL GENERATION ====================
class SignalGenerator:
    """Generates trading signals using deep learning model"""
    
    def _init_(self, model_handler, feature_engineer, config):
        self.model_handler = model_handler
        self.feature_engineer = feature_engineer
        self.config = config
    def generate_signal(self, symbol, current_price=None):
    """Generate trading signal with confidence and uncertainty"""
    logger.info(f"Generating signal for {symbol}")
    
    # TEMPORARY: Force test signals until model is trained
    import random
    signal = random.choice([-1, 1])  # 50% buy, 50% sell (no 0 for testing)
    confidence = random.uniform(0.6, 0.9)
    
    logger.info(f"TEST SIGNAL: {symbol} -> {signal} (confidence: {confidence:.2f})")
    
    details = {
        'signal': signal,
        'confidence': confidence,
        'uncertainty': 0.1,
        'method': 'random_test',
        'timestamp': datetime.now().isoformat()
    }
    
    return signal, confidence, 0.1, details
    
    def generate_signal(self, symbol, current_price=None):
        """Generate trading signal with confidence and uncertainty"""
        logger.info(f"Generating signal for {symbol}")
        
        try:
            # 1. Fetch and prepare data
            if self.config.USE_MULTI_TIMEFRAME:
                df_features = self.feature_engineer.create_multi_timeframe_features(symbol)
            else:
                # Single timeframe (H1)
                df = fetch_candles_simple(symbol, mt5.TIMEFRAME_H1, self.config.CANDLES_FETCH)
                if df is None:
                    return 0, 0.0, 0.0, {}
                
                df_features = self.feature_engineer.create_single_timeframe_features(df)
            
            if df_features is None or len(df_features) == 0:
                logger.warning("No features generated")
                return 0, 0.0, 0.0, {}
            
            # 2. Prepare DL sequence
            sequence = self.feature_engineer.prepare_dl_sequence(
                df_features,
                self.model_handler.sequence_length,
                self.model_handler.feature_names
            )
            
            if sequence is None:
                return 0, 0.0, 0.0, {}
            
            # 3. Scale the sequence
            if self.model_handler.scaler is not None:
                original_shape = sequence.shape
                sequence_2d = sequence.reshape(-1, sequence.shape[-1])
                sequence_scaled = self.model_handler.scaler.transform(sequence_2d)
                sequence = sequence_scaled.reshape(original_shape)
            
            # 4. Generate prediction with uncertainty
            predictions, confidence, uncertainty = self.model_handler.predict_with_uncertainty(sequence)
            
            # 5. Decode prediction to signal
            signals = self.model_handler.decode_prediction(predictions)
            signal = int(signals[0])  # Get first (and only) prediction
            
            # 6. Apply confidence and uncertainty filters
            if confidence[0] < self.config.SIGNAL_CONFIDENCE_THRESHOLD:
                logger.info(f"Low confidence: {confidence[0]:.3f} < {self.config.SIGNAL_CONFIDENCE_THRESHOLD}")
                signal = 0
            
            if uncertainty > self.config.UNCERTAINTY_THRESHOLD:
                logger.info(f"High uncertainty: {uncertainty:.3f} > {self.config.UNCERTAINTY_THRESHOLD}")
                signal = 0
            
            # 7. Additional market condition checks
            if not self.check_market_conditions(df_features, signal):
                signal = 0
            
            # 8. Prepare signal details
            signal_details = {
                'signal': signal,
                'confidence': float(confidence[0]),
                'uncertainty': float(uncertainty),
                'prediction_probs': predictions[0].tolist(),
                'signal_strength': abs(signal),
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'sequence_shape': sequence.shape
            }
            
            logger.info(f"Signal: {signal}, Confidence: {confidence[0]:.3f}, Uncertainty: {uncertainty:.3f}")
            
            return signal, confidence[0], uncertainty, signal_details
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}", exc_info=True)
            return 0, 0.0, 0.0, {}
    
    def check_market_conditions(self, df_features, signal):
        """Check if market conditions are suitable for trading"""
        try:
            # Strip column names
            df_features.columns = df_features.columns.str.strip()
            
            # 1. Check volatility
            vol_cols = [c for c in df_features.columns if 'volatility' in c]
            if vol_cols:
                volatility = df_features[vol_cols[0]].iloc[-1]
                if volatility < self.config.MIN_VOLATILITY_REQUIRED:
                    logger.info(f"Low volatility: {volatility:.6f}")
                    return False
            
            # 2. Check if market is trending strongly against signal
            trend_cols = [c for c in df_features.columns if any(x in c for x in ['ema_ratio', 'ma_distance', 'z_score'])]
            if trend_cols and signal != 0:
                trend_strength = abs(df_features[trend_cols[0]].iloc[-1])
                
                # Check if signal goes against strong trend
                if trend_strength > 1.5:  # Strong trend
                    # Determine trend direction
                    trend_direction = 1 if df_features[trend_cols[0]].iloc[-1] > 0 else -1
                    
                    if (signal > 0 and trend_direction < 0) or (signal < 0 and trend_direction > 0):
                        logger.info(f"Signal against strong trend: signal={signal}, trend={trend_direction}")
                        return False
            
            # 3. Check if market is at extreme (overbought/oversold)
            if 'rsi' in df_features.columns:
                rsi = df_features['rsi'].iloc[-1]
                if (signal > 0 and rsi > 70) or (signal < 0 and rsi < 30):
                    logger.info(f"Signal at RSI extreme: signal={signal}, RSI={rsi:.1f}")
                    return False
            
            # 4. Check volume (if available)
            vol_cols = [c for c in df_features.columns if 'volume' in c.lower()]
            if vol_cols and 'volume_ratio' in df_features.columns:
                volume_ratio = df_features['volume_ratio'].iloc[-1]
                if volume_ratio < 0.5:  # Very low volume
                    logger.info(f"Low volume: ratio={volume_ratio:.2f}")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Market condition check failed: {e}")
            return True  # Default to allowing trade if check fails

# ==================== MAIN PIPELINE CLASS ====================
class Portion1Pipeline:
    """Main pipeline for Portion1 - Data & Signal Generation"""
    
    def _init_(self, config_class=Portion1Config):
        self.config = config_class()
        
        # Initialize components
        self.mt5_initialized = False
        self.model_handler = None
        self.feature_engineer = None
        self.signal_generator = None
        
        self.initialize_pipeline()
    
    def initialize_pipeline(self):
        """Initialize all pipeline components"""
        logger.info("Initializing Portion1 Pipeline")
        
        try:
            # Initialize MT5
            self.mt5_initialized = initialize_mt5()
            if not self.mt5_initialized:
                raise Exception("MT5 initialization failed")
            
            # Initialize Deep Learning components
            if self.config.MODEL_TYPE == "DEEP_LEARNING":
                self.model_handler = DLModelHandler(self.config)
            
            # Initialize feature engineer
            self.feature_engineer = AdvancedFeatureEngineer(self.config)
            
            # Initialize signal generator
            self.signal_generator = SignalGenerator(
                self.model_handler, 
                self.feature_engineer, 
                self.config
            )
            
            logger.info("Portion1 Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            raise
    
    def run_for_symbol(self, symbol):
        """Run the pipeline for a single symbol"""
        if not self.mt5_initialized or self.signal_generator is None:
            logger.error("Pipeline not properly initialized")
            return 0, 0.0, 0.0, {}
        
        try:
            # Generate signal
            signal, confidence, uncertainty, details = self.signal_generator.generate_signal(symbol)
            
            # Log signal details
            self.log_signal(symbol, signal, confidence, uncertainty, details)
            
            return signal, confidence, uncertainty, details
            
        except Exception as e:
            logger.error(f"Error running pipeline for {symbol}: {e}")
            return 0, 0.0, 0.0, {}
    
    def log_signal(self, symbol, signal, confidence, uncertainty, details):
        """Log signal to file and console"""
        signal_map = {
            -2: "STRONG_SELL", -1: "SELL", 0: "HOLD", 1: "BUY", 2: "STRONG_BUY"
        }
        
        signal_text = signal_map.get(signal, "UNKNOWN")
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'signal': signal,
            'signal_text': signal_text,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'details': details
        }
        
        # Save to JSON log file
        import json
        try:
            with open('logs/signals.jsonl', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except:
            pass
        
        # Console log
        if signal != 0:
            logger.info(f"════════════════════════════════════════════════════")
            logger.info(f"TRADING SIGNAL: {signal_text} for {symbol}")
            logger.info(f"Confidence: {confidence:.3f}, Uncertainty: {uncertainty:.3f}")
            
            if 'prediction_probs' in details:
                probs = details['prediction_probs']
                logger.info(f"Probabilities: Strong_SELL={probs[0]:.3f}, SELL={probs[1]:.3f}, "
                           f"HOLD={probs[2]:.3f}, BUY={probs[3]:.3f}, Strong_BUY={probs[4]:.3f}")
            
            logger.info(f"════════════════════════════════════════════════════")
    
    def shutdown(self):
        """Shutdown pipeline and cleanup"""
        logger.info("Shutting down Portion1 Pipeline")
        
        if self.mt5_initialized:
            mt5.shutdown()
            logger.info("MT5 shutdown complete")

# ==================== LEGACY COMPATIBILITY FUNCTIONS ====================
# These functions maintain compatibility with your existing Phase6 pipeline

def fetch_multi_timeframe_features(symbol):
    """
    Legacy function for compatibility with existing pipeline
    Returns DataFrame with latest features
    """
    # Create pipeline instance if not exists
    if 'pipeline' not in globals():
        globals()['pipeline'] = Portion1Pipeline()
    
    pipeline = globals()['pipeline']
    
    if not pipeline.mt5_initialized or pipeline.feature_engineer is None:
        logger.error("Pipeline not initialized")
        return None
    
    try:
        # Fetch features using the new system
        df_features = pipeline.feature_engineer.create_multi_timeframe_features(symbol)
        return df_features
        
    except Exception as e:
        logger.error(f"Error fetching features: {e}")
        return None

def generate_signal(model, features, feature_cols):
    """
    Legacy function for compatibility
    Note: 'model' parameter is ignored in DL mode
    """
    # Create pipeline instance if not exists
    if 'pipeline' not in globals():
        globals()['pipeline'] = Portion1Pipeline()
    
    pipeline = globals()['pipeline']
    
    if not pipeline.mt5_initialized or pipeline.signal_generator is None:
        logger.error("Pipeline not initialized")
        return 0
    
    try:
        # Use the first symbol from config
        symbol = pipeline.config.SYMBOLS[0] if pipeline.config.SYMBOLS else "USDJPYm"
        
        # Generate signal using new system
        signal, confidence, uncertainty, details = pipeline.signal_generator.generate_signal(symbol)
        
        # Return legacy format signal (-1, 0, 1)
        if signal == 2:  # Strong BUY
            return 1
        elif signal == -2:  # Strong SELL
            return -1
        else:
            return signal  # Already -1, 0, or 1
        
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        return 0

def load_trading_model(path=None):
    """
    Legacy function - returns the DL model handler
    """
    try:
        # Return a dummy model object with predict method for compatibility
        class LegacyModelWrapper:
            def _init_(self, pipeline):
                self.pipeline = pipeline
                self.classes_ = [0, 1, 2]  # For compatibility
            
            def predict(self, X):
                # This won't be used in DL mode, but provided for compatibility
                return np.array([0])
            
            def predict_proba(self, X):
                # Return dummy probabilities for compatibility
                return np.array([[0.33, 0.34, 0.33]])
        
        if 'pipeline' not in globals():
            globals()['pipeline'] = Portion1Pipeline()
        
        return LegacyModelWrapper(globals()['pipeline'])
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

# ==================== UTILITY FUNCTIONS ====================
def get_mt5_column_names(symbol, timeframe=mt5.TIMEFRAME_M1, n_candles=10):
    """Debug function to see MT5 column names"""
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
        if rates is None:
            return None
        
        df = pd.DataFrame(rates)
        print(f"MT5 columns for {symbol}: {df.columns.tolist()}")
        print(f"Column details: {[(col, type(df[col].iloc[0]) if len(df) > 0 else 'empty') for col in df.columns]}")
        
        # Check volume column specifically
        if 'Volume ' in df.columns:
            print(f"Volume column sample: {df['Volume '].head()}")
            print(f"Volume dtype: {df['Volume '].dtype}")
        
        return df.columns.tolist()
        
    except Exception as e:
        print(f"Error getting column names: {e}")
        return None

# ==================== MAIN EXECUTION ====================
def main():
    """Main execution for testing Portion1 standalone"""
    logger.info("Starting Portion1 - Advanced Data & Signal Generation")
    
    try:
        # Debug: Check MT5 column names
        print("\nDebug: Checking MT5 column names...")
        cols = get_mt5_column_names("USDJPYm", mt5.TIMEFRAME_M1, 5)
        
        # Initialize pipeline
        pipeline = Portion1Pipeline()
        
        # Test with each symbol
        for symbol in pipeline.config.SYMBOLS:
            logger.info(f"Processing {symbol}")
            
            signal, confidence, uncertainty, details = pipeline.run_for_symbol(symbol)
            
            # Display results
            if signal != 0:
                signal_text = {2: "STRONG BUY", 1: "BUY", -1: "SELL", -2: "STRONG SELL"}.get(signal, "HOLD")
                print(f"\n{'='*60}")
                print(f"Signal for {symbol}: {signal_text}")
                print(f"Confidence: {confidence:.3%}")
                print(f"Uncertainty: {uncertainty:.3%}")
                
                if 'prediction_probs' in details:
                    probs = details['prediction_probs']
                    print(f"Probabilities:")
                    print(f"  Strong SELL: {probs[0]:.3%}")
                    print(f"  SELL: {probs[1]:.3%}")
                    print(f"  HOLD: {probs[2]:.3%}")
                    print(f"  BUY: {probs[3]:.3%}")
                    print(f"  Strong BUY: {probs[4]:.3%}")
                
                print(f"{'='*60}\n")
            else:
                print(f"\nNo signal for {symbol} (HOLD)")
                print(f"Confidence: {confidence:.3%}, Uncertainty: {uncertainty:.3%}\n")
        
        # Shutdown
        pipeline.shutdown()
        
        logger.info("Portion1 execution complete")
        
    except Exception as e:
        logger.error(f"Portion1 execution failed: {e}", exc_info=True)

# ==================== QUICK TEST FUNCTION ====================
def quick_test():
    """Quick test of the Portion1 pipeline"""
    print("Quick testing Portion1...")
    
    try:
        # Use minimal config for quick test
        class QuickConfig(Portion1Config):
            CANDLES_FETCH = 100
            USE_MULTI_TIMEFRAME = False
            SYMBOLS = ["USDJPYm"]
            LOG_LEVEL = logging.WARNING  # Less verbose
        
        print("Initializing pipeline...")
        pipeline = Portion1Pipeline(QuickConfig)
        
        print("Generating signal...")
        signal, confidence, uncertainty, _ = pipeline.run_for_symbol("USDJPYm")
        
        print(f"\nResult:")
        print(f"Signal: {signal}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Uncertainty: {uncertainty:.3f}")
        
        pipeline.shutdown()
        
    except Exception as e:
        print(f"Quick test failed: {e}")
        import traceback
        traceback.print_exc()

# ==================== EXPORT FOR PHASE6 ====================
def get_phase6_interface():
    """
    Returns interface functions for Phase6 integration
    """
    return {
        'fetch_multi_timeframe_features': fetch_multi_timeframe_features,
        'generate_signal': generate_signal,
        'load_trading_model': load_trading_model,
        'Portion1Pipeline': Portion1Pipeline,
        'get_mt5_column_names': get_mt5_column_names  # For debugging
    }

# ==================== EXECUTION ====================
if _name_ == "_main_":
    import argparse
    
    parser = argparse.ArgumentParser(description='Portion1 - Advanced Data & Signal Generation')
    parser.add_argument('--mode', type=str, default='test', 
                       choices=['test', 'quick', 'debug', 'export'],
                       help='Execution mode')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        main()
    elif args.mode == 'quick':
        quick_test()
    elif args.mode == 'debug':
        # Just check MT5 connection and columns
        if initialize_mt5():
            cols = get_mt5_column_names("USDJPYm", mt5.TIMEFRAME_M1, 10)
            mt5.shutdown()
    elif args.mode == 'export':
        interface = get_phase6_interface()
        print("Portion1 interface ready for Phase6 integration")
    
    print("Portion1 execution complete!")