import os
import sys
import time
import math
import numpy as np
import pandas as pd
import json
import asyncio
from binance import AsyncClient
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.volatility import AverageTrueRange
from dotenv import load_dotenv
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_fixed
from cryptography.fernet import Fernet



load_dotenv()

# Validate critical environment variables
FERNET_KEY = os.getenv("ENCRYPTION_KEY")
if not FERNET_KEY:
    print("ERROR: ENCRYPTION_KEY not set in environment variables!")
    print("Please set all required environment variables before running the bot.")
    exit(1)

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
if not API_KEY or not API_SECRET:
    print("ERROR: BINANCE_API_KEY and BINANCE_API_SECRET must be set!")
    print("Please configure your Binance API credentials.")
    exit(1)

try:
    fernet = Fernet(FERNET_KEY.encode())
except Exception as e:
    print(f"ERROR: Invalid ENCRYPTION_KEY format: {e}")
    exit(1)

client = None  

#BITNO LOKALNO RUNNATI S VENVOM ALI KOD SERVER DOCKER JER JE SAM PO SEBI
#IZOLIRANI PA NEMA POTREBE NA VENV 
#DOCKERFILE SVE RJESAVA SAMO TREBA APPENDATI REAL WORKING DIRECTORY
#INCAE NA SVAKI CRON RESET NE SEJVAJU SE FILEOVI JER SU U KONTENJERU A NE NA SERVERU

symbol = "BTCUSDC"
interval = AsyncClient.KLINE_INTERVAL_1MINUTE
# Trade with $100 worth of BTC per trade (quantity will be calculated dynamically)
# Legacy fixed trade size removed; bot now uses dynamic 100% available balance sizing
quantity = None  # Will be set dynamically before each trade
lookback = 150
k_neighbors = 5  # Kolko K bude KNN uzel više K više procesing powera
stop_loss = 0.006  # 0.6% stop loss (1:2 risk/reward ratio)
log_file = "trades.log"
position_file = "position_state.json"
daily_stats_file = "daily_stats.json"
paper_trading = False  # False za live trade

# Balance usage buffers
ENTRY_QUOTE_BUFFER = 0.999   # 99.9% of quote (USDC) when entering
EXIT_ASSET_BUFFER = 0.999    # 99.9% of asset when exiting

# Stale position handling (force exit if stagnant)
stale_exit_minutes = 90            # minutes before stale exit considered
stale_exit_max_abs_change = 0.0015 # 0.15% band regarded as stagnant

# Daily trading limits
max_trades_per_day = 10  # Maximum number of trades per day (better for 6% target)
max_daily_loss = 0.065  # Maximum daily loss (6.5% - safety buffer for 10 trades)
daily_profit_target = 0.06  # Daily profit target (6% - much more achievable now)
trade_cooldown_minutes = 20  # Reduced to 20 minutes (allows 10 trades in 3+ hours)

# File for accumulating all historical klines
historical_klines_file = "historical_klines.csv"




def append_klines_to_csv(df, csv_file=historical_klines_file):
    """Append only newest kline (by 'time') preventing duplicates."""
    if df.empty:
        return
    last_row = df.tail(1)
    new_time = str(last_row.iloc[0]["time"])
    if not os.path.exists(csv_file):
        last_row.to_csv(csv_file, index=False)
        return
    try:
        from collections import deque
        with open(csv_file, "r") as f:
            last_line = deque(f, 1)[0]
        existing_time = last_line.split(",", 1)[0]
        if existing_time != new_time:
            last_row.to_csv(csv_file, mode="a", header=False, index=False)
    except Exception as e:
        print(f"[WARN] Dedup append failed ({e}), appending anyway")
        last_row.to_csv(csv_file, mode="a", header=False, index=False)


def deduplicate_historical_klines(csv_file=historical_klines_file):
    if not os.path.exists(csv_file):
        return
    try:
        df = pd.read_csv(csv_file)
        before = len(df)
        df = df.drop_duplicates(subset=["time"], keep="last")
        after = len(df)
        if after < before:
            df.to_csv(csv_file, index=False)
            print(f"[DEDUP] Removed {before-after} duplicate klines (now {after}).")
    except Exception as e:
        print(f"[DEDUP ERROR] {e}")


def load_full_training_data(csv_file=historical_klines_file):
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)
    else:
        return None


def save_position_state(entry_price, side, timestamp, btc_quantity=None):
    """Save current position state to file"""
    position_data = {
        "entry_price": entry_price,
        "side": side,
        "timestamp": timestamp,
        "active": True,
        "btc_quantity": btc_quantity or quantity  # Save actual BTC quantity
    }
    try:
        with open(position_file, "w") as f:
            json.dump(position_data, f)
        print(f"[POSITION SAVED] {side} at {entry_price} (Qty: {position_data['btc_quantity']:.6f} BTC)")
    except Exception as e:
        print(f"Failed to save position state: {e}")


def load_position_state():
    """Load position state from file"""
    try:
        if os.path.exists(position_file):
            with open(position_file, "r") as f:
                position_data = json.load(f)
            if position_data.get("active", False):
                print(
                    f"[POSITION LOADED] {position_data['side']} at {position_data['entry_price']}"
                )
                return position_data
    except Exception as e:
        print(f"Failed to load position state: {e}")
    return None


def clear_position_state():
    """Clear position state when position is closed"""
    try:
        if os.path.exists(position_file):
            #ideja je da bude neaktivan da se ne provjeri audit trail
            with open(position_file, "r") as f:
                position_data = json.load(f)
            position_data["active"] = False
            position_data["closed_timestamp"] = datetime.now().isoformat()
            with open(position_file, "w") as f:
                json.dump(position_data, f)
        print("[POSITION CLEARED]")
    except Exception as e:
        print(f"Failed to clear position state: {e}")


def get_daily_stats():
    """Get or create daily trading statistics"""
    today = datetime.now().strftime("%Y-%m-%d")

    try:
        if os.path.exists(daily_stats_file):
            with open(daily_stats_file, "r") as f:
                stats = json.load(f)

            if stats.get("date") != today:
                stats = create_new_daily_stats(today)
        else:
            stats = create_new_daily_stats(today)

        return stats
    except Exception as e:
        print(f"Error loading daily stats: {e}")
        return create_new_daily_stats(today)


def create_new_daily_stats(date):
    """Create new daily statistics structure"""
    return {
        "date": date,
        "trades_count": 0,
    # Absolute PnL in quote currency (USDC)
    "total_pnl_abs": 0.0,
    # PnL percentage relative to starting equity of the day
    "total_pnl_pct": 0.0,
        "wins": 0,
        "losses": 0,
        "last_trade_time": None,
        "daily_limit_hit": False,
        "profit_target_hit": False,
    # Equity tracking for correct % thresholds
    "starting_equity": None,
    "last_equity": None,
    }


def save_daily_stats(stats):
    """Save daily statistics to file"""
    try:
        with open(daily_stats_file, "w") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"Error saving daily stats: {e}")


def format_pnl_text(stats):
    """Return human-friendly PnL string based on available fields."""
    if stats.get("starting_equity") and stats.get("total_pnl_pct") is not None:
        return f"{stats['total_pnl_pct']:.2f}% (abs ${stats.get('total_pnl_abs', 0.0):.2f})"
    # legacy fallback
    if "total_pnl" in stats:
        try:
            return f"${stats['total_pnl']:.2f}"
        except Exception:
            pass
    return f"${stats.get('total_pnl_abs', 0.0):.2f}"


def can_trade():
    """Check if trading is allowed based on daily limits"""
    stats = get_daily_stats()
    # If starting_equity exists, enforce % thresholds; else use absolute as fallback

    # Check daily trade limit
    if stats["trades_count"] >= max_trades_per_day:
        return False, "Daily trade limit reached"

    # Self-heal: if profit_target_hit is True but PnL% is below target, reset flag
    if stats.get("starting_equity") and stats.get("total_pnl_pct") is not None:
        if stats.get("profit_target_hit") and stats["total_pnl_pct"] < daily_profit_target * 100:
            stats["profit_target_hit"] = False
            save_daily_stats(stats)
    # Check % limits only if we have starting_equity
    if stats.get("starting_equity") and stats.get("total_pnl_pct") is not None:
        if stats["total_pnl_pct"] <= -max_daily_loss * 100:
            return False, "Daily loss limit reached"
        if stats.get("profit_target_hit"):
            return False, "Daily profit target achieved"
    else:
        # Fallback absolute if % not available yet
        if stats.get("total_pnl_abs", 0.0) <= -max_daily_loss:
            return False, "Daily loss limit reached"

    # Check cooldown period
    if stats["last_trade_time"]:
        last_trade = datetime.fromisoformat(stats["last_trade_time"])
        time_since_last = datetime.now() - last_trade
        if time_since_last.total_seconds() < (trade_cooldown_minutes * 60):
            remaining = (trade_cooldown_minutes * 60) - time_since_last.total_seconds()
            return False, f"Cooldown active ({remaining / 60:.1f} min remaining)"

    return True, "Trading allowed"


async def send_daily_summary():
    """Send daily trading summary to console"""
    stats = get_daily_stats()

    subject = f"[Trading Bot] Daily Summary - {stats['date']}"

    win_rate = (
        (stats["wins"] / max(stats["trades_count"], 1)) * 100
        if stats["trades_count"] > 0
        else 0
    )

    body = f"""
Daily Trading Summary for {stats["date"]}

📊 STATISTICS:
• Total Trades: {stats["trades_count"]}/{max_trades_per_day}
• Total PnL: {format_pnl_text(stats)}
• Wins: {stats["wins"]} | Losses: {stats["losses"]}
• Win Rate: {win_rate:.1f}%

🎯 LIMITS:
• Daily Loss Limit: {max_daily_loss:.1%}
• Daily Profit Target: {daily_profit_target:.1%}
• Max Trades: {max_trades_per_day}

🚦 STATUS:
• Daily Limit Hit: {"Yes" if stats["daily_limit_hit"] else "No"}
• Profit Target Hit: {"Yes" if stats["profit_target_hit"] else "No"}
• Last Trade: {stats["last_trade_time"] or "None"}

Bot Status: {"SUSPENDED" if stats["daily_limit_hit"] or stats["profit_target_hit"] else "ACTIVE"}
    """

    print("[DAILY SUMMARY]")
    print(body)



# Simple in-memory cache for klines
_klines_cache = None
_klines_cache_time = 0
_klines_cache_ttl = 30  # seconds

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
async def fetch_klines():
    global _klines_cache, _klines_cache_time, _klines_fetch_count
    
    if client is None:
        raise ValueError("Binance client not initialized")
        
    now = time.time()
    if _klines_cache is not None and (now - _klines_cache_time) < _klines_cache_ttl:
        return _klines_cache.copy()
    data = await client.get_klines(symbol=symbol, interval=interval, limit=lookback)
    df = pd.DataFrame(
        data,
        columns=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )
    df["close"] = df["close"].astype(float)
    df["returns"] = df["close"].pct_change().fillna(0)
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    df["ema_20"] = EMAIndicator(df["close"], window=20).ema_indicator()
    macd = MACD(df["close"])
    df["macd"] = macd.macd_diff()
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_bbm"] = bb.bollinger_mavg()
    df["bb_bbh"] = bb.bollinger_hband()
    df["bb_bbl"] = bb.bollinger_lband()
    stoch = StochasticOscillator(df["close"], df["high"].astype(float), df["low"].astype(float), window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    atr = AverageTrueRange(df["high"].astype(float), df["low"].astype(float), df["close"], window=14)
    df["atr"] = atr.average_true_range()

    df = df.dropna().reset_index(drop=True)
    append_klines_to_csv(df)
    try:
        _klines_fetch_count += 1
    except NameError:
        _klines_fetch_count = 1
    if _klines_fetch_count % 60 == 0:
        deduplicate_historical_klines()
    _klines_cache = df.copy()
    _klines_cache_time = now
    return df


def knn_predict(df):
    full_df = load_full_training_data()
    if full_df is not None and len(full_df) > 20:
        train_df = full_df.copy()
    else:
        train_df = df.copy()

    features = [
        "returns", "rsi", "ema_20", "macd",
        "bb_bbm", "bb_bbh", "bb_bbl", "stoch_k", "stoch_d", "atr",
        "volume", "quote_asset_volume", "number_of_trades", "trend_strength"
    ]
    # u biti ema20 trend strength
    train_df["trend_strength"] = train_df["ema_20"] - train_df["close"]
    if "trend_strength" not in df.columns:
        df["trend_strength"] = df["ema_20"] - df["close"]

    # samo skaliranje
    X = train_df[features].iloc[:-1].fillna(0)
    y = np.where(train_df["returns"].shift(-1).iloc[:-1] > 0, 1, 0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(X_scaled, y)

    # preciznst print
    acc = None
    if len(X_scaled) > 10:
        acc = (
            knn.score(X_scaled[-100:], y[-100:]) * 100
            if len(X_scaled) > 100
            else knn.score(X_scaled, y) * 100
        )

    latest_features = df[features].iloc[[-1]].fillna(0)
    latest_scaled = scaler.transform(latest_features)
    prediction = knn.predict(latest_scaled)[0]
    return prediction, acc


async def log_trade(action, price, pnl=None, reason=None, entry_price=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} | {action} | Price: {price:.6f}"
    if entry_price is not None:
        line += f" | Entry: {entry_price:.6f}"
    if pnl is not None:
        line += f" | PnL: {pnl:.4f}"
    if reason:
        line += f" | Reason: {reason}"
    encrypted_line = fernet.encrypt(line.encode()).decode()
    with open(log_file, "a") as f:
        f.write(encrypted_line + "\n")
    print(line)  # enkripcija


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
async def get_symbol_info(symbol="BTCUSDC"):
    """Get symbol information including lot size filters"""
    if client is None:
        raise ValueError("Binance client not initialized")
    
    try:
        exchange_info = await client.get_exchange_info()
        
        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol:
                # Extract relevant filters
                filters = {}
                for filter_info in symbol_info['filters']:
                    if filter_info['filterType'] == 'LOT_SIZE':
                        filters['lot_size'] = {
                            'min_qty': float(filter_info['minQty']),
                            'max_qty': float(filter_info['maxQty']),
                            'step_size': float(filter_info['stepSize'])
                        }
                    elif filter_info['filterType'] == 'MIN_NOTIONAL':
                        filters['min_notional'] = float(filter_info['minNotional'])
                    elif filter_info['filterType'] == 'MARKET_LOT_SIZE':
                        filters['market_lot_size'] = {
                            'min_qty': float(filter_info['minQty']),
                            'max_qty': float(filter_info['maxQty']),
                            'step_size': float(filter_info['stepSize'])
                        }
                
                return {
                    'symbol': symbol_info['symbol'],
                    'status': symbol_info['status'],
                    'base_asset': symbol_info['baseAsset'],
                    'quote_asset': symbol_info['quoteAsset'],
                    'base_precision': symbol_info['baseAssetPrecision'],
                    'quote_precision': symbol_info['quoteAssetPrecision'],
                    'filters': filters
                }
        
        raise ValueError(f"Symbol {symbol} not found")
    except Exception as e:
        print(f"Error getting symbol info: {e}")
        # Return default values if API fails
        return {
            'symbol': symbol,
            'filters': {
                'lot_size': {'min_qty': 0.00001, 'max_qty': 9000, 'step_size': 0.00001},
                'min_notional': 10.0
            }
        }


def format_quantity_for_binance(quantity, step_size=0.00001):
    """Format quantity according to Binance step size requirements"""
    try:
        # Derive precision robustly (handle scientific notation) by expanding to 16 decimals then trimming
        if step_size >= 1:
            precision = 0
        else:
            step_str = f"{step_size:.16f}".rstrip('0').rstrip('.')
            if '.' in step_str:
                precision = len(step_str.split('.')[1])
            else:
                precision = 0
        if precision == 0:
            # Integer step size
            adjusted_quantity = math.floor(quantity / step_size) * step_size if step_size > 0 else int(quantity)
            return str(int(adjusted_quantity))
        # Floor to step
        steps = math.floor(quantity / step_size)
        adjusted_quantity = steps * step_size
        # Format with derived precision
        formatted = f"{adjusted_quantity:.{precision}f}"
        # Guard: if formatting produced 0 but original quantity was positive and above one step, bump to one step
        if float(formatted) == 0 and quantity >= step_size:
            formatted = f"{step_size:.{precision}f}"
        # Trim trailing zeros while keeping at least one decimal digit if needed
        if '.' in formatted:
            trimmed = formatted.rstrip('0').rstrip('.')
            if trimmed in ('', '0'):
                return formatted  # keep original to avoid empty
            return trimmed
        return formatted
    except Exception as e:
        print(f"[WARNING] Error formatting quantity {quantity} with step {step_size}: {e}")
        # Fallback to 6 decimal places
        return f"{float(quantity):.6f}"

def normalize_quantity(symbol_info, raw_qty, price=None):
    """Normalize a raw quantity to satisfy LOT_SIZE (minQty/stepSize) and if price given, MIN_NOTIONAL.
    Returns tuple (normalized_qty, ok_flag, message). If not ok, normalized_qty will be 0.
    """
    try:
        filters = symbol_info.get('filters', {}) if symbol_info else {}
        lot = filters.get('lot_size', {}) or {}
        step = float(lot.get('step_size', 0.00001))
        min_qty = float(lot.get('min_qty', 0.00001))
        qty = float(raw_qty)
        if qty <= 0:
            return 0.0, False, 'non_positive'
        # floor to step
        steps = math.floor(qty / step)
        qty = steps * step
        # precision formatting based on step
        if step >= 1:
            qty = float(int(qty))
        else:
            precision = len(str(step).rstrip('0').split('.')[-1])
            qty = float(f"{qty:.{precision}f}")
        if qty < min_qty:
            return 0.0, False, f'below_minQty({qty}<{min_qty})'
        # MIN_NOTIONAL enforcement (best-effort)
        min_notional = filters.get('min_notional')
        if price is not None and min_notional is not None:
            try:
                min_notional_val = float(min_notional)
                notional = qty * price
                if notional < min_notional_val:
                    needed_qty = (min_notional_val / price) * 1.001
                    steps = math.floor(needed_qty / step)
                    needed_qty = steps * step
                    if step < 1:
                        precision = len(str(step).rstrip('0').split('.')[-1])
                        needed_qty = float(f"{needed_qty:.{precision}f}")
                    if needed_qty >= min_qty:
                        qty = needed_qty
            except Exception:
                pass
        return qty, True, 'ok'
    except Exception as e:
        return 0.0, False, f'normalize_error:{e}'


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
async def get_price():
    if client is None:
        raise ValueError("Binance client not initialized")
    ticker = await client.get_symbol_ticker(symbol=symbol)
    return float(ticker["price"])


async def get_account_balance():
    """Get USDC and BTC balances"""
    try:
        if client is None:
            raise ValueError("Binance client not initialized")
        account = await client.get_account()
        balances = {balance['asset']: float(balance['free']) for balance in account['balances']}
        usdc_balance = balances.get('USDC', 0.0)
        btc_balance = balances.get('BTC', 0.0)
        
        # Debug: Print all non-zero balances to help diagnose issues
        print(f"[BALANCE DEBUG] Spot wallet balances:")
        for asset, balance in balances.items():
            if balance > 0:
                print(f"  {asset}: {balance:.6f}")
        
        return usdc_balance, btc_balance
    except Exception as e:
        print(f"Error getting account balance: {e}")
        return 0.0, 0.0


async def get_equity_usdc():
    """Return total equity valued in USDC (USDC + BTC*price)."""
    usdc, btc = await get_account_balance()
    try:
        price = await get_price()
    except Exception:
        price = 0.0
    return usdc + btc * price


async def execute_buy_order(quantity, is_exit_order=False):
    """Execute a market buy order for BTC using USDC"""
    try:
        if client is None:
            raise ValueError("Binance client not initialized")
        
        # Get symbol info for proper formatting
        symbol_info = await get_symbol_info(symbol)
        step_size = symbol_info['filters'].get('lot_size', {}).get('step_size', 0.00001)
        min_qty = symbol_info['filters'].get('lot_size', {}).get('min_qty', 0.00001)
        
        if is_exit_order:
            current_price = await get_price()
            usdc_balance, btc_balance = await get_account_balance()
            available_usdc = usdc_balance * EXIT_ASSET_BUFFER
            requested_notional = quantity * current_price * 1.001
            if requested_notional > available_usdc:
                raw_qty = (available_usdc / current_price)
            else:
                raw_qty = quantity
            normalized_qty, ok, msg = normalize_quantity(symbol_info, raw_qty, price=current_price)
            if not ok:
                raise ValueError(f"Exit buy quantity invalid: {msg}")
            formatted_quantity = format_quantity_for_binance(normalized_qty, step_size)
            if not formatted_quantity or float(formatted_quantity) <= 0:
                raise ValueError(f"Formatted exit buy quantity invalid/empty: '{formatted_quantity}' from {normalized_qty}")
            print(f"[BUY EXIT] raw={raw_qty:.8f} norm={formatted_quantity} price={current_price:.2f} msg={msg}")
            order = await client.order_market_buy(symbol=symbol, quantity=formatted_quantity)
        else:
            # For entry orders, use ALL available USDC with quoteOrderQty (safer)
            usdc_balance, btc_balance = await get_account_balance()
            available_usdc = usdc_balance * ENTRY_QUOTE_BUFFER
            
            # Use quote order (buy $ amount worth of BTC) - this avoids LOT_SIZE issues
            trade_amount = int(available_usdc)  # Binance requires integer for quote qty
            
            if trade_amount < 10:  # Minimum trade
                raise ValueError(f"Trade amount too small: ${trade_amount} (minimum: $10)")
                
            print(f"[BUY ORDER] Using available USDC: ${available_usdc:.6f} (rounded to ${trade_amount})")
            order = await client.order_market_buy(
                symbol=symbol,
                quoteOrderQty=trade_amount  # This avoids LOT_SIZE issues
            )
        
        print(f"[BUY ORDER EXECUTED] Order ID: {order['orderId']}")
        return order
    except Exception as e:
        print(f"Error executing buy order: {e}")
        raise e


async def execute_sell_order(quantity):
    """Execute a market sell order for BTC to get USDC"""
    try:
        if client is None:
            raise ValueError("Binance client not initialized")
        symbol_info = await get_symbol_info(symbol)
        step_size = symbol_info['filters'].get('lot_size', {}).get('step_size', 0.00001)
        min_qty = symbol_info['filters'].get('lot_size', {}).get('min_qty', 0.00001)
        usdc_balance, btc_balance = await get_account_balance()
        max_sellable = btc_balance * EXIT_ASSET_BUFFER
        raw_qty = min(quantity, max_sellable)
        normalized_qty, ok, msg = normalize_quantity(symbol_info, raw_qty, price=await get_price())
        if not ok:
            raise ValueError(f"Sell quantity invalid: {msg}")
        lot = symbol_info['filters'].get('lot_size', {})
        print(f"[SELL DEBUG] step_size={lot.get('step_size')} min_qty={lot.get('min_qty')} raw_qty={raw_qty} normalized_qty={normalized_qty}")
        if normalized_qty < min_qty:
            raise ValueError(f"Normalized quantity {normalized_qty} below minQty {min_qty}")
        formatted_quantity = format_quantity_for_binance(normalized_qty, step_size)
        if not formatted_quantity or float(formatted_quantity) <= 0:
            raise ValueError(f"Formatted sell quantity invalid/empty: '{formatted_quantity}' from {normalized_qty}")
        if raw_qty != quantity:
            print(f"[SELL ADJUST] req={quantity:.6f} raw={raw_qty:.6f}")
        print(f"[SELL ORDER] raw={raw_qty:.8f} norm={formatted_quantity} msg={msg}")
        order = await client.order_market_sell(symbol=symbol, quantity=formatted_quantity)
        print(f"[SELL ORDER EXECUTED] Sold {formatted_quantity} BTC, Order ID: {order['orderId']}")
        return order
    except Exception as e:
        print(f"Error executing sell order: {e}")
        raise e


async def execute_trade(side):
    price = await get_price()
    global quantity
    timestamp = datetime.now().isoformat()
    usdc_balance, btc_balance = await get_account_balance()
    if not paper_trading:
        if side == "BUY":
            available_usdc = usdc_balance * ENTRY_QUOTE_BUFFER
            if available_usdc < 10:
                print(f"[INSUFFICIENT BALANCE] Need minimum $10 USDC to trade, have ${usdc_balance:.6f}")
                return None
            actual_trade_amount = available_usdc
            quantity = round(actual_trade_amount / price, 6)
            print(f"[BUY PLANNING] Using USDC: ${actual_trade_amount:.6f} → {quantity:.6f} BTC")
        else:
            available_btc = btc_balance * EXIT_ASSET_BUFFER
            if available_btc < 0.00001:
                print(f"[INSUFFICIENT BALANCE] Need minimum 0.00001 BTC to sell, have {btc_balance:.6f}")
                return None
            quantity = available_btc
            actual_trade_amount = quantity * price
            print(f"[SELL PLANNING] Using BTC: {quantity:.6f} BTC → ${actual_trade_amount:.6f} USDC")
        try:
            if side == "BUY":
                order = await execute_buy_order(quantity)
                if order['status'] == 'FILLED':
                    actual_price = float(order['fills'][0]['price']) if order['fills'] else price
                    actual_quantity = float(order['executedQty'])
                    print(f"[BUY EXECUTED] {actual_quantity:.6f} BTC at ${actual_price:.6f}")
                else:
                    print(f"[BUY ORDER ERROR] Order status: {order['status']}")
                    return None
            else:
                order = await execute_sell_order(quantity)
                if order['status'] == 'FILLED':
                    actual_price = float(order['fills'][0]['price']) if order['fills'] else price
                    actual_quantity = float(order['executedQty'])
                    print(f"[SELL EXECUTED] {actual_quantity:.6f} BTC at ${actual_price:.6f}")
                else:
                    print(f"[SELL ORDER ERROR] Order status: {order['status']}")
                    return None
            price = actual_price
            quantity = actual_quantity
        except Exception as e:
            print(f"[ORDER EXECUTION FAILED] {e}")
            clear_position_state()
            return None
    else:
        if side == "BUY":
            available_usdc = usdc_balance * ENTRY_QUOTE_BUFFER if usdc_balance > 0 else 100
            quantity = round(available_usdc / price, 6)
            actual_trade_amount = available_usdc
        else:
            available_btc = btc_balance * EXIT_ASSET_BUFFER if btc_balance > 0 else (100 / price)
            quantity = available_btc
            actual_trade_amount = quantity * price
    df = await fetch_klines()
    prediction, acc = knn_predict(df)
    acc_str = (f"Model accuracy: {acc:.2f}%" if acc is not None else "Model accuracy: N/A")
    await log_trade(("SIMULATED " + side) if paper_trading else side, price, pnl=None,
                    reason=f"Entry (Qty: {quantity:.6f} BTC, ${actual_trade_amount:.6f})" if side == "BUY" else f"Exit to USDC (Qty: {quantity:.6f} BTC, ${actual_trade_amount:.6f})",
                    entry_price=price)
    save_position_state(price, side, timestamp, quantity)
    stats = get_daily_stats()
    stats["trades_count"] += 1
    stats["last_trade_time"] = datetime.now().isoformat()
    if stats["trades_count"] >= max_trades_per_day:
        stats["daily_limit_hit"] = True
        print(f"[DAILY LIMIT] Maximum trades per day ({max_trades_per_day}) reached!")
    save_daily_stats(stats)
    print(f"[DAILY STATS] Trades: {stats['trades_count']}/{max_trades_per_day} | PnL: {format_pnl_text(stats)} | W/L: {stats['wins']}/{stats['losses']}")
    stats = get_daily_stats()
    win_rate = ((stats["wins"] / max(stats["trades_count"], 1)) * 100 if stats["trades_count"] > 0 else 0)
    new_usdc_balance, new_btc_balance = await get_account_balance()
    balance_info = f"""
💰 BALANCE UPDATE:
• USDC: ${usdc_balance:.6f} → ${new_usdc_balance:.6f}
• BTC: {btc_balance:.6f} → {new_btc_balance:.6f}
"""
    summary = f"""
Daily Trading Summary for {stats['date']}

📊 STATISTICS:
• Total Trades: {stats['trades_count']}/{max_trades_per_day}
• Total PnL: {format_pnl_text(stats)}
• Wins: {stats['wins']} | Losses: {stats['losses']}
• Win Rate: {win_rate:.1f}%

🎯 LIMITS:
• Daily Loss Limit: {max_daily_loss:.1%}
• Daily Profit Target: {daily_profit_target:.1%}
• Max Trades: {max_trades_per_day}

🚦 STATUS:
• Daily Limit Hit: {'Yes' if stats['daily_limit_hit'] else 'No'}
• Profit Target Hit: {'Yes' if stats['profit_target_hit'] else 'No'}
• Last Trade: {stats['last_trade_time'] or 'None'}

Bot Status: {'SUSPENDED' if stats['daily_limit_hit'] or stats['profit_target_hit'] else 'ACTIVE'}
"""
    subject = f"[Trading Bot] {( 'SIMULATED ' if paper_trading else '')}{side}"
    body = f"Trade executed at price: {price:.6f}\n{acc_str}\n{balance_info if not paper_trading else ''}\n{summary}"
    print(f"[TRADE NOTIFICATION] {subject}")
    print(body)
    return price



# SL/TP + Trailing Stop Loss - REFACTORED FOR MAINTAINABILITY
async def monitor_position(entry_price, side, position_data=None):
    """Monitor position with separate functions for each exit strategy"""
    global quantity
    if side != "BUY":
        print(f"[SPOT-ONLY] Non-BUY position detected in monitor ({side}). Clearing and returning.")
        clear_position_state()
        return
    
    # Get the actual BTC quantity from position data if available
    if position_data and 'btc_quantity' in position_data:
        btc_quantity = position_data['btc_quantity']
        # Validate that btc_quantity is a valid number
        try:
            btc_quantity = float(btc_quantity)
            if btc_quantity <= 0:
                print(f"[WARNING] Invalid BTC quantity: {btc_quantity}, using global quantity")
                btc_quantity = quantity
        except (ValueError, TypeError):
            print(f"[WARNING] Cannot convert BTC quantity to float: {btc_quantity}, using global quantity")
            btc_quantity = quantity
    else:
        # Fallback to global quantity
        btc_quantity = quantity
        if btc_quantity is None:
            print("[ERROR] BTC quantity is None, cannot monitor position")
            return
    
    # Calculate position size in USD for accurate PnL tracking
    position_size_usd = btc_quantity * entry_price
    
    # Position monitoring variables
    # Profit gate for signal-flip exit disabled per user request
    min_profit_target = 0.0
    max_profit_seen = 0.0
    # Adaptive stepped failsafe tiers: (trigger_profit, locked_floor)
    failsafe_tiers = [
        (0.005, 0.003),   # Reach 0.50% -> lock 0.30%
        (0.0075, 0.005),  # Reach 0.75% -> lock 0.50%
        (0.010, 0.007),   # Reach 1.00% -> lock 0.70%
        (0.0125, 0.009),  # Reach 1.25% -> lock 0.90%
    ]
    failsafe_floor = None  # Updated when higher tiers reached

    print(f"[POSITION MONITORING] {side} position: Entry ${entry_price:.6f}, Size: {btc_quantity:.6f} BTC (${position_size_usd:.2f})")
    print(f"[MONITORING THRESHOLDS] Stop Loss: {stop_loss*100:.6f}%, Flip gate: none")

    # Add timeout to prevent infinite monitoring (24 hours max)
    start_time = time.time()
    max_monitoring_time = 24 * 60 * 60  # 24 hours in seconds
    
    iteration_count = 0
    while True:
        iteration_count += 1
        current_price = await get_price()

        # Check for timeout
        elapsed_time = time.time() - start_time
        if elapsed_time > max_monitoring_time:
            print(f"[TIMEOUT] Position monitoring timeout after {elapsed_time/3600:.6f} hours")
            print(f"Position monitoring has been running for {elapsed_time/3600:.6f} hours. Forcing position closure for safety.")
            clear_position_state()
            return

        # BUY-only: profit when BTC price increases
        change = (current_price - entry_price) / entry_price

        # Debug logging every 10 iterations (roughly every 50 seconds)
        if iteration_count % 10 == 1:
            print(
                f"[MONITOR {iteration_count}] {side} | Entry: ${entry_price:.6f} | Current: ${current_price:.6f} | Change: {change*100:.3f}% | Max: {max_profit_seen*100:.3f}% | Floor: {failsafe_floor*100:.2f}%"
                if failsafe_floor is not None
                else f"[MONITOR {iteration_count}] {side} | Entry: ${entry_price:.6f} | Current: ${current_price:.6f} | Change: {change*100:.3f}% | Max: {max_profit_seen*100:.3f}%"
            )
            print(f"[THRESHOLDS] Stop: {-stop_loss*100:.6f}% | Flip gate: none")

        # Update max profit seen and failsafe threshold
        if change > max_profit_seen:
            max_profit_seen = change
            # Evaluate tiers to update failsafe floor
            new_floor = None
            for trigger, floor in failsafe_tiers:
                if max_profit_seen >= trigger:
                    new_floor = floor
            if new_floor is not None and (failsafe_floor is None or new_floor > failsafe_floor):
                failsafe_floor = new_floor
                print(f"[FAILSAFE FLOOR UPDATE] Max profit {max_profit_seen*100:.2f}% → Locked floor {failsafe_floor*100:.2f}%")

        # 1. CHECK ADAPTIVE FAILSAFE (exit if we dropped below locked floor after having set one)
        if failsafe_floor is not None and change < failsafe_floor:
            exit_success, exit_price = await execute_failsafe_exit(side, current_price, max_profit_seen, failsafe_floor)
            if exit_success:
                final_change = (exit_price - entry_price) / entry_price
                await finalize_exit("failsafe", side, exit_price, final_change, position_size_usd, max_profit_seen)
                return
            else:
                await asyncio.sleep(10)
                continue

        # 2. CHECK STOP LOSS
        if change <= -stop_loss:
            exit_success, exit_price = await execute_stop_loss_exit(side, current_price, change)
            if exit_success:
                # Recalculate change with actual exit price
                final_change = (exit_price - entry_price) / entry_price
                await finalize_exit("stop_loss", side, exit_price, final_change, position_size_usd)
                return
            else:
                # Exit failed, wait and continue monitoring
                await asyncio.sleep(10)
                continue

        # TIME-BASED STALE EXIT
        if (elapsed_time >= stale_exit_minutes * 60 and abs(change) < stale_exit_max_abs_change):
            print(f"[STALE EXIT] Elapsed {elapsed_time/60:.1f}m | Change {change*100:.3f}% < ±{stale_exit_max_abs_change*100:.2f}% -> Forcing exit")
            # Use stop-loss exit machinery (market out)
            exit_success, exit_price = await execute_stop_loss_exit(side, current_price, change)
            if exit_success:
                final_change = (exit_price - entry_price) / entry_price
                await finalize_exit('stale_exit', side, exit_price, final_change, position_size_usd, max_profit_seen)
                return
            else:
                await asyncio.sleep(10)
                continue

        # Wait before next monitoring cycle
        await asyncio.sleep(5)


async def execute_failsafe_exit(side, current_price, max_profit_seen, failsafe_floor):
    """Execute adaptive failsafe exit based on locked profit floor"""
    print(f"[FAILSAFE TRIGGER] Max profit: {max_profit_seen*100:.6f}%, Fell below locked floor: {failsafe_floor*100:.2f}%")
    
    if paper_trading:
        print("[FAILSAFE EXIT] Simulated - would exit position")
        return True, current_price
    
    try:
        # Get current balances to handle dust properly
        usdc_balance, btc_balance = await get_account_balance()
        order_executed = False
        actual_exit_price = current_price
        
        # BUY-only: sell BTC back to USDC
        if side != "BUY":
            print(f"[SPOT-ONLY] Failsafe called with non-BUY side ({side}); skipping.")
            return False, current_price
        if btc_balance < 0.00001:
            print(f"[FAILSAFE EXIT SKIPPED] Expected BTC but have {btc_balance:.6f} BTC")
            return False, current_price
        available_btc = btc_balance * EXIT_ASSET_BUFFER
        print(f"[FAILSAFE EXIT] Converting BTC→USDC: {available_btc:.6f} BTC at ${current_price:.6f}")
        order = await execute_sell_order(available_btc)
        order_executed = (order and order['status'] == 'FILLED')
        
        if order_executed:
            actual_exit_price = float(order['fills'][0]['price']) if order['fills'] else current_price
            print(f"[FAILSAFE EXIT EXECUTED] at ${actual_exit_price:.6f}")
            return True, actual_exit_price
        else:
            print(f"[FAILSAFE EXIT FAILED] Order status: {order['status'] if order else 'None'}")
            print(f"Failed to execute failsafe exit order for {side} position")
            return False, current_price
            
    except Exception as e:
        print(f"[FAILSAFE EXIT ERROR] {e}")
        print(f"Failed to execute failsafe exit: {str(e)}")
        return False, current_price


async def execute_stop_loss_exit(side, current_price, change):
    """Execute stop loss exit"""
    print(f"[STOP LOSS TRIGGER] Current change: {change*100:.6f}%, Stop loss: {-stop_loss*100:.6f}%")
    
    if paper_trading:
        print("[STOP LOSS EXIT] Simulated - would exit position")
        return True, current_price
    
    try:
        # Get current balances to handle dust properly
        usdc_balance, btc_balance = await get_account_balance()
        order_executed = False
        actual_exit_price = current_price
        
        # BUY-only: sell BTC back to USDC (cut losses)
        if side != "BUY":
            print(f"[SPOT-ONLY] Stop-loss called with non-BUY side ({side}); skipping.")
            return False, current_price
        if btc_balance < 0.00001:
            print(f"[STOP LOSS EXIT SKIPPED] Expected BTC but have {btc_balance:.6f} BTC")
            return False, current_price
        available_btc = btc_balance * EXIT_ASSET_BUFFER
        print(f"[STOP LOSS EXIT] Converting BTC→USDC: {available_btc:.6f} BTC at ${current_price:.6f}")
        order = await execute_sell_order(available_btc)
        order_executed = (order and order['status'] == 'FILLED')
        
        if order_executed:
            actual_exit_price = float(order['fills'][0]['price']) if order['fills'] else current_price
            print(f"[STOP LOSS EXIT EXECUTED] at ${actual_exit_price:.6f}")
            return True, actual_exit_price
        else:
            print(f"[STOP LOSS EXIT FAILED] Order status: {order['status'] if order else 'None'}")
            print(f"Failed to execute stop loss exit order for {side} position")
            return False, current_price
            
    except Exception as e:
        print(f"[STOP LOSS EXIT ERROR] {e}")
        print(f"Failed to execute stop loss exit: {str(e)}")
        return False, current_price




## Removed update_failsafe_threshold in favor of stepped adaptive failsafe floors


async def finalize_exit(exit_type, side, current_price, change, position_size_usd, max_profit_seen=None):
    """Finalize position exit - update stats, log trade, clear position, send notifications"""
    # Log the trade
    if exit_type == "failsafe":
        reason = f"Profit fell below locked floor after reaching {max_profit_seen*100:.2f}%" if max_profit_seen else "Failsafe triggered"
        await log_trade("Failsafe Take Profit", current_price, change * position_size_usd, reason=reason)
        subject = f"[Trading Bot] Failsafe Take Profit Triggered"
        body = f"Failsafe take profit executed for {side} at price: {current_price:.2f}\nPnL: {change * position_size_usd:.4f}"
    elif exit_type == "stop_loss":
        await log_trade("Stop Loss", current_price, change * position_size_usd)
        subject = f"[Trading Bot] Stop Loss Triggered"
        body = f"Stop loss executed for {side} at price: {current_price:.2f}\nPnL: {change * position_size_usd:.4f}"
    # signal_flip removed
    elif exit_type == "stale_exit":
        await log_trade("Stale Exit", current_price, change * position_size_usd, reason="Time-based forced exit")
        subject = f"[Trading Bot] Stale Position Exit"
        body = f"Stale exit executed for {side} at price: {current_price:.2f}\nPnL: {change * position_size_usd:.4f}"
    
    # Update daily statistics
    stats = get_daily_stats()
    pnl_abs = change * position_size_usd
    stats["total_pnl_abs"] = stats.get("total_pnl_abs", 0.0) + pnl_abs
    # Update equity-based PnL%
    try:
        # compute current equity and pct if starting_equity is set
        current_equity = await get_equity_usdc()
        stats["last_equity"] = current_equity
        if stats.get("starting_equity"):
            base = stats["starting_equity"]
            if base > 0:
                stats["total_pnl_pct"] = ((current_equity - base) / base) * 100.0
    except Exception:
        pass
    if pnl_abs > 0:
        stats["wins"] += 1
    else:
        stats["losses"] += 1
    # Update flags for daily limits / targets based on % if available
    if stats.get("total_pnl_pct") is not None and stats.get("starting_equity"):
        if stats["total_pnl_pct"] <= -max_daily_loss * 100:
            stats["daily_limit_hit"] = True
        if stats["total_pnl_pct"] >= daily_profit_target * 100:
            stats["profit_target_hit"] = True
    else:
        # fallback absolute thresholds (legacy)
        if stats.get("total_pnl_abs", 0.0) <= -max_daily_loss:
            stats["daily_limit_hit"] = True
        if stats.get("total_pnl_abs", 0.0) >= daily_profit_target:
            stats["profit_target_hit"] = True
    save_daily_stats(stats)
    
    # Clear position state
    clear_position_state()
    
    # Print notification
    print(f"[EXIT NOTIFICATION] {subject}")
    print(body)



async def run_bot():
    global client
    print("[BOT STARTED] Daily trading mode activated")
    
    # Initialize Binance client with error handling
    try:
        client = await AsyncClient.create(API_KEY, API_SECRET)
        print("[CLIENT CONNECTED] Binance API connection established")
    except Exception as e:
        print(f"[FATAL ERROR] Failed to connect to Binance API: {e}")
        print("Please check your API credentials and network connection.")
        return

    # Test API connection and server time sync
    try:
        # Check server time first
        server_time = await client.get_server_time()
        local_time = int(time.time() * 1000)
        time_diff = abs(server_time['serverTime'] - local_time)
        
        print(f"[TIME SYNC] Server time: {server_time['serverTime']}")
        print(f"[TIME SYNC] Local time: {local_time}")
        print(f"[TIME SYNC] Time difference: {time_diff}ms")
        
        if time_diff > 5000:  # More than 5 seconds difference
            print(f"[WARNING] Time difference is {time_diff}ms. This may cause signature errors.")
            print("[WARNING] Please sync your system time or check Docker time settings.")
        else:
            print("[TIME SYNC] ✓ Time synchronization is good")
        
        # Debug API credentials (without exposing secrets)
        print(f"[API DEBUG] API Key length: {len(API_KEY) if API_KEY else 0}")
        print(f"[API DEBUG] API Secret length: {len(API_SECRET) if API_SECRET else 0}")
        print(f"[API DEBUG] API Key starts with: {API_KEY[:8] if API_KEY and len(API_KEY) >= 8 else 'N/A'}...")
        
        # Test with simple endpoint first (no signature required)
        print("[API TEST] Testing ping endpoint...")
        ping_result = await client.ping()
        print(f"[API TEST] ✓ Ping successful: {ping_result}")
        
        # Test exchange info (no signature required)
        print("[API TEST] Testing exchange info...")
        exchange_info = await client.get_exchange_info()
        print(f"[API TEST] ✓ Exchange info retrieved, symbols count: {len(exchange_info.get('symbols', []))}")
        
        # Now test signed endpoint
        print("[API TEST] Testing signed endpoint (account info)...")
        account_info = await client.get_account()
        print(f"[API TEST] ✓ Successfully retrieved account info for user: {account_info.get('accountType', 'Unknown')}")
        
        await get_price()
        print("[API TEST] ✓ Successfully retrieved BTC price")
        
        # Test symbol info retrieval
        print("[API TEST] Testing symbol info retrieval...")
        symbol_info = await get_symbol_info(symbol)
        print(f"[API TEST] ✓ Symbol info retrieved for {symbol_info['symbol']}")
        if 'lot_size' in symbol_info['filters']:
            lot_size = symbol_info['filters']['lot_size']
            print(f"[LOT_SIZE] Min: {lot_size['min_qty']}, Step: {lot_size['step_size']}")
        else:
            print("[WARNING] LOT_SIZE filter not found, using defaults")

    except Exception as e:
        print(f"[FATAL ERROR] API test failed: {e}")
        
        # Enhanced error diagnosis
        if "Signature for this request is not valid" in str(e):
            print("\n🔍 SIGNATURE ERROR DIAGNOSIS:")
            print("1. ✓ Time sync is good (117ms difference)")
            print("2. Check API key format:")
            print("   - Should be 64 characters long")
            print("   - Should contain only letters and numbers")
            print("3. Check API secret format:")
            print("   - Should be 64 characters long") 
            print("   - Should contain only letters, numbers, and some symbols")
            print("4. Verify API permissions:")
            print("   - Enable 'Spot & Margin Trading' permission")
            print("   - Enable 'Read' permission")
            print("   - Disable 'Futures' unless needed")
            print("5. Check environment variables:")
            print("   - Ensure no extra spaces in .env file")
            print("   - Ensure no quotes around the keys")
            print("   - Format: BINANCE_API_KEY=your_key_here")
            print("6. If keys are new, wait 5-10 minutes for activation")
            print("\n💡 SOLUTION:")
            print("1. Go to Binance.com → Profile → API Management")
            print("2. Delete the current API key")
            print("3. Create a new API key with these permissions:")
            print("   ✓ Enable Reading")
            print("   ✓ Enable Spot & Margin Trading")
            print("   ✗ Disable Futures (unless needed)")
            print("4. Update your .env file with the new keys")
            print("5. Restart the bot")
        
        print("\nPossible issues:")
        print("1. Check API keys are correct")
        print("2. Ensure API has Spot Trading permissions")
        print("3. Verify system time is synchronized")
        print("4. Check if IP is whitelisted (if API has IP restrictions)")
        print("Bot cannot function without API access. Exiting...")
        return

    # Initialize starting equity for the day if not set
    stats_init = get_daily_stats()
    if not stats_init.get("starting_equity"):
        try:
            eq = await get_equity_usdc()
            stats_init["starting_equity"] = eq
            stats_init["last_equity"] = eq
            stats_init["total_pnl_pct"] = 0.0
            save_daily_stats(stats_init)
            print(f"[EQUITY INIT] Starting equity for {stats_init['date']}: ${eq:.2f}")
        except Exception as e:
            print(f"[EQUITY INIT ERROR] {e}")

    existing_position = load_position_state()
    if existing_position:
        print(
            f"[RESUMING POSITION] {existing_position['side']} at {existing_position['entry_price']}"
        )
        try:
            await monitor_position(existing_position["entry_price"], existing_position["side"], existing_position)
        except Exception as e:
            print(f"[ERROR] Failed to resume position monitoring: {e}")
            clear_position_state()

    last_summary_date = None
    consecutive_errors = 0
    max_consecutive_errors = 5
    last_heartbeat = time.time()
    heartbeat_interval = 300  # 5 minutes

    while True:
        try:
            # Heartbeat message every 5 minutes
            current_time = time.time()
            current_datetime = datetime.now()
            if current_time - last_heartbeat > heartbeat_interval:
                stats = get_daily_stats()
                pnl_text = (
                    f"{stats.get('total_pnl_pct', 0):.2f}%"
                    if stats.get('starting_equity') and stats.get('total_pnl_pct') is not None else f"${stats.get('total_pnl_abs', 0.0):.4f}"
                )
                print(f"[HEARTBEAT {current_datetime.strftime('%H:%M:%S')}] Bot alive - Trades: {stats['trades_count']}/{max_trades_per_day} | PnL: {pnl_text}")
                last_heartbeat = current_time
            
            # Send daily summary at start of new day
            current_date = datetime.now().strftime("%Y-%m-%d")
            if last_summary_date and last_summary_date != current_date:
                try:
                    await send_daily_summary()
                except Exception as e:
                    print(f"[ERROR] Failed to send daily summary: {e}")
            last_summary_date = current_date

            # Reset starting equity at day roll if date changed
            stats = get_daily_stats()
            if stats.get('date') != current_date:
                try:
                    eq = await get_equity_usdc()
                    new_stats = create_new_daily_stats(current_date)
                    new_stats['starting_equity'] = eq
                    new_stats['last_equity'] = eq
                    save_daily_stats(new_stats)
                    print(f"[EQUITY RESET] New day {current_date}, starting equity: ${eq:.2f}")
                except Exception as e:
                    print(f"[EQUITY RESET ERROR] {e}")

            can_trade_now, reason = can_trade()
            if not can_trade_now:
                print(f"[TRADING SUSPENDED {current_datetime.strftime('%H:%M:%S')}] {reason}")
                if "Daily" in reason:
                    await asyncio.sleep(3600) 
                else:
                    await asyncio.sleep(300)  
                continue

            current_position = load_position_state()
            if current_position:
                print(f"[POSITION ACTIVE {current_datetime.strftime('%H:%M:%S')}] Monitoring existing position...")
                try:
                    await monitor_position(
                        current_position["entry_price"], current_position["side"], current_position
                    )
                except Exception as e:
                    print(f"[ERROR] Position monitoring failed: {e}")
                    print(f"Error monitoring position: {str(e)}")
                continue

            try:
                df = await fetch_klines()
                prediction, _ = knn_predict(df)
            except Exception as e:
                print(f"[ERROR] Failed to get prediction: {e}")
                await asyncio.sleep(300)
                continue

            print(f"[KNN PREDICTION {current_datetime.strftime('%H:%M:%S')}] {'BUY' if prediction == 1 else 'SELL'}")

            if prediction == 1:
                # BUY signal: Convert USDC to BTC (spot-long only)
                try:
                    price = await execute_trade("BUY")
                    if price:  # Only monitor if trade was successful
                        await monitor_position(price, "BUY")
                except Exception as e:
                    print(f"[ERROR] BUY trade execution failed: {e}")
                    print(f"BUY trade failed: {str(e)}")
            else:
                # Spot-only mode: ignore SELL prediction; hold or wait for next BUY
                print(f"[SPOT-ONLY] SELL signal ignored; waiting for BUY.")

            # Sleep for a shorter interval to check for trades more frequently
            await asyncio.sleep(60)  # Check every 1 minute instead of 15 minutes
            consecutive_errors = 0  # Reset error counter on successful iteration

        except KeyboardInterrupt:
            print("[BOT STOPPED] Shutting down gracefully...")
            break
        except Exception as e:
            consecutive_errors += 1
            print(f"[ERROR {consecutive_errors}/{max_consecutive_errors}] Unexpected error: {e}")
            
            if consecutive_errors >= max_consecutive_errors:
                print(f"[FATAL ERROR] Too many consecutive errors ({max_consecutive_errors}). Bot stopping to prevent damage.")
                print(f"Bot stopped after {max_consecutive_errors} consecutive errors. Last error: {str(e)}")
                break
                
            await asyncio.sleep(300)  # Wait 5 minutes on error

    # Cleanup
    try:
        if client:
            await client.close_connection()
            print("[CLIENT DISCONNECTED] Binance API connection closed")
    except:
        pass


if __name__ == "__main__":
    print("=== TRADING BOT STARTUP ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Paper trading: {paper_trading}")
    # Fixed trade size removed (dynamic: uses all available balance)
    print(f"Email notifications: Disabled (removed)")
    
    # Check system time
    current_time = datetime.now()
    utc_time = datetime.utcnow()
    print(f"System time: {current_time}")
    print(f"UTC time: {utc_time}")
    print("=" * 28)
    
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Bot stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n[FATAL STARTUP ERROR] {e}")
        print("Bot failed to start. Please check your configuration and try again.")
    finally:
        print("[SHUTDOWN] Trading bot terminated")