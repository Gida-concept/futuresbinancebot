import asyncio
import logging
from datetime import datetime, timedelta
import signal
import sys
import config
from account_manager import AccountManager
from telegram_alerts import TelegramNotifier
from data_manager import DataManager
from ml_model import MLModel
from trade_manager import TradeManager
from websocket_handler import WebSocketHandler

# Setup logging with UTF-8 encoding for Windows
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs

    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self):
        self.account_manager = None
        self.telegram = None
        self.data_managers = {}
        self.ml_models = {}
        self.trade_manager = None
        self.websocket_handler = None
        self.running = False
        self.heartbeat_task = None
        self.last_trade_time = None  # ← ADD: Track when last trade closed
        self.trade_cooldown = 60  # ← ADD: Wait 60 seconds between trades

    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("=" * 50)
            logger.info("Initializing Trading Bot...")
            logger.info("=" * 50)

            # Initialize telegram first
            self.telegram = TelegramNotifier()

            # Initialize account manager
            self.account_manager = AccountManager()
            if not await self.account_manager.initialize():
                logger.error("Failed to initialize account")
                await self.telegram.send_error("Failed to initialize account. Check API keys and network.")
                return False

            # Check if balance is sufficient
            if self.account_manager.get_balance() == 0:
                logger.error("Cannot start bot with 0 balance. Get test funds first.")
                await self.telegram.send_error("Balance is 0. Get test funds from testnet.binancefuture.com")
                return False

            # Initialize trade manager
            self.trade_manager = TradeManager(self.account_manager, self.telegram)
            if not await self.trade_manager.initialize():
                logger.error("Failed to initialize trade manager")
                await self.telegram.send_error("Failed to initialize trade manager")
                return False

            # Initialize data managers and ML models for each symbol
            for symbol in config.SYMBOLS:
                logger.info(f"Initializing {symbol}...")

                # Data manager
                dm = DataManager(symbol)
                dm.set_client(self.account_manager.client)

                # Try to load saved data, otherwise fetch historical
                if not dm.load_data():
                    logger.info(f"Fetching historical data for {symbol}...")
                    df = dm.fetch_historical_data()
                    if df.empty:
                        logger.warning(f"Failed to fetch data for {symbol}, skipping...")
                        continue

                self.data_managers[symbol] = dm

                # ML Model
                model = MLModel(symbol)

                # Try to load saved model, otherwise train new one
                if not model.load_model():
                    logger.info(f"Training new model for {symbol}...")
                    features_df = dm.prepare_features()
                    if not features_df.empty:
                        model.train(features_df)
                    else:
                        logger.warning(f"Cannot train model for {symbol} - insufficient data")

                self.ml_models[symbol] = model

            if not self.data_managers:
                logger.error("No symbols initialized successfully")
                await self.telegram.send_error("No symbols initialized. Check network and API.")
                return False

            # Initialize WebSocket handler
            self.websocket_handler = WebSocketHandler(self.data_managers)

            # Send startup notification
            await self.telegram.send_startup(
                symbols=list(self.data_managers.keys()),
                balance=self.account_manager.get_balance()
            )

            logger.info("=" * 50)
            logger.info("Bot initialization complete!")
            logger.info(f"Trading symbols: {list(self.data_managers.keys())}")
            logger.info(f"Initial balance: ${self.account_manager.get_balance():.2f}")
            logger.info(f"Trade cooldown: {self.trade_cooldown} seconds")
            logger.info("=" * 50)
            return True

        except Exception as e:
            logger.error(f"Error during initialization: {e}", exc_info=True)
            if self.telegram:
                await self.telegram.send_error(f"Initialization failed: {e}")
            return False

    async def run(self):
        """Main bot loop"""
        try:
            self.running = True

            # Start WebSocket
            await self.websocket_handler.start()

            # Start heartbeat
            if config.SEND_HEARTBEAT:
                self.heartbeat_task = asyncio.create_task(self.send_heartbeat())

            logger.info("Bot is now running. Monitoring markets...")

            # Main processing loop
            while self.running:
                await self.process_signals()
                await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            await self.telegram.send_error(f"Bot error: {e}")

    async def process_signals(self):
        """Process trading signals for all symbols and pick the best one"""
        try:
            # Monitor active trade first
            if self.trade_manager.has_active_trade():
                active_symbol = self.trade_manager.active_trade.symbol
                if active_symbol in self.data_managers:
                    # ← FIX: Get LIVE price from exchange, not cached data
                    ticker = self.trade_manager.client.futures_symbol_ticker(symbol=active_symbol)
                    current_price = float(ticker['price'])
                    await self.trade_manager.monitor_active_trade(current_price)
                return

            # ← ADD: Check if we're in cooldown period
            if self.last_trade_time:
                time_since_last_trade = (datetime.now() - self.last_trade_time).total_seconds()
                if time_since_last_trade < self.trade_cooldown:
                    remaining = self.trade_cooldown - time_since_last_trade
                    logger.debug(f"Trade cooldown: {remaining:.0f}s remaining")
                    return

            # Collect all valid signals from all symbols
            signals = []
            long_signals = 0
            short_signals = 0

            for symbol in self.data_managers.keys():
                dm = self.data_managers[symbol]
                model = self.ml_models.get(symbol)

                if not model or not model.is_trained:
                    continue

                # Check if we need to retrain (but not on EVERY candle!)
                if model.should_retrain(dm.candle_count) and dm.candle_count > 0:
                    logger.info(f"Retraining model for {symbol}...")
                    features_df = dm.prepare_features()
                    if not features_df.empty:
                        model.train(features_df)

                # Get latest features and predict
                features = dm.get_latest_features()
                if features is not None:
                    prediction, probability = model.predict(features)

                    if prediction is not None and probability is not None:
                        # ← FIX: Get LIVE price from exchange
                        ticker = self.trade_manager.client.futures_symbol_ticker(symbol=symbol)
                        current_price = float(ticker['price'])

                        direction = 'LONG' if prediction == 1 else 'SHORT'

                        # Count signal directions
                        if prediction == 1:
                            long_signals += 1
                        else:
                            short_signals += 1

                        # Only consider signals above threshold
                        if probability >= config.PREDICTION_THRESHOLD:
                            signals.append({
                                'symbol': symbol,
                                'prediction': prediction,
                                'probability': probability,
                                'price': current_price,
                                'direction': direction
                            })

                            logger.info(f"📊 Signal detected: {symbol} | "
                                        f"{direction} | "
                                        f"Confidence: {probability:.2%} | "
                                        f"Price: ${current_price:,.2f}")

            # Log signal distribution
            total_signals = long_signals + short_signals
            if total_signals > 0:
                logger.info(f"")
                logger.info(
                    f"📈 Signal Distribution: LONG: {long_signals} ({long_signals / total_signals * 100:.1f}%) | "
                    f"SHORT: {short_signals} ({short_signals / total_signals * 100:.1f}%)")
                logger.info(f"")

            # If we have signals, pick the best one
            if signals:
                # Sort by probability (highest confidence first)
                signals.sort(key=lambda x: x['probability'], reverse=True)
                best_signal = signals[0]

                logger.info("=" * 60)
                logger.info(f"🎯 BEST SIGNAL SELECTED:")
                logger.info(f"   Symbol: {best_signal['symbol']}")
                logger.info(f"   Direction: {best_signal['direction']}")
                logger.info(f"   Confidence: {best_signal['probability']:.2%}")
                logger.info(f"   Price: ${best_signal['price']:,.2f}")

                # Show other signals for comparison
                if len(signals) > 1:
                    logger.info(f"\n   Other signals (not selected):")
                    for i, sig in enumerate(signals[1:], 1):
                        logger.info(f"   {i}. {sig['symbol']}: {sig['direction']} "
                                    f"({sig['probability']:.2%}) - "
                                    f"${sig['price']:,.2f}")

                logger.info("=" * 60)

                # Send signal comparison to Telegram
                if len(signals) > 1:
                    comparison_msg = (
                        f"🎯 <b>Signal Analysis</b>\n\n"
                        f"<b>Selected: {best_signal['symbol']}</b>\n"
                        f"Direction: {best_signal['direction']}\n"
                        f"Confidence: {best_signal['probability']:.2%}\n\n"
                        f"<b>Other signals:</b>\n"
                    )
                    for sig in signals[1:4]:
                        comparison_msg += (
                            f"• {sig['symbol']}: {sig['direction']} "
                            f"({sig['probability']:.2%})\n"
                        )

                    # Add signal distribution
                    comparison_msg += (
                        f"\n<b>Distribution:</b>\n"
                        f"LONG: {long_signals} | SHORT: {short_signals}"
                    )

                    await self.telegram.send_message(comparison_msg)

                # Open trade with best signal
                if not self.trade_manager.has_active_trade():
                    success = await self.trade_manager.open_trade(
                        symbol=best_signal['symbol'],
                        prediction=best_signal['prediction'],
                        current_price=best_signal['price']
                    )

                    # ← ADD: Don't set cooldown here, wait for trade to close
            else:
                logger.debug("No signals above threshold. Waiting...")

        except Exception as e:
            logger.error(f"Error processing signals: {e}", exc_info=True)

    def set_trade_cooldown(self):
        """Set cooldown after trade closes"""
        self.last_trade_time = datetime.now()
        logger.info(f"⏰ Trade cooldown activated: {self.trade_cooldown} seconds")

    async def send_heartbeat(self):
        """Send periodic heartbeat messages"""
        while self.running:
            try:
                await asyncio.sleep(config.HEARTBEAT_INTERVAL)

                active_trades = 1 if self.trade_manager.has_active_trade() else 0

                await self.telegram.send_heartbeat(
                    active_trades=active_trades,
                    balance=self.account_manager.get_balance()
                )

                total_pnl = self.account_manager.get_total_pnl()
                pnl_percent = self.account_manager.get_pnl_percentage()

                await self.telegram.send_balance_update(
                    balance=self.account_manager.get_balance(),
                    total_pnl=total_pnl,
                    pnl_percent=pnl_percent
                )

            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down bot...")
        self.running = False

        if self.heartbeat_task:
            self.heartbeat_task.cancel()

        if self.websocket_handler:
            await self.websocket_handler.stop()

        for symbol, dm in self.data_managers.items():
            dm.save_data()

        if self.trade_manager:
            stats = self.trade_manager.get_trade_stats()
            final_balance = self.account_manager.get_balance()

            logger.info("=" * 50)
            logger.info("Bot Statistics:")
            logger.info(f"Total Trades: {stats['total_trades']}")
            logger.info(f"Win Rate: {stats['win_rate']:.2f}%")
            logger.info(f"Total PnL: ${stats['total_pnl']:.2f}")
            logger.info(f"Final Balance: ${final_balance:.2f}")
            logger.info("=" * 50)

            shutdown_msg = (
                f"🛑 <b>BOT STOPPED</b>\n\n"
                f"Total Trades: {stats['total_trades']}\n"
                f"Win Rate: {stats['win_rate']:.2f}%\n"
                f"Total PnL: ${stats['total_pnl']:.2f}\n"
                f"Final Balance: ${final_balance:.2f}"
            )
            await self.telegram.send_message(shutdown_msg)

        logger.info("Shutdown complete")


async def main():
    bot = TradingBot()

    try:
        if await bot.initialize():
            await bot.run()
        else:
            logger.error("Failed to initialize bot")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")