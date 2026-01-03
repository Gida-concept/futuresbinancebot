import logging
from telegram import Bot
from telegram.error import TelegramError
import config
import asyncio

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self):
        self.bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.enabled = True

        # Validate on init
        if not config.TELEGRAM_BOT_TOKEN or config.TELEGRAM_BOT_TOKEN == 'your_bot_token':
            logger.warning("Telegram bot token not configured - notifications disabled")
            self.enabled = False
        if not config.TELEGRAM_CHAT_ID or config.TELEGRAM_CHAT_ID == 'your_chat_id':
            logger.warning("Telegram chat ID not configured - notifications disabled")
            self.enabled = False

    async def send_message(self, message: str):
        """Send message to Telegram"""
        if not self.enabled:
            logger.debug(f"Telegram disabled, would send: {message[:50]}...")
            return

        try:
            await asyncio.wait_for(
                self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='HTML'
                ),
                timeout=10.0  # 10 second timeout
            )
            logger.debug(f"Telegram message sent: {message[:50]}...")
        except asyncio.TimeoutError:
            logger.error("Telegram message timeout - check VPN and internet")
        except TelegramError as e:
            logger.error(f"Failed to send Telegram message: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram: {e}")

    async def send_trade_opened(self, symbol: str, side: str, entry_price: float,
                                position_size: float, tp_price: float, sl_price: float):
        """Send trade opened notification"""
        message = (
            f"🟢 <b>TRADE OPENED</b>\n\n"
            f"Symbol: {symbol}\n"
            f"Side: {side}\n"
            f"Entry Price: ${entry_price:.4f}\n"
            f"Position Size: ${position_size:.2f}\n"
            f"Take Profit: ${tp_price:.4f}\n"
            f"Stop Loss: ${sl_price:.4f}\n"
            f"Leverage: {config.LEVERAGE}x"
        )
        await self.send_message(message)

    async def send_trade_closed(self, symbol: str, side: str, entry_price: float,
                                exit_price: float, pnl: float, balance: float, reason: str):
        """Send trade closed notification with PnL"""
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        pnl_sign = "+" if pnl >= 0 else ""

        message = (
            f"{pnl_emoji} <b>TRADE CLOSED</b>\n\n"
            f"Symbol: {symbol}\n"
            f"Side: {side}\n"
            f"Entry: ${entry_price:.4f}\n"
            f"Exit: ${exit_price:.4f}\n"
            f"PnL: {pnl_sign}${pnl:.2f}\n"
            f"Reason: {reason}\n"
            f"Updated Balance: ${balance:.2f}"
        )
        await self.send_message(message)

    async def send_balance_update(self, balance: float, total_pnl: float, pnl_percent: float):
        """Send periodic balance update"""
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        pnl_sign = "+" if total_pnl >= 0 else ""

        message = (
            f"{pnl_emoji} <b>BALANCE UPDATE</b>\n\n"
            f"Current Balance: ${balance:.2f}\n"
            f"Total PnL: {pnl_sign}${total_pnl:.2f} ({pnl_sign}{pnl_percent:.2f}%)"
        )
        await self.send_message(message)

    async def send_heartbeat(self, active_trades: int, balance: float):
        """Send heartbeat message"""
        message = (
            f"💚 <b>BOT HEARTBEAT</b>\n\n"
            f"Status: Running\n"
            f"Active Trades: {active_trades}\n"
            f"Balance: ${balance:.2f}"
        )
        await self.send_message(message)

    async def send_error(self, error_message: str):
        """Send error notification"""
        message = f"⚠️ <b>ERROR</b>\n\n{error_message}"
        await self.send_message(message)

    async def send_startup(self, symbols: list, balance: float):
        """Send bot startup notification"""
        symbols_str = ", ".join(symbols)
        message = (
            f"🚀 <b>BOT STARTED</b>\n\n"
            f"Symbols: {symbols_str}\n"
            f"Timeframe: {config.TIMEFRAME}\n"
            f"Leverage: {config.LEVERAGE}x\n"
            f"Initial Balance: ${balance:.2f}\n"
            f"Max Active Trades: {config.MAX_ACTIVE_TRADES}"
        )
        await self.send_message(message)