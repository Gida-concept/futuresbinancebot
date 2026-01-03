import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
import requests
import config

logger = logging.getLogger(__name__)


class AccountManager:
    def __init__(self):
        self.client = None
        self.balance = 0.0
        self.initial_balance = 0.0

    async def initialize(self):
        """Initialize account and fetch initial balance"""
        try:
            logger.info(f"Connecting to Binance (Testnet: {config.BINANCE_TESTNET})...")

            if config.BINANCE_TESTNET:
                self.client = Client(
                    api_key=config.BINANCE_API_KEY,
                    api_secret=config.BINANCE_API_SECRET,
                    testnet=True
                )
                logger.info("Using Binance Futures Testnet")
            else:
                self.client = Client(
                    api_key=config.BINANCE_API_KEY,
                    api_secret=config.BINANCE_API_SECRET
                )
                logger.info("Using Binance Futures Production")

            try:
                logger.info("Testing connection to Binance...")
                result = self.client.futures_ping()
                logger.info(f"[OK] Ping successful: {result}")
            except BinanceRequestException as e:
                logger.error(f"[FAIL] Binance request error: {str(e)}")
                logger.error("\nTroubleshooting:")
                logger.error("1. Your API keys might be incorrect")
                logger.error("2. API keys must be from https://testnet.binancefuture.com")
                logger.error("3. Make sure 'Enable Futures' is checked")
                logger.error("4. Try creating new API keys")
                raise

            logger.info("Fetching account balance...")
            await self.update_balance()

            if self.balance == 0:
                logger.warning("=" * 60)
                logger.warning("WARNING: Your testnet balance is 0 USDT!")
                logger.warning("To get test funds:")
                logger.warning("1. Go to https://testnet.binancefuture.com")
                logger.warning("2. Login and look for 'Get Test Funds'")
                logger.warning("=" * 60)

            self.initial_balance = self.balance
            logger.info(f"[OK] Account initialized. Balance: ${self.balance:.2f} USDT")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize account: {e}")
            return False

    async def update_balance(self):
        """Fetch current USDT balance from Binance Futures"""
        try:
            account_info = self.client.futures_account_balance()

            for asset in account_info:
                if asset['asset'] == 'USDT':
                    self.balance = float(asset['balance'])
                    logger.info(f"Balance updated: ${self.balance:.2f} USDT")
                    return self.balance

            logger.warning("USDT balance not found in account")
            self.balance = 0.0
            return 0.0

        except BinanceAPIException as e:
            logger.error(f"Binance API error getting balance: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return None

    def get_balance(self) -> float:
        """Get current balance"""
        return self.balance

    def calculate_position_size(self) -> float:
        """Calculate position size based on account balance"""
        if self.balance == 0:
            logger.error("Cannot calculate position size: balance is 0")
            return 0.0

        # Position size in USDT (before leverage)
        position_size = self.balance * config.POSITION_SIZE_PERCENT / 100

        logger.info(f"Calculated position size: ${position_size:.2f} USDT "
                    f"({config.POSITION_SIZE_PERCENT}% of ${self.balance:.2f})")
        return position_size

    def calculate_position_size_with_leverage(self) -> float:
        """Calculate position size with leverage applied"""
        base_size = self.calculate_position_size()
        leveraged_size = base_size * config.LEVERAGE
        logger.info(f"Position size with {config.LEVERAGE}x leverage: ${leveraged_size:.2f}")
        return leveraged_size

    def calculate_take_profit_amount(self, position_size: float) -> float:
        """
        Calculate take profit amount based on POSITION size

        Args:
            position_size: The actual position size in USDT (with leverage)

        Returns:
            TP amount in USDT
        """
        tp_amount = position_size * config.TAKE_PROFIT_PERCENT / 100
        logger.info(f"Take profit: ${tp_amount:.2f} "
                    f"({config.TAKE_PROFIT_PERCENT}% of position ${position_size:.2f})")
        return tp_amount

    def calculate_stop_loss_amount(self, position_size: float) -> float:
        """
        Calculate stop loss amount based on POSITION size

        Args:
            position_size: The actual position size in USDT (with leverage)

        Returns:
            SL amount in USDT
        """
        sl_amount = position_size * config.STOP_LOSS_PERCENT / 100
        logger.info(f"Stop loss: ${sl_amount:.2f} "
                    f"({config.STOP_LOSS_PERCENT}% of position ${position_size:.2f})")
        return sl_amount

    def get_total_pnl(self) -> float:
        """Calculate total PnL since bot started"""
        return self.balance - self.initial_balance

    def get_pnl_percentage(self) -> float:
        """Calculate PnL percentage"""
        if self.initial_balance == 0:
            return 0.0
        return ((self.balance - self.initial_balance) / self.initial_balance) * 100