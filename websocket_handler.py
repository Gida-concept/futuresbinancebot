import asyncio
import json
import logging
from binance import AsyncClient, BinanceSocketManager
from datetime import datetime
import config

logger = logging.getLogger(__name__)


class WebSocketHandler:
    def __init__(self, data_managers: dict):
        self.data_managers = data_managers  # {symbol: DataManager}
        self.client = None
        self.bsm = None
        self.tasks = []
        self.running = False

    async def start(self):
        """Start WebSocket connections for all symbols"""
        try:
            self.client = await AsyncClient.create(
                config.BINANCE_API_KEY,
                config.BINANCE_API_SECRET,
                testnet=config.BINANCE_TESTNET
            )

            self.bsm = BinanceSocketManager(self.client)
            self.running = True

            # Start WebSocket for each symbol
            for symbol in config.SYMBOLS:
                task = asyncio.create_task(self._handle_symbol_stream(symbol))
                self.tasks.append(task)
                logger.info(f"Started WebSocket for {symbol}")

            logger.info("All WebSocket connections started")

        except Exception as e:
            logger.error(f"Error starting WebSocket: {e}")
            self.running = False

    async def _handle_symbol_stream(self, symbol: str):
        """Handle WebSocket stream for a specific symbol"""
        while self.running:
            try:
                stream = self.bsm.kline_socket(symbol, interval=config.TIMEFRAME)

                async with stream as s:
                    while self.running:
                        msg = await s.recv()

                        if msg['e'] == 'kline':
                            kline = msg['k']

                            # Only process closed candles
                            if kline['x']:  # is_closed
                                candle_data = {
                                    'timestamp': kline['t'],
                                    'open': kline['o'],
                                    'high': kline['h'],
                                    'low': kline['l'],
                                    'close': kline['c'],
                                    'volume': kline['v']
                                }

                                # Add to data manager
                                if symbol in self.data_managers:
                                    self.data_managers[symbol].add_candle(candle_data)
                                    logger.debug(f"New candle for {symbol}: Close={kline['c']}")

            except asyncio.CancelledError:
                logger.info(f"WebSocket cancelled for {symbol}")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket for {symbol}: {e}")
                logger.info(f"Reconnecting WebSocket for {symbol} in {config.WS_RECONNECT_DELAY}s...")
                await asyncio.sleep(config.WS_RECONNECT_DELAY)

    async def stop(self):
        """Stop all WebSocket connections"""
        self.running = False

        for task in self.tasks:
            task.cancel()

        await asyncio.gather(*self.tasks, return_exceptions=True)

        if self.client:
            await self.client.close_connection()

        logger.info("All WebSocket connections stopped")