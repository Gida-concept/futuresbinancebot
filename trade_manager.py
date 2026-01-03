import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime
import asyncio
import config

logger = logging.getLogger(__name__)


class Trade:
    def __init__(self, symbol: str, side: str, entry_price: float,
                 position_size: float, tp_price: float, sl_price: float):
        self.symbol = symbol
        self.side = side  # 'LONG' or 'SHORT'
        self.entry_price = entry_price
        self.position_size = position_size
        self.tp_price = tp_price
        self.sl_price = sl_price
        self.entry_time = datetime.now()
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0.0
        self.status = 'OPEN'
        self.close_reason = None


class TradeManager:
    def __init__(self, account_manager, telegram_notifier):
        self.client = None
        self.account_manager = account_manager
        self.telegram = telegram_notifier
        self.active_trade = None
        self.trade_history = []

    async def initialize(self):
        """Initialize trade manager"""
        try:
            # Use the same client instance from account_manager
            self.client = self.account_manager.client

            if self.client is None:
                logger.error("Account manager client not initialized")
                return False

            # Check for existing open positions and close them
            await self.cleanup_existing_positions()

            # Set leverage for all symbols
            for symbol in config.SYMBOLS:
                try:
                    self.client.futures_change_leverage(
                        symbol=symbol,
                        leverage=config.LEVERAGE
                    )
                    logger.info(f"Leverage set to {config.LEVERAGE}x for {symbol}")
                except Exception as e:
                    logger.warning(f"Could not set leverage for {symbol}: {e}")

            logger.info("✓ Trade manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing trade manager: {e}")
            return False

    async def cleanup_existing_positions(self):
        """Close any existing open positions from previous sessions"""
        try:
            logger.info("Checking for existing open positions...")

            # Get all open positions
            positions = self.client.futures_position_information()

            positions_closed = 0
            total_pnl = 0.0

            for position in positions:
                position_amt = float(position['positionAmt'])

                # If position amount is not zero, there's an open position
                if position_amt != 0:
                    symbol = position['symbol']
                    entry_price = float(position['entryPrice'])
                    unrealized_pnl = float(position['unRealizedProfit'])

                    logger.warning(f"⚠️  Found open position: {symbol}")
                    logger.warning(f"   Amount: {position_amt}")
                    logger.warning(f"   Entry Price: ${entry_price:.2f}")
                    logger.warning(f"   Unrealized PnL: ${unrealized_pnl:.2f}")

                    # Close the position
                    try:
                        side = 'SELL' if position_amt > 0 else 'BUY'
                        quantity = abs(position_amt)

                        logger.info(f"Closing position for {symbol}...")

                        # Market order to close position
                        close_order = self.client.futures_create_order(
                            symbol=symbol,
                            side=side,
                            type='MARKET',
                            quantity=quantity,
                            reduceOnly=True
                        )

                        logger.info(f"✓ Closed position for {symbol}: {close_order}")
                        positions_closed += 1
                        total_pnl += unrealized_pnl

                        # Send Telegram notification
                        await self.telegram.send_message(
                            f"⚠️ <b>CLEANUP: Position Closed on Restart</b>\n\n"
                            f"Symbol: {symbol}\n"
                            f"Amount: {position_amt}\n"
                            f"Entry: ${entry_price:.2f}\n"
                            f"PnL: ${unrealized_pnl:.2f}\n"
                            f"Reason: Bot restart - safety cleanup"
                        )

                    except Exception as e:
                        logger.error(f"Failed to close position for {symbol}: {e}")
                        await self.telegram.send_error(
                            f"Failed to close existing position for {symbol}: {e}"
                        )

            # Cancel all open orders
            await self.cancel_all_open_orders()

            if positions_closed > 0:
                logger.warning(
                    f"🔧 Cleanup complete: Closed {positions_closed} position(s), Total PnL: ${total_pnl:.2f}")
                await self.telegram.send_message(
                    f"🔧 <b>Cleanup Summary</b>\n\n"
                    f"Positions Closed: {positions_closed}\n"
                    f"Total PnL: ${total_pnl:.2f}\n"
                    f"Bot is now ready for fresh trades"
                )
            else:
                logger.info("✓ No existing positions found - all clear!")

            # Update account balance after cleanup
            await self.account_manager.update_balance()

        except Exception as e:
            logger.error(f"Error during position cleanup: {e}")
            await self.telegram.send_error(f"Error during position cleanup: {e}")

    async def cancel_all_open_orders(self):
        """Cancel all open orders for all symbols"""
        try:
            logger.info("Canceling all open orders...")

            orders_cancelled = 0

            for symbol in config.SYMBOLS:
                try:
                    # Get all open orders for this symbol
                    open_orders = self.client.futures_get_open_orders(symbol=symbol)

                    for order in open_orders:
                        try:
                            # Cancel the order
                            self.client.futures_cancel_order(
                                symbol=symbol,
                                orderId=order['orderId']
                            )
                            logger.info(f"Cancelled order {order['orderId']} for {symbol}")
                            orders_cancelled += 1
                        except Exception as e:
                            logger.warning(f"Could not cancel order {order['orderId']}: {e}")

                except Exception as e:
                    logger.warning(f"Could not get open orders for {symbol}: {e}")

            if orders_cancelled > 0:
                logger.info(f"✓ Cancelled {orders_cancelled} open order(s)")
            else:
                logger.info("✓ No open orders to cancel")

        except Exception as e:
            logger.error(f"Error canceling open orders: {e}")

    def has_active_trade(self) -> bool:
        """Check if there's an active trade"""
        return self.active_trade is not None and self.active_trade.status == 'OPEN'

    async def open_trade(self, symbol: str, prediction: int, current_price: float) -> bool:
        """
        Open a new trade with position-based TP/SL using REAL-TIME prices
        prediction: 1 for LONG, 0 for SHORT
        """
        try:
            if self.has_active_trade():
                logger.warning("Cannot open trade: already have an active trade")
                return False

            # Determine side
            side = 'LONG' if prediction == 1 else 'SHORT'
            position_side = 'BUY' if side == 'LONG' else 'SELL'

            # Get account balance and calculate position
            await self.account_manager.update_balance()

            # Get position size WITHOUT leverage (base USDT amount)
            position_size_base = self.account_manager.calculate_position_size()

            # Get position size WITH leverage (actual position value)
            position_size_leveraged = self.account_manager.calculate_position_size_with_leverage()

            if position_size_base == 0:
                logger.error("Position size is 0 - cannot open trade")
                return False

            # Get REAL-TIME market price RIGHT NOW
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            real_time_price = float(ticker['price'])

            logger.info(f"🔄 Using REAL-TIME price: ${real_time_price:,.2f} (signal was ${current_price:,.2f})")

            # Calculate quantity based on leveraged position size and REAL-TIME price
            quantity = position_size_leveraged / real_time_price
            quantity = self._round_quantity(symbol, quantity)

            # Log detailed trade information BEFORE placing order
            logger.info("=" * 70)
            logger.info(f"🎯 OPENING {side} TRADE FOR {symbol}")
            logger.info("=" * 70)
            logger.info(f"📊 Position Details:")
            logger.info(f"   Real-time Price: ${real_time_price:,.2f}")
            logger.info(f"   Base Position: ${position_size_base:.2f} ({config.POSITION_SIZE_PERCENT}% of account)")
            logger.info(f"   Leveraged Position: ${position_size_leveraged:.2f} ({config.LEVERAGE}x)")
            logger.info(f"   Quantity: {quantity} {symbol.replace('USDT', '')}")
            logger.info("=" * 70)

            # Place market order
            logger.info(f"📤 Placing market order...")
            order = self.client.futures_create_order(
                symbol=symbol,
                side=position_side,
                type='MARKET',
                quantity=quantity
            )

            logger.info(f"✅ Order executed successfully!")
            logger.debug(f"Order details: {order}")

            # Get actual entry price from order
            actual_entry = float(order.get('avgPrice', real_time_price))
            if actual_entry == 0 or actual_entry is None:
                # If avgPrice not available, fetch current market price again
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                actual_entry = float(ticker['price'])

            logger.info(f"✅ Actual entry price: ${actual_entry:,.2f}")

            # Calculate TP and SL based on ACTUAL ENTRY PRICE
            if side == 'LONG':
                tp_price = actual_entry * (1 + (config.TAKE_PROFIT_PERCENT / 100))
                sl_price = actual_entry * (1 - (config.STOP_LOSS_PERCENT / 100))
            else:
                tp_price = actual_entry * (1 - (config.TAKE_PROFIT_PERCENT / 100))
                sl_price = actual_entry * (1 + (config.STOP_LOSS_PERCENT / 100))

            tp_price = self._round_price(symbol, tp_price)
            sl_price = self._round_price(symbol, sl_price)

            # Calculate expected PnL
            if side == 'LONG':
                price_change_tp = (tp_price - actual_entry) / actual_entry
                price_change_sl = (actual_entry - sl_price) / actual_entry
            else:
                price_change_tp = (actual_entry - tp_price) / actual_entry
                price_change_sl = (sl_price - actual_entry) / actual_entry

            expected_profit = price_change_tp * position_size_leveraged
            expected_loss = price_change_sl * position_size_leveraged
            risk_reward_ratio = expected_profit / expected_loss if expected_loss > 0 else 0

            # Log TP/SL details
            logger.info(f"")
            logger.info(f"🎯 Take Profit:")
            logger.info(f"   TP Price: ${tp_price:,.2f} (+{config.TAKE_PROFIT_PERCENT}% from entry)")
            logger.info(f"   Expected Profit: ${expected_profit:.2f}")
            logger.info(f"")
            logger.info(f"🛑 Stop Loss:")
            logger.info(f"   SL Price: ${sl_price:,.2f} (-{config.STOP_LOSS_PERCENT}% from entry)")
            logger.info(f"   Expected Loss: ${expected_loss:.2f}")
            logger.info(f"")
            logger.info(f"⚖️  Risk Management:")
            logger.info(f"   Risk/Reward Ratio: 1:{risk_reward_ratio:.2f}")
            logger.info(f"   Max Risk: {(expected_loss / self.account_manager.get_balance()) * 100:.2f}% of account")
            logger.info(f"   Max Gain: {(expected_profit / self.account_manager.get_balance()) * 100:.2f}% of account")
            logger.info("=" * 70)

            # Create trade object
            self.active_trade = Trade(
                symbol=symbol,
                side=side,
                entry_price=actual_entry,
                position_size=position_size_leveraged,
                tp_price=tp_price,
                sl_price=sl_price
            )

            # Place TP and SL orders
            tp_sl_success = await self._place_tp_sl_orders(symbol, side, quantity, tp_price, sl_price)

            if not tp_sl_success:
                logger.error("Trade aborted due to TP/SL placement failure")
                return False

            # Send Telegram notification
            await self.telegram.send_trade_opened(
                symbol=symbol,
                side=side,
                entry_price=actual_entry,
                position_size=position_size_leveraged,
                tp_price=tp_price,
                sl_price=sl_price
            )

            # Send detailed analysis to Telegram
            analysis_msg = (
                f"📊 <b>Trade Analysis ({side})</b>\n\n"
                f"Entry: ${actual_entry:,.2f}\n"
                f"Take Profit: ${tp_price:,.2f}\n"
                f"Stop Loss: ${sl_price:,.2f}\n\n"
                f"Expected Profit: ${expected_profit:.2f}\n"
                f"Expected Loss: ${expected_loss:.2f}\n"
                f"Risk/Reward: 1:{risk_reward_ratio:.2f}\n"
                f"Account Risk: {(expected_loss / self.account_manager.get_balance()) * 100:.2f}%"
            )
            await self.telegram.send_message(analysis_msg)

            logger.info(f"✅ {side} Trade opened successfully!")
            return True

        except BinanceAPIException as e:
            logger.error(f"❌ Binance API error opening trade: {e}")
            await self.telegram.send_error(f"Failed to open trade: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error opening trade: {e}")
            await self.telegram.send_error(f"Failed to open trade: {e}")
            return False

    async def _place_tp_sl_orders(self, symbol: str, side: str, quantity: float,
                                  tp_price: float, sl_price: float) -> bool:
        """
        Place take profit and stop loss orders with validation
        WORKS FOR BOTH LONG AND SHORT
        Returns True if successful, False if failed (and position was closed)
        """
        try:
            close_side = 'SELL' if side == 'LONG' else 'BUY'

            logger.info(f"📋 Placing TP/SL orders for {side} position...")

            # Get current market price to validate SL/TP
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            current_market_price = float(ticker['price'])

            logger.info(f"   Market price: ${current_market_price:,.2f}")
            logger.info(f"   TP: ${tp_price:,.2f}")
            logger.info(f"   SL: ${sl_price:,.2f}")

            adjustment_made = False

            if side == 'LONG':
                # For LONG: TP must be above current price, SL must be below
                if tp_price <= current_market_price:
                    logger.warning(f"⚠️  TP ${tp_price:,.2f} too close to market ${current_market_price:,.2f}")
                    tp_price = current_market_price * (1 + (config.TAKE_PROFIT_PERCENT / 100) + 0.001)
                    tp_price = self._round_price(symbol, tp_price)
                    logger.info(f"   ✓ Adjusted TP to ${tp_price:,.2f}")
                    adjustment_made = True

                if sl_price >= current_market_price:
                    logger.warning(f"⚠️  SL ${sl_price:,.2f} too close to market ${current_market_price:,.2f}")
                    sl_price = current_market_price * (1 - (config.STOP_LOSS_PERCENT / 100) - 0.001)
                    sl_price = self._round_price(symbol, sl_price)
                    logger.info(f"   ✓ Adjusted SL to ${sl_price:,.2f}")
                    adjustment_made = True
            else:  # SHORT
                # For SHORT: TP must be BELOW current price, SL must be ABOVE
                if tp_price >= current_market_price:
                    logger.warning(f"⚠️  SHORT TP ${tp_price:,.2f} too close to market ${current_market_price:,.2f}")
                    tp_price = current_market_price * (1 - (config.TAKE_PROFIT_PERCENT / 100) - 0.001)
                    tp_price = self._round_price(symbol, tp_price)
                    logger.info(f"   ✓ Adjusted SHORT TP to ${tp_price:,.2f}")
                    adjustment_made = True

                if sl_price <= current_market_price:
                    logger.warning(f"⚠️  SHORT SL ${sl_price:,.2f} too close to market ${current_market_price:,.2f}")
                    sl_price = current_market_price * (1 + (config.STOP_LOSS_PERCENT / 100) + 0.001)
                    sl_price = self._round_price(symbol, sl_price)
                    logger.info(f"   ✓ Adjusted SHORT SL to ${sl_price:,.2f}")
                    adjustment_made = True

            # Update trade object with final prices
            if self.active_trade and adjustment_made:
                self.active_trade.tp_price = tp_price
                self.active_trade.sl_price = sl_price

            # Place Take Profit order
            try:
                tp_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=close_side,
                    type='TAKE_PROFIT_MARKET',
                    stopPrice=tp_price,
                    closePosition=True
                )
                logger.info(f"✅ TP order ({close_side}) placed at ${tp_price:,.2f}")
                logger.debug(f"TP order: {tp_order}")
            except BinanceAPIException as e:
                logger.error(f"❌ Failed to place TP order: {e}")
                raise

            # Place Stop Loss order
            try:
                sl_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=close_side,
                    type='STOP_MARKET',
                    stopPrice=sl_price,
                    closePosition=True
                )
                logger.info(f"✅ SL order ({close_side}) placed at ${sl_price:,.2f}")
                logger.debug(f"SL order: {sl_order}")
            except BinanceAPIException as e:
                logger.error(f"❌ Failed to place SL order: {e}")
                # If SL fails but TP succeeded, cancel TP
                try:
                    logger.warning("Attempting to cancel TP order...")
                    open_orders = self.client.futures_get_open_orders(symbol=symbol)
                    for order in open_orders:
                        if order['type'] == 'TAKE_PROFIT_MARKET':
                            self.client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                            logger.info("TP order cancelled")
                except:
                    pass
                raise

            logger.info(f"✅ TP/SL orders successfully placed for {side} position!")
            return True

        except Exception as e:
            logger.error(f"❌ Error placing TP/SL orders: {e}")

            # Try to close position if TP/SL failed
            try:
                logger.warning("⚠️  TP/SL placement failed, closing position for safety...")
                close_side = 'SELL' if side == 'LONG' else 'BUY'

                close_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=close_side,
                    type='MARKET',
                    quantity=quantity,
                    reduceOnly=True
                )
                logger.info(f"✅ {side} position closed for safety")

                await self.telegram.send_error(
                    f"⚠️ <b>TP/SL Failed - Position Closed</b>\n\n"
                    f"Symbol: {symbol}\n"
                    f"Side: {side}\n"
                    f"Reason: {str(e)}\n"
                    f"Action: Position closed for safety"
                )

                # Clear active trade
                self.active_trade = None

                return False

            except Exception as e2:
                logger.error(f"❌ CRITICAL: Failed to close position after TP/SL error: {e2}")
                await self.telegram.send_error(
                    f"🚨 <b>CRITICAL ALERT</b>\n\n"
                    f"Symbol: {symbol}\n"
                    f"Side: {side}\n"
                    f"TP/SL Error: {str(e)}\n"
                    f"Close Error: {str(e2)}\n\n"
                    f"⚠️ Please close position manually IMMEDIATELY!"
                )
                return False

    async def monitor_active_trade(self, current_price: float):
        """
        Monitor active trade and close if TP or SL hit
        WORKS FOR BOTH LONG AND SHORT
        """
        if not self.has_active_trade():
            return

        try:
            trade = self.active_trade

            # Check if TP or SL hit
            if trade.side == 'LONG':
                # LONG: TP above, SL below
                if current_price >= trade.tp_price:
                    logger.info(f"🎯 Take Profit hit for LONG {trade.symbol}!")
                    await self._close_trade(current_price, 'TP')
                elif current_price <= trade.sl_price:
                    logger.info(f"🛑 Stop Loss hit for LONG {trade.symbol}!")
                    await self._close_trade(current_price, 'SL')
            else:  # SHORT
                # SHORT: TP below, SL above (opposite of LONG)
                if current_price <= trade.tp_price:
                    logger.info(f"🎯 Take Profit hit for SHORT {trade.symbol}!")
                    await self._close_trade(current_price, 'TP')
                elif current_price >= trade.sl_price:
                    logger.info(f"🛑 Stop Loss hit for SHORT {trade.symbol}!")
                    await self._close_trade(current_price, 'SL')

        except Exception as e:
            logger.error(f"Error monitoring trade: {e}")

    async def _close_trade(self, exit_price: float, reason: str):
        """
        Close the active trade
        CALCULATES PNL CORRECTLY FOR BOTH LONG AND SHORT
        """
        try:
            trade = self.active_trade
            trade.exit_price = exit_price
            trade.exit_time = datetime.now()
            trade.close_reason = reason
            trade.status = 'CLOSED'

            # Calculate PnL correctly
            if trade.side == 'LONG':
                # LONG: profit when exit > entry
                price_diff = exit_price - trade.entry_price
            else:  # SHORT
                # SHORT: profit when entry > exit (sell high, buy low)
                price_diff = trade.entry_price - exit_price

            trade.pnl = (price_diff / trade.entry_price) * trade.position_size

            # Calculate percentage gains
            pnl_percent = (trade.pnl / self.account_manager.get_balance()) * 100
            price_move_percent = ((exit_price - trade.entry_price) / trade.entry_price) * 100

            duration = (trade.exit_time - trade.entry_time).total_seconds()
            duration_str = f"{int(duration // 60)}m {int(duration % 60)}s"

            logger.info("=" * 70)
            logger.info(f"🔔 {trade.side} TRADE CLOSED - {reason}")
            logger.info("=" * 70)
            logger.info(f"Symbol: {trade.symbol}")
            logger.info(f"Side: {trade.side}")
            logger.info(f"Entry: ${trade.entry_price:,.2f}")
            logger.info(f"Exit: ${exit_price:,.2f}")
            logger.info(f"Price Move: {price_move_percent:+.2f}%")
            logger.info(f"PnL: ${trade.pnl:+.2f} ({pnl_percent:+.2f}% of account)")
            logger.info(f"Duration: {duration_str}")
            logger.info("=" * 70)

            # Update account balance
            await self.account_manager.update_balance()

            # Send Telegram notification
            await self.telegram.send_trade_closed(
                symbol=trade.symbol,
                side=trade.side,
                entry_price=trade.entry_price,
                exit_price=exit_price,
                pnl=trade.pnl,
                balance=self.account_manager.get_balance(),
                reason=reason
            )

            # Add to history
            self.trade_history.append(trade)
            self.active_trade = None

            logger.info(f"✅ {trade.side} trade logged to history. Total trades: {len(self.trade_history)}")

        except Exception as e:
            logger.error(f"Error closing trade: {e}")

    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to valid precision"""
        try:
            info = self.client.futures_exchange_info()
            for s in info['symbols']:
                if s['symbol'] == symbol:
                    for f in s['filters']:
                        if f['filterType'] == 'LOT_SIZE':
                            step_size = float(f['stepSize'])
                            precision = len(str(step_size).rstrip('0').split('.')[-1])
                            return round(quantity - (quantity % step_size), precision)
        except Exception as e:
            logger.warning(f"Could not get quantity precision for {symbol}: {e}")
        return round(quantity, 3)

    def _round_price(self, symbol: str, price: float) -> float:
        """Round price to valid precision"""
        try:
            info = self.client.futures_exchange_info()
            for s in info['symbols']:
                if s['symbol'] == symbol:
                    for f in s['filters']:
                        if f['filterType'] == 'PRICE_FILTER':
                            tick_size = float(f['tickSize'])
                            precision = len(str(tick_size).rstrip('0').split('.')[-1])
                            return round(price - (price % tick_size), precision)
        except Exception as e:
            logger.warning(f"Could not get price precision for {symbol}: {e}")
        return round(price, 2)

    def get_trade_stats(self) -> dict:
        """Get trading statistics"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }

        winning = [t for t in self.trade_history if t.pnl > 0]
        losing = [t for t in self.trade_history if t.pnl < 0]

        total_pnl = sum(t.pnl for t in self.trade_history)
        avg_win = sum(t.pnl for t in winning) / len(winning) if winning else 0.0
        avg_loss = sum(t.pnl for t in losing) / len(losing) if losing else 0.0
        largest_win = max((t.pnl for t in winning), default=0.0)
        largest_loss = min((t.pnl for t in losing), default=0.0)

        return {
            'total_trades': len(self.trade_history),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': (len(winning) / len(self.trade_history)) * 100 if self.trade_history else 0.0,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }