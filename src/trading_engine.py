"""
交易执行模块
负责订单执行、仓位管理和策略触发
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """订单执行结果"""
    success: bool
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    price: float = 0.0
    message: str = ""


class TradingEngine:
    """交易执行引擎"""

    def __init__(self, config: dict, binance_fetcher):
        self.config = config
        self.binance = binance_fetcher
        self.pending_orders: Dict[str, dict] = {}  # order_id -> order_info
        self.executed_orders: List[dict] = []
        self.trading_fee = config["trading"]["trading_fee"]

    def execute_buy(self, symbol: str, quantity: float, price: float,
                   order_type: str = "LIMIT") -> OrderResult:
        """
        执行买入

        Args:
            symbol: 交易对
            quantity: 数量
            price: 价格
            order_type: LIMIT 或 MARKET

        Returns:
            OrderResult
        """
        try:
            logger.info(f"执行买入: {symbol} | 数量: {quantity} | 价格: {price} | 类型: {order_type}")

            if self.config["runtime"].get("backtest_mode", False):
                # 回测模式：模拟成交
                logger.info(f"[回测] 买入订单模拟成交: {symbol}")
                return OrderResult(
                    success=True,
                    order_id=f"backtest_{int(time.time())}",
                    symbol=symbol,
                    side="BUY",
                    quantity=quantity,
                    price=price,
                    message="回测模拟成交"
                )

            result = self.binance.place_order(symbol, "BUY", order_type, quantity, price)

            if "orderId" in result and result["orderId"] > 0:
                order_record = {
                    "order_id": str(result["orderId"]),
                    "symbol": symbol,
                    "side": "BUY",
                    "quantity": quantity,
                    "price": price,
                    "timestamp": datetime.now(),
                    "status": "filled"
                }
                self.pending_orders[str(result["orderId"])] = order_record
                self.executed_orders.append(order_record)

                return OrderResult(
                    success=True,
                    order_id=str(result["orderId"]),
                    symbol=symbol,
                    side="BUY",
                    quantity=quantity,
                    price=price,
                    message="下单成功"
                )
            else:
                return OrderResult(
                    success=False,
                    symbol=symbol,
                    side="BUY",
                    message=str(result.get("msg", "下单失败"))
                )

        except Exception as e:
            logger.error(f"买入执行失败: {e}")
            return OrderResult(success=False, symbol=symbol, side="BUY", message=str(e))

    def execute_sell(self, symbol: str, quantity: float, price: float,
                    order_type: str = "LIMIT") -> OrderResult:
        """
        执行卖出

        Args:
            symbol: 交易对
            quantity: 数量
            price: 价格
            order_type: LIMIT 或 MARKET

        Returns:
            OrderResult
        """
        try:
            logger.info(f"执行卖出: {symbol} | 数量: {quantity} | 价格: {price} | 类型: {order_type}")

            if self.config["runtime"].get("backtest_mode", False):
                logger.info(f"[回测] 卖出订单模拟成交: {symbol}")
                return OrderResult(
                    success=True,
                    order_id=f"backtest_{int(time.time())}",
                    symbol=symbol,
                    side="SELL",
                    quantity=quantity,
                    price=price,
                    message="回测模拟成交"
                )

            result = self.binance.place_order(symbol, "SELL", order_type, quantity, price)

            if "orderId" in result and result["orderId"] > 0:
                order_record = {
                    "order_id": str(result["orderId"]),
                    "symbol": symbol,
                    "side": "SELL",
                    "quantity": quantity,
                    "price": price,
                    "timestamp": datetime.now(),
                    "status": "filled"
                }
                self.pending_orders[str(result["orderId"])] = order_record
                self.executed_orders.append(order_record)

                return OrderResult(
                    success=True,
                    order_id=str(result["orderId"]),
                    symbol=symbol,
                    side="SELL",
                    quantity=quantity,
                    price=price,
                    message="下单成功"
                )
            else:
                return OrderResult(
                    success=False,
                    symbol=symbol,
                    side="SELL",
                    message=str(result.get("msg", "下单失败"))
                )

        except Exception as e:
            logger.error(f"卖出执行失败: {e}")
            return OrderResult(success=False, symbol=symbol, side="SELL", message=str(e))

    def get_current_price(self, symbol: str) -> float:
        """获取当前市场价格"""
        return self.binance.get_symbol_price(symbol)

    def get_all_prices(self) -> Dict[str, float]:
        """获取所有交易对当前价格"""
        return self.binance.get_all_prices()

    def get_order_status(self, order_id: str) -> dict:
        """查询订单状态"""
        if order_id.startswith("backtest_"):
            return {"status": "filled"}
        return self.pending_orders.get(order_id, {})

    def get_trade_history(self) -> List[dict]:
        """获取交易历史"""
        return self.executed_orders.copy()


class StrategyEngine:
    """策略执行引擎"""

    def __init__(self, config: dict, trading_engine: TradingEngine,
                 position_manager, model_manager):
        self.config = config
        self.trading_engine = trading_engine
        self.position_manager = position_manager
        self.model_manager = model_manager

    def generate_signals(self, symbol: str) -> dict:
        """
        生成交易信号

        Args:
            symbol: 交易对

        Returns:
            信号字典
        """
        if symbol not in self.model_manager.models:
            logger.warning(f"{symbol} 无可用模型，返回HOLD信号")
            return {"signal": "HOLD", "confidence": 0, "symbol": symbol}

        try:
            # 获取最新数据
            prediction = self.model_manager.models[symbol].predict(
                self.model_manager.data_fetcher.get_latest_data(symbol)
            )
            return {**prediction, "symbol": symbol}
        except Exception as e:
            logger.error(f"信号生成失败 {symbol}: {e}")
            return {"signal": "HOLD", "confidence": 0, "symbol": symbol, "error": str(e)}

    def execute_strategy(self) -> dict:
        """
        执行策略

        Returns:
            执行结果
        """
        symbols = self.config["symbols"]["primary_pairs"]
        actions = []
        prices = {}

        # 1. 获取所有币种最新价格
        all_prices = self.trading_engine.get_all_prices()
        for sym in symbols:
            prices[sym] = all_prices.get(sym, self.trading_engine.get_current_price(sym))

        # 2. 检查现有持仓状态
        close_signals = self.position_manager.update_positions(prices)

        # 3. 执行平仓信号
        for close_sig in close_signals:
            sym = close_sig["symbol"]
            result = self.trading_engine.execute_sell(
                sym,
                self.position_manager.positions[sym].quantity,
                close_sig["price"],
                order_type="MARKET"
            )
            if result.success:
                self.position_manager.close_position(sym, close_sig["price"], close_sig["reason"])
                actions.append(f"平仓 {sym}: {close_sig['reason']}")

        # 4. 生成新信号并决策
        for sym in symbols:
            if sym in self.position_manager.positions:
                continue  # 已在持仓，跳过

            signal_data = self.generate_signals(sym)
            sig = signal_data.get("signal", "HOLD")
            conf = signal_data.get("confidence", 0)

            if sig == "BUY" and conf >= self.config["model"]["buy_threshold"]:
                # 计算买入数量
                price = prices[sym]
                balance = self.trading_engine.binance.get_balance("USDT")
                max_pct = self.config["trading"]["max_position_size"]
                invest_amount = balance * max_pct
                quantity = invest_amount / price

                if quantity * price < self.config["trading"]["min_trade_amount"]:
                    continue

                # 开仓
                result = self.trading_engine.execute_buy(sym, quantity, price)
                if result.success:
                    pos = self.position_manager.open_position(sym, quantity, price, signal="LSTM")
                    if pos:
                        actions.append(f"开仓 {sym}: LSTM信号 BUY (置信度: {conf:.2%})")

            elif sig == "SELL":
                logger.info(f"{sym} 持有多头，生成SELL信号（模型建议）")

        return {
            "actions": actions,
            "signals": {sym: self.generate_signals(sym) for sym in symbols},
            "prices": prices
        }

    def run_backtest(self, symbols: List[str], start, end) -> dict:
        """
        运行回测

        Args:
            symbols: 交易对列表
            start: 开始时间
            end: 结束时间

        Returns:
            回测结果
        """
        logger.info(f"开始回测: {start} - {end}")
        # 回测逻辑：遍历时间，检查信号，执行交易
        # 这里返回模拟结果
        results = {
            "total_return": 0.0,
            "num_trades": 0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
        }
        logger.info(f"回测完成: {results}")
        return results
