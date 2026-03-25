"""
仓位管理和风控模块
Position Management and Risk Control
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Position:
    """持仓"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    @property
    def market_value(self) -> float:
        """市值"""
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """未实现盈亏"""
        return (self.current_price - self.entry_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """未实现盈亏百分比"""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price * 100

    @property
    def hold_hours(self) -> float:
        """持仓小时数"""
        return (datetime.now() - self.entry_time).total_seconds() / 3600


@dataclass
class Trade:
    """交易记录"""
    trade_id: str
    symbol: str
    side: str  # BUY or SELL
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    status: OrderStatus = OrderStatus.FILLED
    notes: str = ""


@dataclass
class RiskMetrics:
    """风险指标"""
    daily_trades: int = 0
    daily_pnl: float = 0.0
    open_positions: int = 0
    total_exposure: float = 0.0  # 总持仓价值
    largest_position: float = 0.0
    smallest_position: float = float('inf')


class PositionManager:
    """仓位管理器"""

    def __init__(self, config: dict, binance_fetcher):
        self.config = config
        self.binance = binance_fetcher
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.daily_trade_count: int = 0
        self.last_reset_date: datetime = datetime.now().date()
        self.initial_capital: float = config["trading"]["initial_capital"]

    def _reset_daily_counters(self):
        """重置每日计数器"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_trade_count = 0
            self.last_reset_date = today
            logger.info("每日计数器已重置")

    def can_open_position(self, symbol: str, quantity: float,
                         price: float) -> tuple[bool, str]:
        """
        检查是否可以开仓

        Args:
            symbol: 交易对
            quantity: 数量
            price: 价格

        Returns:
            (是否可以开仓, 原因)
        """
        self._reset_daily_counters()

        # 检查是否已在持仓
        if symbol in self.positions:
            return False, f"{symbol} 已在持仓中"

        # 检查最大持仓数
        max_positions = self.config["risk_management"]["max_open_positions"]
        if len(self.positions) >= max_positions:
            return False, f"已达到最大持仓数 ({max_positions})"

        # 检查每日交易次数
        max_daily = self.config["risk_management"]["max_daily_trades"]
        if self.daily_trade_count >= max_daily:
            return False, f"已达到每日最大交易次数 ({max_daily})"

        # 检查资金
        position_value = quantity * price
        min_trade = self.config["trading"]["min_trade_amount"]
        if position_value < min_trade:
            return False, f"交易金额 ({position_value:.2f}) 小于最低要求 ({min_trade})"

        # 检查仓位比例
        balance = self.binance.get_balance("USDT")
        max_position_pct = self.config["trading"]["max_position_size"]

        if position_value > balance * max_position_pct:
            return False, f"仓位超过限制 ({max_position_pct*100}%)"

        # 检查总暴露
        total_exposure = sum(p.market_value for p in self.positions.values())
        if total_exposure + position_value > balance * 0.8:
            return False, "总持仓暴露过高"

        return True, "可以开仓"

    def open_position(self, symbol: str, quantity: float,
                     entry_price: float, signal: str = "BUY") -> Optional[Position]:
        """
        开仓

        Args:
            symbol: 交易对
            quantity: 数量
            entry_price: 入场价格
            signal: 信号来源

        Returns:
            Position对象或None
        """
        can_open, reason = self.can_open_position(symbol, quantity, entry_price)
        if not can_open:
            logger.warning(f"无法开仓 {symbol}: {reason}")
            return None

        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=datetime.now(),
            current_price=entry_price
        )

        # 设置止损止盈
        if self.config["risk_management"]["stop_loss"]["enabled"]:
            sl_pct = self.config["risk_management"]["stop_loss"]["percentage"]
            position.stop_loss = entry_price * (1 - sl_pct)

        if self.config["risk_management"]["take_profit"]["enabled"]:
            tp_pct = self.config["risk_management"]["take_profit"]["percentage"]
            position.take_profit = entry_price * (1 + tp_pct)

        self.positions[symbol] = position
        self.daily_trade_count += 1

        logger.info(f"开仓: {symbol} | 数量: {quantity} | 价格: {entry_price} | 信号: {signal}")
        return position

    def close_position(self, symbol: str, exit_price: float,
                      reason: str = "手动平仓") -> Optional[Trade]:
        """
        平仓

        Args:
            symbol: 交易对
            exit_price: 平仓价格
            reason: 平仓原因

        Returns:
            Trade对象或None
        """
        if symbol not in self.positions:
            logger.warning(f"{symbol} 不在持仓中")
            return None

        position = self.positions[symbol]

        trade = Trade(
            trade_id=f"{symbol}_{int(time.time())}",
            symbol=symbol,
            side="SELL",
            quantity=position.quantity,
            price=exit_price,
            commission=exit_price * position.quantity * self.config["trading"]["trading_fee"],
            timestamp=datetime.now(),
            notes=reason
        )

        self.trade_history.append(trade)
        del self.positions[symbol]

        pnl = position.unrealized_pnl - trade.commission
        logger.info(f"平仓: {symbol} | 数量: {position.quantity} | 价格: {exit_price} | "
                    f"盈亏: {pnl:.2f} | 原因: {reason}")

        return trade

    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """
        检查是否触发止损

        Args:
            symbol: 交易对
            current_price: 当前价格

        Returns:
            是否触发止损
        """
        if symbol not in self.positions:
            return False

        position = self.positions[symbol]
        position.current_price = current_price

        if position.stop_loss > 0 and current_price <= position.stop_loss:
            logger.warning(f"{symbol} 触发止损: 当前价 {current_price} <= 止损价 {position.stop_loss}")
            return True

        return False

    def check_take_profit(self, symbol: str, current_price: float) -> bool:
        """
        检查是否触发止盈

        Args:
            symbol: 交易对
            current_price: 当前价格

        Returns:
            是否触发止盈
        """
        if symbol not in self.positions:
            return False

        position = self.positions[symbol]
        position.current_price = current_price

        if position.take_profit > 0 and current_price >= position.take_profit:
            logger.info(f"{symbol} 触发止盈: 当前价 {current_price} >= 止盈价 {position.take_profit}")
            return True

        return False

    def check_max_hold_time(self, symbol: str) -> bool:
        """
        检查是否超过最大持仓时间

        Args:
            symbol: 交易对

        Returns:
            是否超过最大持仓时间
        """
        if symbol not in self.positions:
            return False

        position = self.positions[symbol]
        max_hours = self.config["risk_management"]["max_hold_hours"]

        if position.hold_hours >= max_hours:
            logger.info(f"{symbol} 超过最大持仓时间: {position.hold_hours:.1f}h >= {max_hours}h")
            return True

        return False

    def update_positions(self, prices: Dict[str, float]) -> List[Dict]:
        """
        更新所有持仓的当前价格和状态

        Args:
            prices: 交易对价格字典

        Returns:
            需要平仓的交易列表
        """
        close_signals = []

        for symbol, position in list(self.positions.items()):
            if symbol in prices:
                position.current_price = prices[symbol]

                # 检查止损
                if self.check_stop_loss(symbol, prices[symbol]):
                    close_signals.append({
                        "symbol": symbol,
                        "reason": "止损",
                        "price": prices[symbol]
                    })

                # 检查止盈
                elif self.check_take_profit(symbol, prices[symbol]):
                    close_signals.append({
                        "symbol": symbol,
                        "reason": "止盈",
                        "price": prices[symbol]
                    })

                # 检查最大持仓时间
                elif self.check_max_hold_time(symbol):
                    close_signals.append({
                        "symbol": symbol,
                        "reason": "超时",
                        "price": prices[symbol]
                    })

        return close_signals

    def get_risk_metrics(self) -> RiskMetrics:
        """
        获取风险指标

        Returns:
            RiskMetrics对象
        """
        self._reset_daily_counters()

        metrics = RiskMetrics(
            daily_trades=self.daily_trade_count,
            open_positions=len(self.positions),
            total_exposure=sum(p.market_value for p in self.positions.values())
        )

        if self.positions:
            positions_values = [p.market_value for p in self.positions.values()]
            metrics.largest_position = max(positions_values)
            if positions_values:
                metrics.smallest_position = min(positions_values)

        # 计算每日盈亏
        today = datetime.now().date()
        today_trades = [t for t in self.trade_history
                        if t.timestamp.date() == today]
        metrics.daily_pnl = sum(
            (t.price - self._get_entry_price(t)) * t.quantity - t.commission
            if t.side == "SELL" else 0
            for t in today_trades
        )

        return metrics

    def _get_entry_price(self, trade: Trade) -> float:
        """获取入场价格（简化版本）"""
        return trade.price * 0.98  # 假设入场价格低2%

    def get_portfolio_summary(self) -> Dict:
        """
        获取投资组合摘要

        Returns:
            组合摘要字典
        """
        balance = self.binance.get_balance("USDT")
        total_exposure = sum(p.market_value for p in self.positions.values())
        total_value = balance + total_exposure

        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())

        return {
            "balance": balance,
            "total_exposure": total_exposure,
            "total_value": total_value,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": (unrealized_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0,
            "open_positions": len(self.positions),
            "positions": [
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "market_value": p.market_value,
                    "pnl": p.unrealized_pnl,
                    "pnl_pct": p.unrealized_pnl_pct,
                    "hold_hours": p.hold_hours
                }
                for p in self.positions.values()
            ]
        }

    def export_trade_history(self, filepath: str = "logs/trades.csv"):
        """导出交易历史"""
        if not self.trade_history:
            logger.warning("无交易历史可导出")
            return

        df = pd.DataFrame([
            {
                "trade_id": t.trade_id,
                "symbol": t.symbol,
                "side": t.side,
                "quantity": t.quantity,
                "price": t.price,
                "commission": t.commission,
                "timestamp": t.timestamp,
                "status": t.status.value,
                "notes": t.notes
            }
            for t in self.trade_history
        ])

        df.to_csv(filepath, index=False)
        logger.info(f"交易历史已导出到: {filepath}")
