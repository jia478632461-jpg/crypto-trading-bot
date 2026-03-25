"""
===================================================================
回测引擎 (Backtesting Engine)
支持多币种、多策略、事件驱动、完整风控评估
===================================================================
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import json
import urllib.request
import urllib.error

from src.analytics_engine import TechnicalAnalyzer, MarketRegimeDetector, MarketRegime
from src.game_theory import StatisticalModels, GameTheoryAnalyzer

logger = logging.getLogger(__name__)


# ========== 数据类型 ==========
@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    side: str            # "LONG" / "SHORT"
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    fees: float
    signal: str
    regime: str
    holding_bars: int


@dataclass
class EquityPoint:
    timestamp: datetime
    equity: float
    drawdown: float


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    calmar_ratio: float
    win_loss_ratio: float
    avg_bars_held: float
    trades: List[Trade]
    equity_curve: List[EquityPoint]
    monthly_returns: Dict[str, float]
    regime_stats: Dict[str, Dict[str, float]]
    summary: Dict[str, Any]

    def print_summary(self):
        print("=" * 60)
        print(f"  回测报告: {self.strategy_name} | {self.symbol}")
        print("=" * 60)
        print(f"  周期: {self.start_date} → {self.end_date}")
        print(f"  总交易次数: {self.total_trades}   胜率: {self.win_rate:.1%}")
        print(f"  总收益率: {self.total_return:+.1%}  年化: {self.annualized_return:+.1%}")
        print(f"  Sharpe: {self.sharpe_ratio:+.2f}  Sortino: {self.sortino_ratio:+.2f}")
        print(f"  Calmar: {self.calmar_ratio:+.2f}  Profit Factor: {self.profit_factor:.2f}")
        print(f"  最大回撤: {self.max_drawdown:.1f} ({self.max_drawdown_pct:.1%})")
        print(f"  平均持仓: {self.avg_bars_held:.1f} 根K线")
        print(f"  平均盈利: {self.avg_win:.2f} | 平均亏损: {self.avg_loss:.2f}")
        print("-" * 60)


# ========== 历史数据获取 ==========
class BinanceHistoryFetcher:
    """从 Binance 公开 API 获取历史K线"""

    BASE = "https://api.binance.com/api/v3/klines"

    @staticmethod
    def fetch(
        symbol: str,
        interval: str = "1h",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        获取历史K线数据

        Args:
            symbol: e.g. "BTCUSDT"
            interval: "1m","5m","15m","1h","4h","1d","1w"
            start_time / end_time: Unix ms timestamps
            limit: 1~1000
        """
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{BinanceHistoryFetcher.BASE}?{query}"

        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = json.loads(resp.read().decode())
        except Exception as e:
            logger.error(f"获取Binance历史数据失败: {e}")
            return pd.DataFrame()

        rows = []
        for k in raw:
            rows.append({
                "open_time": pd.to_datetime(k[0], unit="ms"),
                "open": float(k[1]), "high": float(k[2]),
                "low": float(k[3]), "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": pd.to_datetime(k[6], unit="ms"),
                "quote_volume": float(k[7]),
                "trades": int(k[8]),
                "taker_buy_base": float(k[9]),
                "taker_buy_quote": float(k[10]),
            })

        df = pd.DataFrame(rows)
        df.set_index("open_time", inplace=True)
        return df

    @staticmethod
    def fetch_range(
        symbol: str,
        interval: str = "1h",
        days: int = 90,
    ) -> pd.DataFrame:
        """自动分页获取指定天数"""
        end = datetime.now()
        start = end - timedelta(days=days)
        end_ms = int(end.timestamp() * 1000)
        start_ms = int(start.timestamp() * 1000)

        all_dfs = []
        current_start = start_ms

        for _ in range(10):  # 最多10次请求
            chunk = BinanceHistoryFetcher.fetch(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_ms,
                limit=1000,
            )
            if chunk.empty:
                break
            all_dfs.append(chunk)
            # 取下一批
            next_start = int(chunk.index[-1].timestamp() * 1000) + 1
            if next_start >= end_ms or len(chunk) < 1000:
                break
            current_start = next_start

        if not all_dfs:
            return pd.DataFrame()
        return pd.concat(all_dfs).drop_duplicates()


# ========== 信号生成器 ==========
class SignalGenerator:
    """使用 UnifiedAnalyticsEngine 生成交易信号"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.ta = TechnicalAnalyzer()
        self.regime_detector = MarketRegimeDetector()
        self.stats = StatisticalModels()
        self.gt = GameTheoryAnalyzer()

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对历史DataFrame生成每日信号

        Returns DataFrame with columns:
            signal (1=LONG, -1=SHORT, 0=HOLD),
            confidence (0~1),
            regime,
            score,
            indicators (dict)
        """
        df = df.copy()
        n = len(df)
        if n < 30:
            df["signal"] = 0
            df["confidence"] = 0.0
            return df

        # 逐根K线生成信号（使用前n根计算指标，用于当前信号）
        signals = []
        confidences = []
        regimes = []
        scores = []

        for i in range(n):
            if i < 20:
                signals.append(0)
                confidences.append(0.0)
                regimes.append("unknown")
                scores.append(0.0)
                continue

            window = df.iloc[: i + 1]
            c = window["close"].values
            h = window["high"].values
            lo = window["low"].values
            v = window["volume"].values if "volume" in window.columns else np.ones_like(c)

            # 技术指标
            try:
                ind = self.ta.compute_all(pd.DataFrame({
                    "close": c, "high": h, "low": lo, "volume": v
                }))
                tech_score, _ = self.ta.score_technical(ind)
            except Exception:
                tech_score = 0.0

            # 市场状态
            try:
                mini_df = pd.DataFrame({
                    "close": c[-200:], "high": h[-200:], "low": lo[-200:],
                    "volume": v[-200:]
                })
                regime, _ = self.regime_detector.detect_regime(mini_df)
                regime_val = regime.value
            except Exception:
                regime_val = "unknown"

            # 统计模型
            try:
                z = self.stats.z_score(c)
                mr_signal, mr_conf = self.stats.mean_reversion_signal(c)
                hurst = self.stats.hurst_exponent(c)
                stats_score = np.tanh((z + hurst * 2) / 5) if hurst < 0.6 else 0.0
            except Exception:
                stats_score = 0.0

            # 博弈论
            try:
                mini_df2 = pd.DataFrame({
                    "close": c[-100:], "high": h[-100:], "low": lo[-100:],
                    "volume": v[-100:]
                })
                gt_result = self.gt.analyze_market_game(mini_df2)
                gt_score = gt_result.get("overall_game_score", 0.0)
            except Exception:
                gt_score = 0.0

            # 综合评分
            total_score = (
                tech_score * 0.45 +
                stats_score * 0.25 +
                gt_score * 0.30
            )
            scores.append(total_score)

            # 信号 + 置信度
            if regime_val in ("bull_trend", "breakout") and total_score > 0.15:
                sig, conf = 1, min(1.0, abs(total_score) * 2)
            elif regime_val in ("bear_trend", "high_volatility") and total_score < -0.15:
                sig, conf = -1, min(1.0, abs(total_score) * 2)
            elif abs(total_score) > 0.25:
                sig, conf = (1 if total_score > 0 else -1), min(1.0, abs(total_score))
            else:
                sig, conf = 0, 0.0

            signals.append(sig)
            confidences.append(conf)
            regimes.append(regime_val)

        df["signal"] = signals
        df["confidence"] = confidences
        df["regime"] = regimes
        df["score"] = scores
        return df


# ========== 核心回测器 ==========
class Backtester:
    """
    向量化 + 事件驱动混合回测引擎

    Features:
    - 支持多空双向
    - 固定仓位 + 动态仓位（ATR/Kelly）
    - 滑点模拟 (0.05%~0.2%)
    - 交易手续费（Maker/Taker）
    - 分币种统计
    - 月度收益表
    - 状态统计
    - 图表输出（生成 HTML）
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.initial_capital = float(self.config.get("initial_capital", 10000))
        self.fee = float(self.config.get("fee", 0.001))      # 0.1% per side
        self.slippage = float(self.config.get("slippage", 0.0005))  # 0.05%
        self.max_position_pct = float(self.config.get("max_position_pct", 0.95))
        self.stop_loss_pct = float(self.config.get("stop_loss", 0.05))
        self.take_profit_pct = float(self.config.get("take_profit", 0.15))

        self.fetcher = BinanceHistoryFetcher()
        self.signal_gen = SignalGenerator()

    @staticmethod
    def generate_synthetic_data(
        symbol: str = "BTCUSDT",
        days: int = 90,
        interval: str = "1h",
        trend: str = "mixed",
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        生成合成价格数据（当Binance不可访问时使用）

        trend: "bull" / "bear" / "sideways" / "volatile" / "mixed"
        """
        np.random.seed(seed)
        interval_minutes = {
            "1m": 1, "5m": 5, "15m": 15, "1h": 60,
            "4h": 240, "1d": 1440, "1w": 10080,
        }.get(interval, 60)
        n_bars = int(days * 24 * 60 / interval_minutes)

        # 基础价格和参数
        base_price = 50000 if "BTC" in symbol.upper() else (3000 if "ETH" in symbol.upper() else 100)
        vol = 0.03 if trend == "volatile" else (0.015 if trend in ("bull", "bear") else 0.02)
        drift = 0.0002 if trend == "bull" else (-0.0002 if trend == "bear" else 0.0)

        returns = np.random.randn(n_bars) * vol + drift
        close = base_price * np.cumprod(1 + returns)
        high = close * (1 + np.abs(np.random.randn(n_bars) * vol * 0.5))
        low = close * (1 - np.abs(np.random.randn(n_bars) * vol * 0.5))
        open_prices = close * (1 + np.random.randn(n_bars) * vol * 0.2)
        volume = np.random.lognormal(15, 1.0, n_bars).astype(float)

        start = pd.Timestamp.now() - timedelta(days=days)
        dates = pd.date_range(start=start, periods=n_bars, freq=f"{interval_minutes}min")
        # 清理日期到合理范围
        dates = pd.bdate_range(start=dates[0], periods=n_bars, freq=f"{interval_minutes}min")

        df = pd.DataFrame({
            "open": open_prices, "high": high, "low": low,
            "close": close, "volume": volume,
        }, index=dates)
        return df

    def run(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        days: int = 90,
        signal_config: Optional[Dict] = None,
    ) -> BacktestResult:
        """
        运行完整回测

        Args:
            symbol: 交易对
            interval: K线周期
            days: 历史数据天数
            signal_config: 信号过滤配置
        """
        print(f"\n⏳ 正在下载 {symbol} 最近 {days} 天 {interval} 数据...")
        df = self.fetcher.fetch_range(symbol=symbol, interval=interval, days=days)

        if df.empty:
            print(f"  ⚠️ Binance API 不可用 ({symbol})，生成合成数据替代...")
            df = self.generate_synthetic_data(symbol=symbol, days=days, interval=interval)
            print(f"  ✅ 合成数据: {len(df)} 根K线 | {df.index[0].date()} ~ {df.index[-1].date()}")

        print(f"✅ 获取 {len(df)} 根K线 | {df.index[0].date()} ~ {df.index[-1].date()}")
        print(f"   手续费: {self.fee:.2%} | 滑点: {self.slippage:.3%} | 初始资金: ${self.initial_capital:,.0f}")
        print("⏳ 正在生成信号...")

        # 生成信号
        signals_df = self.signal_gen.generate(df)
        trades = self._execute_trades(signals_df, symbol)
        equity_curve, monthly = self._compute_equity(trades, df)

        print(f"✅ 回测完成: {len(trades)} 笔交易 | 胜率 {self._win_rate(trades):.1%}")

        return self._build_result(
            trades=trades,
            equity_curve=equity_curve,
            monthly_returns=monthly,
            df=df,
            strategy_name="UnifiedStrategy",
            symbol=symbol,
        )

    def run_strategy(
        self,
        df: pd.DataFrame,
        strategy_fn,
        strategy_name: str = "CustomStrategy",
        **kwargs,
    ) -> BacktestResult:
        """
        用自定义信号函数回测

        strategy_fn(df: pd.DataFrame) -> pd.Series(signal=1/-1/0)
        """
        df = df.copy()
        signals = strategy_fn(df, **kwargs)
        df["signal"] = signals.values
        df["confidence"] = 0.5
        df["regime"] = "custom"
        df["score"] = signals.astype(float) * 0.5

        trades = self._execute_trades(df, "CUSTOM")
        equity_curve, monthly = self._compute_equity(trades, df)

        return self._build_result(
            trades=trades,
            equity_curve=equity_curve,
            monthly_returns=monthly,
            df=df,
            strategy_name=strategy_name,
            symbol="CUSTOM",
        )

    def _execute_trades(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> List[Trade]:
        """事件驱动：逐根K线执行交易"""
        trades = []
        position = None  # 当前持仓
        current_equity = self.initial_capital  # 实时权益

        n = len(df)
        for i in range(n):
            bar = df.iloc[i]
            close = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])
            ts = bar.name.to_pydatetime() if hasattr(bar.name, "to_pydatetime") else bar.name
            sig = int(bar["signal"]) if not pd.isna(bar["signal"]) else 0
            conf = float(bar["confidence"]) if not pd.isna(bar["confidence"]) else 0.0

            if position is None:
                # ---- 建仓 ----
                if sig != 0 and conf > 0.25:
                    side = "LONG" if sig > 0 else "SHORT"
                    # 滑点执行
                    exec_price = close * (1 + self.slippage if sig > 0 else 1 - self.slippage)
                    entry_price = exec_price
                    fee = exec_price * self.fee
                    stop_loss = entry_price * (1 - self.stop_loss_pct if sig > 0 else 1 + self.stop_loss_pct)
                    take_profit = entry_price * (1 + self.take_profit_pct if sig > 0 else 1 - self.take_profit_pct)
                    regime_at_entry = str(bar["regime"]) if "regime" in bar else "unknown"
                    position = {
                        "side": side,
                        "entry_price": entry_price,
                        "entry_time": ts,
                        "entry_bar": i,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "regime": regime_at_entry,
                        "signal": f"{sig:+d} ({conf:.0%})",
                        # 名义金额 = 账户权益 × 仓位上限比例
                        "quantity": current_equity * self.max_position_pct / entry_price,
                    }

            else:
                # ---- 持仓中：检查止损/止盈/出场信号 ----
                should_exit = False
                exit_reason = "FORCE_CLOSE"
                ep = position["entry_price"]
                sl = position["stop_loss"]
                tp = position["take_profit"]
                side = position["side"]

                if side == "LONG":
                    if low <= sl:
                        exit_price = sl; should_exit = True; exit_reason = "STOP_LOSS"
                    elif high >= tp:
                        exit_price = tp; should_exit = True; exit_reason = "TAKE_PROFIT"
                    elif sig == -1 and conf > 0.3:
                        exit_price = close; should_exit = True; exit_reason = "SIGNAL_REVERSE"
                    elif i - position["entry_bar"] > 96 * 7:
                        exit_price = close; should_exit = True; exit_reason = "TIME_EXIT"
                else:  # SHORT
                    if high >= sl:
                        exit_price = sl; should_exit = True; exit_reason = "STOP_LOSS"
                    elif low <= tp:
                        exit_price = tp; should_exit = True; exit_reason = "TAKE_PROFIT"
                    elif sig == 1 and conf > 0.3:
                        exit_price = close; should_exit = True; exit_reason = "SIGNAL_REVERSE"
                    elif i - position["entry_bar"] > 96 * 7:
                        exit_price = close; should_exit = True; exit_reason = "TIME_EXIT"

                if should_exit:
                    # 执行价格（含滑点）
                    exec_price = exit_price * (1 - self.slippage if side == "LONG" else 1 + self.slippage)
                    # 收益率 = (出场价 - 入场价) / 入场价（fraction）
                    raw_return = (exec_price - ep) / ep
                    direction = 1 if side == "LONG" else -1
                    pnl_pct = float(raw_return * direction)  # fraction, e.g. -0.05 = -5%
                    # 手续费（fraction = 手续费美元 / 名义金额）
                    notional = ep * position["quantity"]
                    total_fee_frac = (ep * self.fee + exec_price * self.fee) / notional
                    net_pct = pnl_pct - total_fee_frac  # fraction net return
                    trade_pnl_dollars = notional * net_pct  # actual $PnL

                    trades.append(Trade(
                        entry_time=position["entry_time"],
                        exit_time=ts,
                        side=side,
                        entry_price=ep,
                        exit_price=exec_price,
                        quantity=position["quantity"],
                        pnl=float(trade_pnl_dollars),
                        pnl_pct=float(net_pct),
                        fees=float(ep * self.fee + exec_price * self.fee),
                        signal=position["signal"],
                        regime=position["regime"],
                        holding_bars=i - position["entry_bar"],
                    ))
                    position = None
                    current_equity *= (1.0 + float(net_pct))

        # 平仓未结束的头寸（最后一日收盘平仓）
        if position is not None:
            last_close = float(df.iloc[-1]["close"])
            last_ts = df.index[-1].to_pydatetime()
            ep = position["entry_price"]
            side = position["side"]
            exit_price = last_close
            raw_return = (exit_price - ep) / ep
            pnl_pct = raw_return * (1 if side == "LONG" else -1)
            notional = ep * position["quantity"]
            total_fee_frac = (ep * self.fee + exit_price * self.fee) / notional
            net_pct = float(pnl_pct - total_fee_frac)
            trade_pnl_dollars = notional * net_pct
            trades.append(Trade(
                entry_time=position["entry_time"],
                exit_time=last_ts,
                side=side,
                entry_price=ep,
                exit_price=exit_price,
                quantity=position["quantity"],
                pnl=float(trade_pnl_dollars),
                pnl_pct=net_pct,
                fees=float(ep * self.fee + exit_price * self.fee),
                signal=position["signal"] + " [OPEN]",
                regime=position["regime"],
                holding_bars=len(df) - position["entry_bar"],
            ))
            current_equity *= (1.0 + net_pct)

        return trades

    def _compute_equity(
        self,
        trades: List[Trade],
        price_df: pd.DataFrame,
    ) -> Tuple[List[EquityPoint], Dict[str, float]]:
        """计算权益曲线 + 月度收益"""
        monthly_returns = {}
        for t in trades:
            ts_key = t.exit_time.strftime("%Y-%m")
            monthly_returns[ts_key] = monthly_returns.get(ts_key, 0.0) + float(t.pnl_pct)

        # 权益曲线：按时间轴重建
        if price_df.empty or not trades:
            eq = [
                EquityPoint(pd.Timestamp(price_df.index[i]) if i < len(price_df) else pd.Timestamp.now(),
                            self.initial_capital, 0.0)
                for i in range(min(len(price_df), 1))
            ]
            return eq, monthly_returns

        sorted_trades = sorted(trades, key=lambda t: t.exit_time)
        trade_idx = 0
        cumulative_factor = 1.0
        peak_equity = self.initial_capital
        equity_points = []

        for ts in price_df.index:
            ts_dt = pd.Timestamp(ts)
            while trade_idx < len(sorted_trades) and sorted_trades[trade_idx].exit_time <= ts_dt:
                cumulative_factor *= (1.0 + float(sorted_trades[trade_idx].pnl_pct))
                trade_idx += 1
            cur_equity = self.initial_capital * cumulative_factor
            dd = peak_equity - cur_equity
            equity_points.append(EquityPoint(ts_dt, cur_equity, dd))
            peak_equity = max(peak_equity, cur_equity)

        return equity_points, monthly_returns

    def _win_rate(self, trades: List[Trade]) -> float:
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.pnl > 0)
        return wins / len(trades)

    def _build_result(
        self,
        trades: List[Trade],
        equity_curve: List[EquityPoint],
        monthly_returns: Dict[str, float],
        df: pd.DataFrame,
        strategy_name: str,
        symbol: str,
    ) -> BacktestResult:
        if not trades:
            return self._empty_result(symbol)

        pnls = np.array([t.pnl_pct for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        total_return = float(np.prod(1 + pnls) - 1)
        neg_eq = np.array([e.equity for e in equity_curve])
        peak = np.maximum.accumulate(neg_eq)
        # drawdowns[i] = (peak[i] - equity[i]) / peak[i]  → fraction
        drawdowns_frac = (peak - neg_eq) / peak
        max_dd_pct = float(np.max(drawdowns_frac))  # already fraction
        max_dd = float(np.max(peak - neg_eq))      # dollar amount

        n = len(trades)
        years = max((df.index[-1] - df.index[0]).days / 365.0, 0.01)
        # total_return is fraction (e.g. 0.05 = 5% total return)
        ann_ret = float(max(-0.99, min(10.0, (1 + max(-0.999, total_return)) ** (1 / years) - 1)))

        ret_array = np.array([t.pnl_pct for t in trades])
        excess = ret_array - 0.0  # 无风险利率=0
        ret_std = float(np.std(excess)) if len(ret_array) > 1 else 0.02
        sharpe = float(np.mean(excess) / max(ret_std, 1e-4) * np.sqrt(252 * n / years / 252)) if n >= 5 else float(np.mean(excess) / max(ret_std, 1e-4))

        downside = np.array([r for r in ret_array if r < 0])
        dd_std = float(np.std(downside)) if len(downside) > 1 else max(float(np.abs(np.mean(downside))) if len(downside) == 1 else 0.02, 0.001)
        sortino = float(np.mean(excess) / max(dd_std, 1e-4) * np.sqrt(252)) if n >= 5 else 0.0

        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
        win_rate = float(len(wins) / n)

        # 状态统计
        regimes = {}
        for t in trades:
            r = t.regime
            if r not in regimes:
                regimes[r] = {"count": 0, "wins": 0, "total_pnl": 0.0}
            regimes[r]["count"] += 1
            if t.pnl > 0:
                regimes[r]["wins"] += 1
            regimes[r]["total_pnl"] += t.pnl

        # 月度统计
        for r in regimes:
            c = regimes[r]["count"]
            regimes[r]["win_rate"] = regimes[r]["wins"] / c
            regimes[r]["avg_pnl"] = regimes[r]["total_pnl"] / c

        start_date = df.index[0].strftime("%Y-%m-%d") if len(df) > 0 else ""
        end_date = df.index[-1].strftime("%Y-%m-%d") if len(df) > 0 else ""

        summary = {
            "total_return_pct": f"{total_return:+.1%}",
            "ann_return_pct": f"{ann_ret:+.1%}",
            "sharpe": f"{sharpe:+.2f}",
            "sortino": f"{sortino:+.2f}",
            "max_drawdown_pct": f"{max_dd_pct:.1%}",
            "win_rate": f"{win_rate:.1%}",
            "profit_factor": f"{abs(avg_win/avg_loss) if avg_loss != 0 else 0:.2f}",
            "total_trades": n,
            "best_trade": f"{max(pnls):+.1%}",
            "worst_trade": f"{min(pnls):+.1%}",
            "calmar": f"{ann_ret/max_dd if max_dd > 0 else 0:+.2f}",
        }

        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            total_trades=n,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=abs(avg_win / avg_loss) if avg_loss != 0 else 0.0,
            total_return=total_return,
            annualized_return=ann_ret,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            calmar_ratio=ann_ret / max_dd if max_dd > 0 else 0.0,
            win_loss_ratio=abs(avg_win / avg_loss) if avg_loss != 0 else 0.0,
            avg_bars_held=float(np.mean([t.holding_bars for t in trades])),
            trades=trades,
            equity_curve=equity_curve,
            monthly_returns=monthly_returns,
            regime_stats=regimes,
            summary=summary,
        )

    def _empty_result(self, symbol: str) -> BacktestResult:
        return BacktestResult(
            strategy_name="Empty", symbol=symbol,
            start_date="", end_date="",
            total_trades=0, win_rate=0.0,
            avg_win=0.0, avg_loss=0.0, profit_factor=0.0,
            total_return=0.0, annualized_return=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0,
            max_drawdown=0.0, max_drawdown_pct=0.0,
            calmar_ratio=0.0, win_loss_ratio=0.0,
            avg_bars_held=0.0,
            trades=[], equity_curve=[],
            monthly_returns={}, regime_stats={},
            summary={},
        )

    # ========== 策略对比 ==========
    def compare_strategies(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        days: int = 60,
        strategies: Optional[List[Tuple[str, callable]]] = None,
    ) -> Dict[str, BacktestResult]:
        """
        对比多个策略

        strategies: [("策略名", signal_fn), ...]
        signal_fn(df) -> pd.Series(signal=1/-1/0)
        """
        print(f"\n⏳ 下载 {symbol} 数据用于策略对比...")
        df = self.fetcher.fetch_range(symbol=symbol, interval=interval, days=days)
        if df.empty:
            return {}

        results = {}
        for name, fn in (strategies or []):
            print(f"  回测: {name}...")
            r = self.run_strategy(df, fn, name)
            r.print_summary()
            results[name] = r

        # 汇总表
        if results:
            print("\n" + "=" * 70)
            print(f"{'策略':<20} {'交易数':>7} {'胜率':>8} {'收益率':>10} {'Sharpe':>8} {'最大回撤':>10}")
            print("-" * 70)
            for name, r in results.items():
                print(f"{name:<20} {r.total_trades:>7} {r.win_rate:>8.1%} "
                      f"{r.total_return:>+10.1%} {r.sharpe_ratio:>+8.2f} {r.max_drawdown_pct:>10.1%}")
            print("=" * 70)

        return results
