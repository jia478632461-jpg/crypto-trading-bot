"""
===================================================================
自适应策略引擎 (Adaptive Strategy Engine)
根据市场状态实时切换最优策略组合
===================================================================
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
logger = logging.getLogger(__name__)


class StrategyType(Enum):
    TREND_FOLLOWING = "TREND_FOLLOWING"
    MEAN_REVERSION  = "MEAN_REVERSION"
    BREAKOUT        = "BREAKOUT"
    MOMENTUM        = "MOMENTUM"
    PAIRS           = "PAIRS"
    MARKET_MAKING   = "MARKET_MAKING"
    SCALPING        = "SCALPING"
    SWING           = "SWING"
    DEEP_VALUE      = "DEEP_VALUE"


@dataclass
class StrategyAllocation:
    strategy_type: StrategyType
    weight: float
    confidence: float
    parameters: Dict = field(default_factory=dict)


class TechnicalAnalyzerLite:
    @staticmethod
    def sma(c, p):
        s = pd.Series(c)
        return s.rolling(p).mean().values
    @staticmethod
    def ema(c, p):
        s = pd.Series(c)
        return s.ewm(span=p, adjust=False).mean().values
    @staticmethod
    def rsi(c, p=14):
        s = pd.Series(c)
        d = s.diff()
        g = d.where(d > 0, 0.0); l = (-d).where(d < 0, 0.0)
        ag = g.ewm(com=p-1, min_periods=p).mean()
        al = l.ewm(com=p-1, min_periods=p).mean()
        return (100 - 100 / (1 + ag / (al + 1e-10))).values
    @staticmethod
    def atr(h, lo, c, p=14):
        hi_s = pd.Series(h); lo_s = pd.Series(lo); c_s = pd.Series(c)
        tr = pd.concat([hi_s - lo_s, abs(hi_s - c_s.shift(1)), abs(lo_s - c_s.shift(1))], axis=1).max(axis=1)
        return tr.rolling(p).mean().values
    @staticmethod
    def adx(hi, lo, c, p=14):
        hi_s = pd.Series(hi); lo_s = pd.Series(lo); c_s = pd.Series(c)
        hp = hi_s.diff(); lm = -lo_s.diff()
        hp[hp < 0] = 0; lm[lm < 0] = 0
        tr = pd.concat([hi_s - lo_s, abs(hi_s - c_s.shift(1)), abs(lo_s - c_s.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(p).mean()
        pdi_s = 100*(hp.rolling(p).mean()/atr); mdi_s = 100*(lm.rolling(p).mean()/atr)
        dx = 100*abs(pdi_s-mdi_s)/(pdi_s+mdi_s+1e-10)
        return dx.rolling(p).mean().values, pdi_s.values, mdi_s.values
    @staticmethod
    def bollinger_bands(c, p=20, sd=2.0):
        c_s = pd.Series(c)
        m = c_s.rolling(p).mean().values
        s = c_s.rolling(p).std().values
        u = m + sd*s; l = m - sd*s
        bp = (c - l) / (u - l + 1e-10)
        return m, u, l, bp
    @staticmethod
    def ao(h, lo, f=5, s=34):
        med = pd.Series((pd.Series(h) + pd.Series(lo)) / 2.0)
        return (med.rolling(f).mean() - med.rolling(s).mean()).values
    def compute_all(self, df):
        c = df["close"].values; h = df["high"].values
        lo = df["low"].values
        v = df["volume"].values if "volume" in df.columns else np.ones_like(c)
        n = len(c)
        def lv(a, d=0.0):
            a_arr = np.asarray(a, dtype=float).ravel()
            valid = a_arr[~np.isnan(a_arr)]
            return float(valid[-1]) if len(valid) > 0 else float(d)
        adx_a, pdi, mdi = TechnicalAnalyzerLite.adx(h, lo, c)
        bb_m, bb_u, bb_l, bb_p = TechnicalAnalyzerLite.bollinger_bands(c)
        roc_val = float((c[-1]-c[-13])/(c[-13]+1e-10)*100) if n >= 13 else 0.0
        mom_val = float(c[-1]-c[-11]) if n >= 11 else 0.0
        ma7 = TechnicalAnalyzerLite.sma(c, 7); ma25 = TechnicalAnalyzerLite.sma(c, 25)
        ma50 = TechnicalAnalyzerLite.sma(c, 50); ma99 = TechnicalAnalyzerLite.sma(c, 99)
        ema12 = TechnicalAnalyzerLite.ema(c, 12); ema26 = TechnicalAnalyzerLite.ema(c, 26)
        ao_val = TechnicalAnalyzerLite.ao(h, lo)
        vol5 = float(c[-5:].std()/c[-5:].mean()) if n >= 5 else 0.0
        vol20 = float(c[-20:].std()/c[-20:].mean()) if n >= 20 else 0.0
        natr = lv(TechnicalAnalyzerLite.atr(h, lo, c)/c*100) if c[-1] > 0 else 0.0
        return {
            "close": lv(c), "sma7": lv(ma7), "sma25": lv(ma25),
            "sma50": lv(ma50), "sma99": lv(ma99),
            "ema12": lv(ema12), "ema26": lv(ema26),
            "rsi": lv(TechnicalAnalyzerLite.rsi(c)),
            "adx": lv(adx_a), "plus_di": lv(pdi), "minus_di": lv(mdi),
            "bb_position": lv(bb_p), "ao": lv(ao_val),
            "volume_ratio": float(v[-1]/v[-20:].mean()) if n >= 20 else 1.0,
            "momentum": mom_val, "roc": roc_val,
            "natr": natr, "volatility_5": vol5, "volatility_20": vol20,
            "price_change": float((c[-1]-c[-2])/c[-2]) if n >= 2 else 0.0,
            "high_low_ratio": float(h[-1]/lo[-1]) if lo[-1] > 0 else 1.0,
            "ma_bull_score": sum([1 if c[-1] > lv(ma50) else -1,
                                   1 if lv(ma7) > lv(ma25) else -1,
                                   1 if lv(ema12) > lv(ema26) else -1]) / 3,
            "hurst": 0.5, "spread": 0.005
        }


class AdaptiveStrategyEngine:
    """
    自适应策略引擎
    核心：根据市场状态动态选择和配置策略
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.ta = TechnicalAnalyzerLite()
        self.strategy_params = {
            StrategyType.TREND_FOLLOWING: {"adx_threshold": 25, "ma_fast": 10, "ma_slow": 50},
            StrategyType.MEAN_REVERSION:   {"bb_period": 20, "rsi_oversold": 30, "rsi_overbought": 70},
            StrategyType.BREAKOUT:        {"lookback": 20, "volume_mult": 1.5},
            StrategyType.MOMENTUM:         {"roc_period": 12, "volume_period": 20},
            StrategyType.SCALPING:         {"ema_fast": 5, "ema_slow": 13},
            StrategyType.SWING:            {"sma50": 50, "sma200": 200, "rsi_period": 14},
            StrategyType.DEEP_VALUE:       {"rsi_threshold_low": 35, "rsi_threshold_high": 65},
            StrategyType.PAIRS:            {"correlation_threshold": 0.8, "z_entry": 2.0},
            StrategyType.MARKET_MAKING:    {"spread_mult": 1.5, "inventory_limit": 0.2},
        }

    def detect_market_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检测市场上下文"""
        c = df["close"].values; h = df["high"].values
        lo = df["low"].values; v = df["volume"].values if "volume" in df.columns else np.ones_like(c)
        n = len(c)
        adx_a, pdi, mdi = TechnicalAnalyzerLite.adx(h, lo, c)
        bb_m, bb_u, bb_l, bb_p = TechnicalAnalyzerLite.bollinger_bands(c)
        # Keep full ATR array for percentile, then extract last as percentage-of-price
        atr_arr = TechnicalAnalyzerLite.atr(h, lo, c)
        natr_arr = atr_arr / c * 100.0  # percentage of price
        natr = float(natr_arr[-1]) if c[-1] > 0 else 0.0
        adx_val = float(adx_a[~np.isnan(adx_a)][-1]) if np.any(~np.isnan(adx_a)) else 0.0
        rsi_arr = TechnicalAnalyzerLite.rsi(c)
        rsi = float(rsi_arr[~np.isnan(rsi_arr)][-1]) if np.any(~np.isnan(rsi_arr)) else 50.0
        vol_ratio = float(v[-1]/v[-20:].mean()) if n >= 20 else 1.0
        bb_arr = np.asarray(bb_p)
        bb_position = float(bb_arr[~np.isnan(bb_arr)][-1]) if np.any(~np.isnan(bb_arr)) else 0.5

        # 趋势强度
        trend_strength = float(min(1.0, adx_val / 50.0))
        # 波动率等级 (percentile of ATR % across history)
        natr_for_pct = natr_arr[~np.isnan(natr_arr)]
        if len(natr_for_pct) > 5:
            vol_pct = float(np.percentile(natr_for_pct, 80) / 100.0)
        else:
            vol_pct = natr / 100.0
        volatility_level = float(min(1.0, natr / 50.0))

        # 市场状态
        if adx_val > 25:
            regime = "BULL_TREND" if float(pdi[~np.isnan(pdi)][-1]) > float(mdi[~np.isnan(mdi)][-1]) else "BEAR_TREND"
        else:
            regime = "SIDEWAYS"

        if bb_position > 0.95 or bb_position < 0.05:
            regime = "BREAKOUT"
        if natr > 40:
            regime = "HIGH_VOL"
        if vol_ratio > 1.5:
            regime = "BREAKOUT"

        return {
            "regime": regime,
            "adx": adx_val, "rsi": rsi,
            "trend_strength": trend_strength,
            "volatility": volatility_level,
            "natr": natr,
            "volume_ratio": vol_ratio,
            "bb_position": bb_position,
            "indicators": self.ta.compute_all(df)
        }

    def score_strategies(self, ctx: Dict) -> Dict[StrategyType, float]:
        """根据市场上下文评分各策略"""
        scores = {}
        regime = ctx["regime"]
        ts = ctx["trend_strength"]
        vol = ctx["volatility"]
        bb = ctx["bb_position"]
        rsi = ctx["rsi"]
        ind = ctx["indicators"]

        for st in StrategyType:
            s = 0.5

            if st == StrategyType.TREND_FOLLOWING:
                if regime in ["BULL_TREND", "BEAR_TREND"] and ts > 0.3: s += 0.3
                if ctx["adx"] > 25: s += 0.2
                if vol > 0.5: s -= 0.1

            elif st == StrategyType.MEAN_REVERSION:
                if regime == "SIDEWAYS" and vol < 0.3: s += 0.4
                if abs(rsi - 50) > 20: s += 0.2
                if ind.get("ma_bull_score", 0) == 0: s += 0.1

            elif st == StrategyType.BREAKOUT:
                if bb > 0.95 or bb < 0.05: s += 0.3
                if ctx["volume_ratio"] > 1.5: s += 0.3
                if regime == "HIGH_VOL": s += 0.2

            elif st == StrategyType.MOMENTUM:
                if ind.get("momentum", 0) > 0: s += 0.25
                if ind.get("roc", 0) > 0: s += 0.25
                if ctx["volume_ratio"] > 1.2: s += 0.2

            elif st == StrategyType.SCALPING:
                if vol < 0.3: s += 0.3
                if regime == "SIDEWAYS": s += 0.2

            elif st == StrategyType.SWING:
                if regime in ["BULL_TREND", "BEAR_TREND"]: s += 0.3
                if rsi < 70 and rsi > 30: s += 0.2

            elif st == StrategyType.DEEP_VALUE:
                if rsi < 35: s += 0.4
                elif rsi > 65: s -= 0.2

            elif st == StrategyType.PAIRS:
                if regime == "SIDEWAYS": s += 0.3

            elif st == StrategyType.MARKET_MAKING:
                if vol < 0.3: s += 0.3

            scores[st] = max(0.0, min(1.0, s))

        return scores

    def select_strategies(self, ctx: Dict, maxStrategies: int = 3) -> List[StrategyAllocation]:
        """选择并分配策略权重"""
        scores = self.score_strategies(ctx)
        sorted_strats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top = sorted_strats[:maxStrategies]
        total = sum(s for _, s in top) + 1e-10
        return [
            StrategyAllocation(
                strategy_type=st, weight=s/total, confidence=s,
                parameters=self.strategy_params.get(st, {})
            )
            for st, s in top
        ]

    def _strategy_signal(self, st: StrategyType, params: Dict, ctx: Dict) -> float:
        """单个策略产生信号"""
        ind = ctx["indicators"]; regime = ctx["regime"]
        rsi = ctx["rsi"]; bb = ctx["bb_position"]
        adx = ctx["adx"]; ts = ctx["trend_strength"]

        if st == StrategyType.TREND_FOLLOWING:
            if regime == "BULL_TREND" and adx > 25: return min(1.0, adx/60)
            if regime == "BEAR_TREND" and adx > 25: return -min(1.0, adx/60)
            return 0.0

        elif st == StrategyType.MEAN_REVERSION:
            if bb < 0.2 and rsi < 45: return 0.8
            if bb > 0.8 and rsi > 55: return -0.8
            if bb < 0.5: return 0.4 * (1 - rsi/100)
            if bb > 0.5: return -0.4 * (rsi/100)
            return 0.0

        elif st == StrategyType.BREAKOUT:
            vr = ctx["volume_ratio"]
            pc = ind.get("price_change", 0)
            if vr > 1.5 and pc > 0.01: return 0.8
            if vr > 1.5 and pc < -0.01: return -0.8
            if bb > 0.95: return 0.6
            if bb < 0.05: return -0.6
            return 0.0

        elif st == StrategyType.MOMENTUM:
            mom = ind.get("momentum", 0); roc = ind.get("roc", 0)
            c = ind.get("close", 1)
            sig = float(np.tanh(mom/c*10)) * 0.6 + float(np.tanh(roc/10)) * 0.4
            return max(-1, min(1, sig))

        elif st == StrategyType.SCALPING:
            if ind.get("natr", 0) < 30:
                return 0.5 * (1 - 2*abs(rsi-50)/50)
            return 0.0

        elif st == StrategyType.SWING:
            if regime == "BULL_TREND" and rsi < 70: return 0.6
            if regime == "BEAR_TREND" and rsi > 30: return -0.6
            return 0.0

        elif st == StrategyType.DEEP_VALUE:
            if rsi < 35: return 0.7
            if rsi > 65: return -0.5
            return 0.0

        return 0.0

    def generate_signal(self, allocations: List[StrategyAllocation], ctx: Dict) -> Dict[str, Any]:
        """综合所有策略分配生成最终信号"""
        if not allocations:
            return {"signal": "HOLD", "confidence": 0.0, "score": 0.0}

        total_signal = 0.0
        for alloc in allocations:
            sig = self._strategy_signal(alloc.strategy_type, alloc.parameters, ctx)
            total_signal += sig * alloc.weight * alloc.confidence

        total_signal = max(-1.0, min(1.0, total_signal))
        confidence = abs(total_signal)

        if total_signal > 0.3: signal = "BUY"
        elif total_signal < -0.3: signal = "SELL"
        else: signal = "HOLD"

        return {
            "signal": signal,
            "score": float(total_signal),
            "confidence": float(confidence),
            "top_strategy": allocations[0].strategy_type.value,
            "allocations": {a.strategy_type.value: {"weight": a.weight, "confidence": a.confidence} for a in allocations},
            "regime": ctx["regime"]
        }

    def run_backtest(self, df: pd.DataFrame, strategy: StrategyType) -> Dict[str, float]:
        """回测单个策略"""
        close = df["close"].values
        n = len(close)
        if n < 50:
            return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}

        returns = np.diff(close) / (close[:-1] + 1e-10)
        signals = np.zeros(n)
        for i in range(20, n):
            ctx = self.detect_market_context(df.iloc[:i+1])
            allocs = self.select_strategies(ctx, maxStrategies=5)
            for a in allocs:
                if a.strategy_type == strategy:
                    signals[i] = self._strategy_signal(strategy, a.parameters, ctx)

        pos = np.zeros(n)
        pos[:-1] = signals[1:]
        strat_ret = pos[:-1] * returns
        total_ret = float(np.sum(strat_ret))
        std_ret = float(np.std(strat_ret))
        sharpe = total_ret / std_ret * np.sqrt(252) if std_ret > 1e-10 else 0.0
        cumret = np.cumprod(1 + strat_ret)
        peak = np.maximum.accumulate(cumret)
        mdd = float(np.min((cumret - peak) / (peak + 1e-10)))
        wins = strat_ret > 0
        win_rate = float(np.mean(wins)) if len(wins) > 0 else 0.0
        return {"total_return": total_ret, "sharpe": sharpe, "max_drawdown": mdd, "win_rate": win_rate}
