"""
===================================================================
统一分析引擎 (Unified Analytics Engine)
=====================p==============================================
整合所有分析模块，为量化交易提供全方位实时分析能力

模块组成:
  1. TechnicalAnalyzer   - 全技术指标库 (60+ 指标)
  2. MarketRegimeDetector - 市场状态识别
  3. StatisticalModels    - 统计模型 (Hurst/协整/Granger/GARCH)
  4. GameTheoryAnalyzer  - 博弈论分析
  5. OptionsAnalyzer     - 期权希腊值
  6. RiskAnalyzer        - 风险指标
  7. PortfolioOptimizer  - 组合优化
  8. OrderBookAnalyzer   - 订单簿/微观结构
  9. AdaptiveStrategyEngine - 自适应策略引擎

使用方法:
  engine = UnifiedAnalyticsEngine(config)
  result = engine.analyze(df, orderbook=ob)
  signal = engine.generate_signal(result)
===================================================================
"""

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_sys.argv[0])) if _sys.argv[0] else "/workspace")

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import sys
import os

logger = logging.getLogger(__name__)

# =============================================================================
# 导入所有子模块
# =============================================================================

try:
    from src.analytics_engine import TechnicalAnalyzer, MarketRegimeDetector, MarketRegime
except ImportError:
    logger.warning("analytics_engine.py 未找到")
    TechnicalAnalyzer = None; MarketRegimeDetector = None; MarketRegime = None

try:
    from src.game_theory import StatisticalModels, GameTheoryAnalyzer
except ImportError:
    logger.warning("game_theory.py 未找到")
    StatisticalModels = None; GameTheoryAnalyzer = None

try:
    from src.analytics_options_risk import (
        OptionsAnalyzer, RiskAnalyzer, PortfolioOptimizer, OrderBookAnalyzer
    )
except ImportError:
    logger.warning("analytics_options_risk.py 未找到")
    OptionsAnalyzer = None
    RiskAnalyzer = None
    PortfolioOptimizer = None
    OrderBookAnalyzer = None

try:
    from src.adaptive_strategy import AdaptiveStrategyEngine, StrategyType
except ImportError:
    logger.warning("adaptive_strategy.py 未找到")
    AdaptiveStrategyEngine = None
    StrategyType = None


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class FullAnalysisResult:
    """完整分析结果"""
    symbol: str
    timestamp: datetime

    # 市场状态
    regime: str
    regime_confidence: float

    # 各维度评分
    technical_score: float      # -1 ~ 1
    momentum_score: float
    mean_reversion_score: float
    volume_score: float
    game_theory_score: float
    ml_score: float
    risk_score: float
    options_score: float
    overall_score: float        # -1 ~ 1

    # 综合信号
    signal: str                  # BUY / SELL / HOLD
    signal_confidence: float    # 0 ~ 1
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float

    # 推荐策略
    recommended_strategies: List[Dict]

    # 详细指标
    all_indicators: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    regime_context: Dict[str, Any] = field(default_factory=dict)
    game_theory: Dict[str, Any] = field(default_factory=dict)
    portfolio_metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"[{self.symbol}] {self.signal} ({self.signal_confidence:.0%}) | "
            f" Regime: {self.regime} | Overall: {self.overall_score:+.2f} | "
            f" Tech: {self.technical_score:+.2f} | Risk: {self.risk_score:.2f}"
        )


# =============================================================================
# 统一分析引擎
# =============================================================================

class UnifiedAnalyticsEngine:
    """
    统一分析引擎
    调用所有子模块，输出完整的分析结果
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._init_modules()

    def _init_modules(self):
        """初始化所有模块"""
        self.tech = TechnicalAnalyzer() if TechnicalAnalyzer else None
        self.regime_detector = MarketRegimeDetector() if MarketRegimeDetector else None
        self.stats = StatisticalModels() if StatisticalModels else None
        self.gt = GameTheoryAnalyzer() if GameTheoryAnalyzer else None
        self.options = OptionsAnalyzer() if OptionsAnalyzer else None
        self.risk = RiskAnalyzer() if RiskAnalyzer else None
        self.portfolio = PortfolioOptimizer() if PortfolioOptimizer else None
        self.ob_analyzer = OrderBookAnalyzer() if OrderBookAnalyzer else None
        self.adaptive = AdaptiveStrategyEngine(self.config) if AdaptiveStrategyEngine else None

    def analyze(self, df: pd.DataFrame,
                symbol: str = "UNKNOWN",
                orderbook: Dict = None,
                market_returns: np.ndarray = None) -> FullAnalysisResult:
        """
        执行完整分析

        Args:
            df: K线数据 DataFrame，必须包含 high, low, close, volume 列
            symbol: 交易对名称
            orderbook: 订单簿数据（可选）
            market_returns: 市场基准收益率（用于Beta计算）

        Returns:
            FullAnalysisResult
        """
        t0 = datetime.now()
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values if "volume" in df.columns else np.ones_like(close)
        n = len(close)

        # ===== 1. 技术指标 =====
        indicators = {}
        tech_score = 0.0
        if self.tech:
            try:
                indicators = self.tech.compute_all(df)
                tech_score, _ = self.tech.score_technical(indicators)
            except Exception as e:
                logger.warning(f"技术指标计算失败: {e}")

        # ===== 2. 市场状态 =====
        regime = MarketRegime.UNKNOWN if MarketRegime else "UNKNOWN"
        regime_confidence = 0.0
        regime_context = {}
        if self.regime_detector:
            try:
                regime, regime_confidence = self.regime_detector.detect_regime(df)
                regime_context = {"regime": regime.value, "confidence": regime_confidence}
            except Exception as e:
                logger.warning(f"状态检测失败: {e}")

        # ===== 3. 统计模型 =====
        stats_output = {}
        mr_score = 0.0
        if self.stats and n >= 30:
            try:
                returns = np.diff(close) / (close[:-1] + 1e-10)
                hurst = self.stats.hurst_exponent(close)
                z = self.stats.z_score(close)
                mr_signal = self.stats.mean_reversion_signal(close)
                adf = self.stats._adf_test(returns)
                serial = self.stats.serial_correlation(returns)
                stats_output = {
                    "hurst": hurst, "z_score": z, "mean_reversion_signal": mr_signal[0],
                    "mean_reversion_confidence": mr_signal[1],
                    "adf_p_value": adf.get("p_value", 1.0),
                    "serial_corr_p_value": serial.get("p_value", 1.0)
                }
                mr_score = -float(np.tanh(z)) if hurst < 0.5 else 0.0
            except Exception as e:
                logger.warning(f"统计模型失败: {e}")

        # ===== 4. 博弈论 =====
        gt_score = 0.0
        gt_output = {}
        if self.gt:
            try:
                gt_output = self.gt.analyze_market_game(df, orderbook)
                gt_score = gt_output.get("overall_game_score", 0.0)
            except Exception as e:
                logger.warning(f"博弈论分析失败: {e}")

        # ===== 5. 风险分析 =====
        risk_output = {}
        risk_score = 0.5
        if self.risk and n >= 20:
            try:
                returns = np.diff(close) / (close[:-1] + 1e-10)
                risk_output = self.risk.risk_metrics_all(returns, close, market_returns)
                var_95 = risk_output.get("var_95", 0.0)
                sharpe = risk_output.get("sharpe_ratio", 0.0)
                mdd = abs(risk_output.get("max_drawdown", 0.0))
                risk_score = min(1.0, var_95 * 10 + mdd * 0.5)
                risk_score = max(0.0, risk_score)
            except Exception as e:
                logger.warning(f"风险分析失败: {e}")

        # ===== 6. 自适应策略 =====
        adaptive_output = {}
        ml_score = 0.0
        recommended_strategies = []
        if self.adaptive and n >= 30:
            try:
                ctx = self.adaptive.detect_market_context(df)
                allocations = self.adaptive.select_strategies(ctx)
                signal_result = self.adaptive.generate_signal(allocations, ctx)
                adaptive_output = signal_result
                ml_score = signal_result.get("score", 0.0)
                recommended_strategies = [
                    {"strategy": k, "weight": v["weight"], "confidence": v["confidence"]}
                    for k, v in signal_result.get("allocations", {}).items()
                ]
            except Exception as e:
                logger.warning(f"自适应策略失败: {e}")

        # ===== 7. 动量评分 =====
        momentum_score = tech_score * 0.4 + mr_score * 0.3 + gt_score * 0.3

        # ===== 8. 成交量评分 =====
        volume_score = 0.0
        if indicators.get("volume_ratio"):
            vol_r = indicators.get("volume_ratio", 1.0)
            volume_score = 1.0 if vol_r > 1.2 else (-0.5 if vol_r < 0.8 else 0.0)

        # ===== 9. 综合评分 =====
        weights = {
            "tech": 0.25, "momentum": 0.20, "volume": 0.10,
            "gt": 0.15, "risk": 0.10, "ml": 0.20
        }
        overall = (
            tech_score * weights["tech"] +
            momentum_score * weights["momentum"] +
            volume_score * weights["volume"] +
            gt_score * weights["gt"] +
            (1 - risk_score) * weights["risk"] +  # 风险低则加分
            ml_score * weights["ml"]
        )
        overall = max(-1.0, min(1.0, overall))

        # ===== 10. 信号生成 =====
        if overall > 0.3: signal = "BUY"
        elif overall < -0.3: signal = "SELL"
        else: signal = "HOLD"
        confidence = abs(overall)

        # ===== 11. 止损止盈 =====
        atr_val = indicators.get("atr", close[-1] * 0.02) if indicators else close[-1] * 0.02
        atr_val = max(atr_val, close[-1] * 0.001)

        if signal == "BUY":
            entry = close[-1]
            stop = entry - atr_val * 2.0
            tp = entry + atr_val * 3.0
            rr = 3.0
        elif signal == "SELL":
            entry = close[-1]
            stop = entry + atr_val * 2.0
            tp = entry - atr_val * 3.0
            rr = 3.0
        else:
            entry = close[-1]
            stop = close[-1]
            tp = close[-1]
            rr = 0.0

        # ===== 12. 建议 =====
        recs = []
        warnings = []

        if overall > 0.5: recs.append("强买入信号，综合评分优秀")
        elif overall > 0.3: recs.append("温和买入信号")
        elif overall < -0.5: recs.append("强卖出信号，建议平仓")
        elif overall < -0.3: recs.append("温和卖出信号")
        else: recs.append("中性信号，等待确认")

        if risk_score > 0.7: warnings.append(f"⚠️ 风险较高 (VaR 95%: {risk_output.get('var_95', 0):.2%})")
        if gt_output.get("manipulation_prob", {}).get("is_manipulated", False):
            warnings.append("⚠️ 检测到价格操纵风险")
        if gt_output.get("whale_detection", {}).get("is_whale", False):
            warnings.append("🐋 检测到大户活动")
        if regime == MarketRegime.HIGH_VOLATILITY if MarketRegime else regime == "HIGH_VOL":
            warnings.append("⚡ 高波动环境，注意风险管理")
        if regime == MarketRegime.SIDEWAYS if MarketRegime else regime == "SIDEWAYS":
            recs.append("盘整市场，适合均值回归策略")
        if stats_output.get("mean_reversion_signal") == "TREND_FOLLOWING":
            recs.append(" Hurst > 0.6，适合趋势跟随策略")

        # ===== 13. 构建结果 =====
        result = FullAnalysisResult(
            symbol=symbol,
            timestamp=t0,
            regime=regime.value if hasattr(regime, 'value') else str(regime),
            regime_confidence=float(regime_confidence),
            technical_score=float(tech_score),
            momentum_score=float(momentum_score),
            mean_reversion_score=float(mr_score),
            volume_score=float(volume_score),
            game_theory_score=float(gt_score),
            ml_score=float(ml_score),
            risk_score=float(risk_score),
            options_score=0.0,
            overall_score=float(overall),
            signal=signal,
            signal_confidence=float(confidence),
            entry_price=float(entry),
            stop_loss=float(stop),
            take_profit=float(tp),
            risk_reward_ratio=float(rr),
            recommended_strategies=recommended_strategies,
            all_indicators=indicators,
            risk_metrics=risk_output,
            regime_context=regime_context,
            game_theory=gt_output,
            portfolio_metrics={},
            recommendations=recs,
            warnings=warnings
        )

        logger.info(f"分析完成: {result.summary()} | 耗时: {(datetime.now()-t0).total_seconds():.2f}s")
        return result

    def analyze_multi_symbol(self, data_dict: Dict[str, pd.DataFrame],
                              orderbooks: Dict[str, Dict] = None
                              ) -> Dict[str, FullAnalysisResult]:
        """分析多个标的，返回排序建议"""
        results = {}
        scores = {}

        for symbol, df in data_dict.items():
            try:
                ob = orderbooks.get(symbol) if orderbooks else None
                result = self.analyze(df, symbol=symbol, orderbook=ob)
                results[symbol] = result
                scores[symbol] = result.overall_score
            except Exception as e:
                logger.error(f"{symbol} 分析失败: {e}")

        # 按评分排序
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"多标的分析完成: {[(s, f'{sc:.2f}') for s, sc in ranked]}")

        return results

    def portfolio_analysis(self, df_dict: Dict[str, pd.DataFrame],
                           weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        组合层面的分析
        """
        if not self.risk or not self.portfolio:
            return {}

        returns_dict = {}
        prices_dict = {}

        for symbol, df in df_dict.items():
            close = df["close"].values
            ret = np.diff(close) / (close[:-1] + 1e-10)
            returns_dict[symbol] = ret
            prices_dict[symbol] = close

        symbols = list(returns_dict.keys())
        n = len(symbols)
        if n == 0: return {}

        min_len = min(len(r) for r in returns_dict.values())
        aligned_returns = np.column_stack([returns_dict[s][:min_len] for s in symbols])

        # 组合风险指标
        w = np.array([weights.get(s, 1.0/n) for s in symbols])
        w = w / w.sum()
        port_returns = aligned_returns @ w

        risk_metrics = self.risk.risk_metrics_all(port_returns, np.zeros(len(port_returns)))

        # 优化配置
        try:
            cov = np.cov(aligned_returns.T)
            exp_ret = np.array([np.mean(r) for r in aligned_returns.T])
            mv_weights = self.portfolio.mean_variance_optimizer(exp_ret, cov)
            rp_weights = self.portfolio.risk_parity(aligned_returns)
        except Exception as e:
            logger.warning(f"组合优化失败: {e}")
            mv_weights = w; rp_weights = w

        return {
            "portfolio_return": float(np.mean(port_returns) * 252),
            "portfolio_volatility": float(np.std(port_returns) * np.sqrt(252)),
            "sharpe": risk_metrics.get("sharpe_ratio", 0.0),
            "max_drawdown": risk_metrics.get("max_drawdown", 0.0),
            "var_95": risk_metrics.get("var_95", 0.0),
            "optimal_weights": dict(zip(symbols, mv_weights.tolist() if hasattr(mv_weights,'tolist') else [float(x) for x in mv_weights])),
            "risk_parity_weights": dict(zip(symbols, rp_weights.get("weights", []) if isinstance(rp_weights, dict) else [float(x) for x in rp_weights])),
            "correlation_matrix": self._build_corr_matrix(aligned_returns, symbols)
        }

    def _build_corr_matrix(self, returns: np.ndarray, symbols: List[str]) -> Dict:
        n = returns.shape[1]
        corr = np.corrcoef(returns.T)
        return {symbols[i]: {symbols[j]: float(corr[i,j]) for j in range(n)} for i in range(n)}


# =============================================================================
# 工厂函数
# =============================================================================

def create_engine(config: dict = None) -> UnifiedAnalyticsEngine:
    """创建统一分析引擎"""
    return UnifiedAnalyticsEngine(config)


# =============================================================================
# 演示 / 测试
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 生成模拟数据
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range(start="2024-01-01", periods=n)
    close = 100 * np.cumprod(1 + np.random.randn(n) * 0.02)
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    volume = np.random.randint(1000, 10000, n)

    df = pd.DataFrame({"date": dates, "open": close, "high": high, "low": low, "close": close, "volume": volume})
    df.set_index("date", inplace=True)

    # 创建引擎并分析
    engine = create_engine()
    result = engine.analyze(df, symbol="TEST/USDT")

    print("\n" + "="*70)
    print("统一分析引擎 - 测试结果")
    print("="*70)
    print(f"标的: {result.symbol}")
    print(f"信号: {result.signal} ({result.signal_confidence:.1%})")
    print(f"综合评分: {result.overall_score:+.3f}")
    print(f"市场状态: {result.regime} ({result.regime_confidence:.1%})")
    print(f"技术评分: {result.technical_score:+.3f}")
    print(f"动量评分: {result.momentum_score:+.3f}")
    print(f"博弈评分: {result.game_theory_score:+.3f}")
    print(f"风险评分: {result.risk_score:.3f}")
    print(f"入场价: {result.entry_price:.4f}")
    print(f"止损: {result.stop_loss:.4f}")
    print(f"止盈: {result.take_profit:.4f}")
    print(f"风险回报比: {result.risk_reward_ratio:.1f}")
    print(f"\n推荐策略:")
    for s in result.recommended_strategies:
        print(f"  - {s}")
    print(f"\n建议:")
    for r in result.recommendations:
        print(f"  • {r}")
    if result.warnings:
        print(f"\n警告:")
        for w in result.warnings:
            print(f"  ⚠️ {w}")
    print("\n关键指标:")
    for k, v in list(result.all_indicators.items())[:15]:
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\n风险指标:")
    for k, v in result.risk_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("="*70)
