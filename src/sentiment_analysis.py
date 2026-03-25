"""
===================================================================
情绪分析模块 (Sentiment Analysis)
集成新闻摘要 + 社交媒体情绪 + FUD/FOMO 指数
===================================================================
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    score: float          # -1 (极度恐慌) ~ +1 (极度贪婪)
    fomo_index: float     # 0~1
    fear_greed_index: float
    news_count: int
    social_score: float
    whale_sentiment: float
    overall: float        # 综合情绪信号
    details: Dict


class SentimentAnalyzer:
    """
    多源情绪分析：
    1. 财经新闻关键词打分
    2. 社交媒体情绪（Twitter/Reddit 风格关键词）
    3. 鲸鱼地址异动（链上代理指标）
    4. K线实体/影线比例（替代传统Fear & Greed指数）
    """

    # 关键词词典
    POSITIVE_WORDS = {
        "bullish", "bull", "moon", "pump", "breakout", "surge",
        "soar", "rally", "gain", "rise", "uptrend", "high",
        "growth", "adoption", "buy signal", "golden cross",
        "institutional", "ETF", "approval", "halving", "upgrade",
        "partnership", "launch", "rallying", "recover", "boom",
        "all-time high", "ATH", "new high", "profit", "yield",
    }
    NEGATIVE_WORDS = {
        "bearish", "bear", "dump", "crash", "plunge", "drop",
        "sell-off", "rejection", "hack", "ban", "regulation",
        "fear", "uncertainty", "risk", "liquidation", "margin",
        "whale sell", "outflow", "death cross", "breakdown",
        "panic", "warn", "investor", "caution", "volatile",
        "delist", "fraud", "scam", "correction", "downtrend",
        "all-time low", "ATL", "new low", "loss", "decline",
    }
    FOMO_WORDS = {"FOMO", "buying", "chase", "entry", "accumulation", "dip"}
    FEAR_WORDS = {"FUD", "sell", "exit", "panic", "safe", "protect"}

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._news_cache: Dict[str, SentimentResult] = {}
        self._cache_ttl_minutes = 10

    def analyze(
        self,
        symbol: str,
        news_texts: Optional[List[str]] = None,
        price_data: Optional[pd.DataFrame] = None,
        social_signals: Optional[Dict] = None,
        use_cache: bool = True,
    ) -> SentimentResult:
        """
        综合多源情绪分析

        Args:
            symbol: 交易对符号 (e.g. "BTC")
            news_texts: 最新新闻文本列表
            price_data: 最近K线 DataFrame (date, open, high, low, close, volume)
            social_signals: 社交媒体信号 dict，如 {"tweets_positive": 120, "tweets_negative": 40}
            use_cache: 是否使用缓存
        """
        cache_key = symbol
        if use_cache and cache_key in self._news_cache:
            cached = self._news_cache[cache_key]
            # 缓存10分钟内有效
            if hasattr(cached, "_fetch_time"):
                age = (datetime.now() - cached._fetch_time).total_seconds() / 60
                if age < self._cache_ttl_minutes:
                    return cached

        # ---- 1. 新闻情绪 ----
        news_score, news_count = self._analyze_news(news_texts or [])

        # ---- 2. 价格形态情绪 (K线结构) ----
        candle_score = self._analyze_candles(price_data) if price_data is not None else 0.0

        # ---- 3. 社交媒体信号 ----
        social_score = self._analyze_social(social_signals or {})

        # ---- 4. 综合计算 ----
        # 各维度权重
        w_news = 0.40
        w_candle = 0.30
        w_social = 0.30

        sentiment_raw = (
            news_score * w_news +
            candle_score * w_candle +
            social_score * w_social
        )
        sentiment = float(np.tanh(sentiment_raw))  # [-1, 1]

        # Fear & Greed Index (映射到 0~100)
        fear_greed = float((sentiment + 1) / 2 * 100)

        # FOMO 指数
        fomo = self._calc_fomo_index(news_texts or [], social_signals or {}, price_data)

        # 整体信号（增强版）
        overall = self._signal(sentiment, fomo, fear_greed)

        result = SentimentResult(
            score=sentiment,
            fomo_index=fomo,
            fear_greed_index=fear_greed,
            news_count=news_count,
            social_score=social_score,
            whale_sentiment=candle_score,
            overall=overall,
            details={
                "news_score": news_score,
                "candle_score": candle_score,
                "social_score": social_score,
                "sentiment_label": self._label(sentiment),
                "fg_label": self._fg_label(fear_greed),
            }
        )
        result._fetch_time = datetime.now()
        self._news_cache[cache_key] = result
        return result

    def _analyze_news(self, texts: List[str]) -> tuple[float, int]:
        """基于关键词的新闻情绪打分"""
        if not texts:
            return 0.0, 0
        scores = []
        for text in texts:
            text_lower = text.lower()
            pos = sum(1 for w in self.POSITIVE_WORDS if w in text_lower)
            neg = sum(1 for w in self.NEGATIVE_WORDS if w in text_lower)
            score = (pos - neg) / (pos + neg + 1)
            scores.append(score)
        avg = float(np.mean(scores)) if scores else 0.0
        return avg, len(texts)

    def _analyze_candles(self, df: pd.DataFrame) -> float:
        """
        基于K线形态评估市场情绪
        原理：实体大+影线短=趋势确认；上下影线比例反映多空博弈
        """
        c = np.asarray(df["close"].values, dtype=float)
        h = np.asarray(df["high"].values, dtype=float)
        lo = np.asarray(df["low"].values, dtype=float)
        o = np.asarray(df["open"].values, dtype=float)
        n = len(c)
        if n < 5:
            return 0.0

        # 最近 N 根K线
        N = min(20, n)
        c_, h_, lo_, o_ = c[-N:], h[-N:], lo[-N:], o[-N:]

        # 1) 实体比例 (body / range)
        body = np.abs(c_ - o_)
        range_ = h_ - lo_ + 1e-10
        entity_ratio = body / range_

        # 2) 上/下影线比例
        upper_shadow = h_ - np.maximum(o_, c_)
        lower_shadow = np.minimum(o_, c_) - lo_
        total_shadow = upper_shadow + lower_shadow + 1e-10
        upper_ratio = upper_shadow / total_shadow
        lower_ratio = lower_shadow / total_shadow

        # 3) 趋势方向
        trend = 1.0 if c[-1] > c[-N] else (-1.0 if c[-1] < c[-N * 2 // 3] else 0.0)

        # 4) 成交量变化
        if "volume" in df.columns:
            v = np.asarray(df["volume"].values, dtype=float)
            vr = float(v[-1] / (np.mean(v[-N:]) + 1e-10)) if n >= N else 1.0
        else:
            vr = 1.0

        # 综合打分
        bull_power = np.mean(lower_ratio[-5:]) - np.mean(upper_ratio[-5:])
        vol_effect = 0.5 if vr > 1.2 else (-0.3 if vr < 0.8 else 0.0)
        entity_avg = float(np.mean(entity_ratio[-5:]))
        score = (np.mean(bull_power) * 0.4 + entity_avg * trend * 0.3 + vol_effect * 0.3)
        return float(max(-1.0, min(1.0, score)))

    def _analyze_social(self, signals: Dict) -> float:
        """基于社交媒体数据的情绪"""
        if not signals:
            return 0.0
        pos = float(signals.get("positive_count", 0))
        neg = float(signals.get("negative_count", 0))
        mentions = float(signals.get("total_mentions", max(pos + neg, 1)))
        engagement = float(signals.get("total_engagement", 0))

        # 加权情绪
        raw = (pos - neg) / (pos + neg + 1)
        # 提及量放大因子
        vol_factor = float(np.log1p(mentions) / np.log1p(10000))
        # 互动率影响
        engage_factor = float(np.tanh(engagement / 1e6))

        score = raw * (0.6 + 0.2 * vol_factor + 0.2 * engage_factor)
        return float(max(-1.0, min(1.0, score)))

    def _calc_fomo_index(
        self,
        texts: List[str],
        social: Dict,
        price_data: Optional[pd.DataFrame],
    ) -> float:
        """FOMO 指数 0~1"""
        fomo_score = 0.0

        # 新闻FOMO词频
        if texts:
            fomo_count = sum(
                sum(1 for w in self.FOMO_WORDS if w in t.upper())
                for t in texts
            )
            fear_count = sum(
                sum(1 for w in self.FEAR_WORDS if w in t.upper())
                for t in texts
            )
            fomo_score += (fomo_count - fear_count) / (len(texts) + 1) * 0.4

        # 社交信号FOMO
        if social:
            vol_ratio = float(social.get("volume_ratio", 1.0))
            if vol_ratio > 1.5:
                fomo_score += 0.3
            active_users = float(social.get("active_users", 0))
            if active_users > 100000:
                fomo_score += 0.2 * min(1.0, active_users / 500000)

        # 价格FOMO（急涨+放量）
        if price_data is not None:
            c = np.asarray(price_data["close"].values, dtype=float)
            n = len(c)
            if n >= 5:
                pchg = float(abs(c[-1] - c[-5]) / (c[-5] + 1e-10))
                if "volume" in price_data.columns:
                    v = np.asarray(price_data["volume"].values, dtype=float)
                    vr = float(v[-1] / (np.mean(v[-10:]) + 1e-10)) if n >= 10 else 1.0
                    if pchg > 0.03 and vr > 1.5:
                        fomo_score += 0.2

        return float(max(0.0, min(1.0, fomo_score)))

    def _signal(self, sentiment: float, fomo: float, fg: float) -> float:
        """综合情绪信号，分数 -1 ~ +1"""
        # 高FOMO + 高贪婪 = 警告信号（反向）
        if fomo > 0.7 and fg > 75:
            return sentiment * 0.5  # 减弱，追高危险
        if fomo < 0.2 and fg < 25:
            return sentiment * 0.6 + 0.4 * (-1)  # 抄底机会
        return sentiment

    def _label(self, score: float) -> str:
        if score >= 0.6: return "STRONG_BULLISH"
        elif score >= 0.2: return "BULLISH"
        elif score >= -0.2: return "NEUTRAL"
        elif score >= -0.6: return "BEARISH"
        else: return "STRONG_BEARISH"

    def _fg_label(self, fg: float) -> str:
        if fg >= 75: return "Extreme Greed"
        elif fg >= 55: return "Greed"
        elif fg >= 45: return "Neutral"
        elif fg >= 25: return "Fear"
        else: return "Extreme Fear"

    # ---- 便捷方法 ----
    def quick_score(self, texts: List[str]) -> SentimentResult:
        """仅用新闻文本快速打分"""
        return self.analyze(symbol="QUICK", news_texts=texts,
                            price_data=None, social_signals={})
