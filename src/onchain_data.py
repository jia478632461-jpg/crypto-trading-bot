"""
===================================================================
链上数据分析模块 (On-Chain Data)
数据源：CoinGecko 公开 API + Binance 公开接口
无需 API Key，免费调用
===================================================================
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import json
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


@dataclass
class OnChainResult:
    score: float          # -1 (看跌) ~ +1 (看涨)
    exchange_flow_ratio: float   # 交易所净流入/流出
    whale_tx_count: int          # 大额转账笔数
    market_domination: float      # 主流币市值占比
    long_short_ratio: float       # 多空比 (Binance Funding)
    funding_rate: float           # 资金费率
    open_interest_change: float   # 未平仓合约变化 %
    details: Dict[str, Any]


class OnChainAnalyzer:
    """
    链上 + 交易所数据：

    数据源（全部免费，无需 API Key）：
    1. CoinGecko /coins/{id}/market_chart — 历史市值、交易量
    2. Binance /ticker/24hr — 交易所行情补充
    3. Glassnode (部分免费公共端点) 或 自计算指标

    派生指标（无需外部数据）：
    - 链上实现价格 (Realized Price)
    - MVRV 比率
    - 活跃地址变化
    - 大户持币地址数变化
    """

    COINGECKO_BASE = "https://api.coingecko.com/api/v3"
    BINANCE_BASE = "https://api.binance.com/api/v3"

    # 交易所充值地址（用于链上监控，可扩展）
    KNOWN_EXCHANGE_ADDRESSES = {
        "binance": ["0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be"],
        "coinbase": ["0xa903c6a028c8a4a73b2b8d7f9c3c9b8c9b9a8f7e"],
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._cache: Dict[str, tuple[datetime, Any]] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._session_headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; CryptoBot/1.0)",
        }

    def analyze(
        self,
        symbol: str,
        price_history: Optional[pd.DataFrame] = None,
        days: int = 30,
    ) -> OnChainResult:
        """
        综合链上 + 交易所数据分析

        Args:
            symbol: 币种符号 (e.g. "BTC", "ETH")
            price_history: 最近至少30天的 pd.DataFrame (date, close, volume, high, low)
            days: 获取历史数据的天数
        """
        coin_id = self._symbol_to_coingecko_id(symbol)
        cache_key = f"{symbol}_onchain"
        now = datetime.now()

        # 取缓存
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if now - ts < self._cache_ttl:
                return data

        # 1) 尝试获取 CoinGecko 市场数据
        market_data = self._fetch_coingecko_market(coin_id, days)

        # 2) 尝试获取 Binance 资金费率（永续合约）
        funding_rate = 0.0
        try:
            funding_rate = self._fetch_binance_funding(symbol)
        except Exception:
            pass

        # 3) 自计算链上代理指标（来自价格历史）
        derived = self._derive_onchain_metrics(price_history)

        # 4) 综合评分
        score = self._composite_score(
            market_data=market_data,
            derived=derived,
            funding_rate=funding_rate,
        )

        result = OnChainResult(
            score=score,
            exchange_flow_ratio=derived.get("exchange_flow_ratio", 1.0),
            whale_tx_count=int(derived.get("whale_tx_count", 0)),
            market_domination=derived.get("domination", market_data.get("mkt_cap_share", 0)),
            long_short_ratio=derived.get("long_short_ratio", 1.0),
            funding_rate=funding_rate,
            open_interest_change=derived.get("open_interest_change", 0.0),
            details={
                "coin_id": coin_id,
                "market_data": market_data,
                "derived_metrics": derived,
                "signal": self._signal_label(score),
            }
        )

        self._cache[cache_key] = (now, result)
        return result

    def _http_get(self, url: str, timeout: int = 8) -> Optional[Dict]:
        """HTTP GET，返回 dict 或 None"""
        try:
            req = urllib.request.Request(url, headers=self._session_headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            logger.debug(f"HTTP GET failed for {url}: {e}")
            return None

    def _symbol_to_coingecko_id(self, symbol: str) -> str:
        mapping = {
            "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
            "SOL": "solana", "XRP": "ripple", "ADA": "cardano",
            "DOGE": "dogecoin", "DOT": "polkadot", "MATIC": "matic-network",
            "LINK": "chainlink", "AVAX": "avalanche-2", "UNI": "uniswap",
            "ATOM": "cosmos", "LTC": "litecoin", "BCH": "bitcoin-cash",
        }
        sym = symbol.upper().replace("USDT", "").replace("BUSD", "").replace("USD", "")
        return mapping.get(sym, sym.lower())

    def _fetch_coingecko_market(self, coin_id: str, days: int) -> Dict:
        """获取 CoinGecko 市场数据"""
        url = f"{self.COINGECKO_BASE}/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
        data = self._http_get(url)
        if not data:
            return {}
        try:
            prices = data.get("market_caps", [])
            totals = data.get("total_volumes", [])
            if prices and totals:
                mcaps = [p[1] for p in prices]
                vols = [v[1] for v in totals]
                current_mcap = float(mcaps[-1]) if mcaps else 0.0
                mcap_change_30d = float((mcaps[-1] - mcaps[0]) / (mcaps[0] + 1)) if mcaps[0] > 0 else 0.0
                avg_volume_30d = float(np.mean(vols[-30:])) if vols else 0.0
                return {
                    "market_cap": current_mcap,
                    "mkt_cap_change_30d": mcap_change_30d,
                    "avg_volume_30d": avg_volume_30d,
                    "vol_to_mcap": avg_volume_30d / (current_mcap + 1),
                }
        except Exception as e:
            logger.debug(f"CoinGecko parse error: {e}")
        return {}

    def _fetch_binance_funding(self, symbol: str) -> float:
        """获取 Binance 永续合约资金费率"""
        sym = symbol.upper().replace("USDT", "USDT")
        url = f"{self.BINANCE_BASE}/premiumIndex?symbol={sym}"
        data = self._http_get(url)
        if data and "lastFundingRate" in data:
            return float(data["lastFundingRate"])
        return 0.0

    def _derive_onchain_metrics(
        self,
        price_data: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        从价格历史 + 成交量 自计算链上代理指标
        无需外部API
        """
        result = {}
        if price_data is None:
            return result

        c = np.asarray(price_data["close"].values, dtype=float)
        v = np.asarray(price_data["volume"].values, dtype=float) if "volume" in price_data.columns else np.ones_like(c)
        h = np.asarray(price_data["high"].values, dtype=float) if "high" in price_data.columns else c * 1.01
        lo = np.asarray(price_data["low"].values, dtype=float) if "low" in price_data.columns else c * 0.99
        n = len(c)
        if n < 7:
            return result

        # 1) MVRV 比率（简化版：当前价 / 实现价格代理）
        # 实现价格 ≈ 30天平均买入成本
        realized_price = float(np.mean(c[-30:])) if n >= 30 else float(np.mean(c))
        mvrv = float(c[-1] / (realized_price + 1e-10))
        result["mvrv"] = mvrv
        result["mvrv_signal"] = "OVERVALUED" if mvrv > 3.5 else ("UNDERVALUED" if mvrv < 1.5 else "FAIR")

        # 2) 交易所净流量（代理：净成交量变化）
        # 假设：高成交量+价格下跌 = 交易所净流入（卖出提币）
        #      高成交量+价格上涨 = 交易所净流出（抄底积累）
        returns = np.diff(c) / (c[:-1] + 1e-10)
        vol_norm = (v[1:] - np.mean(v)) / (np.std(v) + 1e-10)
        price_dir = np.where(returns > 0, 1.0, -1.0)
        flow_signal = float(np.mean(vol_norm * price_dir))
        # 归一化到 [-1, 1]
        result["exchange_flow_ratio"] = float(np.tanh(flow_signal * 0.5))
        result["exchange_flow_label"] = "INFLOW" if flow_signal < 0 else "OUTFLOW"

        # 3) 鲸鱼转账代理（成交量突增 + 价格异动）
        vol_ma = np.mean(v[-20:]) if n >= 20 else np.mean(v)
        vol_std = np.std(v[-20:]) if n >= 20 else 1.0
        z_vol = float((v[-1] - vol_ma) / (vol_std + 1e-10))
        price_z = float(abs(returns[-1]) / (np.std(returns[-20:]) + 1e-10)) if n >= 20 else 0.0
        whale_score = z_vol + price_z
        # 估算鲸鱼转账笔数（成交量 / 平均单笔金额）
        avg_trade_size_usd = 10000  # 假设平均每笔1万美元
        est_total_value = v[-1] * c[-1]
        est_tx_count = int(est_total_value / avg_trade_size_usd)
        # z_score > 2 视为大额活动
        result["whale_tx_count"] = est_tx_count if z_vol > 1.5 else est_tx_count // 5
        result["whale_activity"] = "HIGH" if z_vol > 2 else ("MEDIUM" if z_vol > 1 else "LOW")

        # 4) 长期持有者压力（短期/长期波动率比）
        short_vol = float(np.std(returns[-7:])) if n >= 7 else 0.0
        long_vol = float(np.std(returns[-30:])) if n >= 30 else short_vol + 1e-10
        vol_regime = short_vol / (long_vol + 1e-10)
        result["vol_regime"] = vol_regime
        result["vol_regime_label"] = "VOLATILE" if vol_regime > 1.5 else ("CALM" if vol_regime < 0.7 else "NORMAL")

        # 5) 多空比代理（基于OI变化 + 资金费率方向）
        # 简化：OI增加+上涨=多头占优；OI增加+下跌=空头占优
        if "open_interest" in price_data.columns:
            oi = np.asarray(price_data["open_interest"].values, dtype=float)
            if len(oi) >= 2 and oi[-2] > 0:
                oi_change = float((oi[-1] - oi[-2]) / oi[-2])
                result["open_interest_change"] = oi_change
                if oi_change > 0.1 and returns[-1] > 0:
                    result["long_short_ratio"] = 1.3
                elif oi_change > 0.1 and returns[-1] < 0:
                    result["long_short_ratio"] = 0.7
                else:
                    result["long_short_ratio"] = 1.0
        else:
            # 用成交量推断
            vol7 = float(np.mean(v[-7:])) if n >= 7 else v[-1]
            vol30 = float(np.mean(v[-30:])) if n >= 30 else vol7
            vol_ratio_7_30 = vol7 / (vol30 + 1e-10)
            if vol_ratio_7_30 > 1.5 and returns[-1] > 0:
                result["long_short_ratio"] = min(1.5, 1.0 + (vol_ratio_7_30 - 1) * 0.3)
            elif vol_ratio_7_30 > 1.5 and returns[-1] < 0:
                result["long_short_ratio"] = max(0.5, 1.0 - (vol_ratio_7_30 - 1) * 0.3)
            else:
                result["long_short_ratio"] = 1.0

        # 6) 主导地位（BTC市值/总市值代理）
        # 简化：用BTC相关价格数据
        # 这里返回0，因为需要多币种数据
        result["domination"] = 0.0

        return result

    def _composite_score(
        self,
        market_data: Dict,
        derived: Dict,
        funding_rate: float,
    ) -> float:
        """综合链上指标打分 -1 ~ +1"""
        scores = []

        # MVRV 评分
        mvrv = derived.get("mvrv", 1.0)
        if mvrv < 1.0:
            scores.append(0.8)   # 严重低估
        elif mvrv < 1.5:
            scores.append(0.4)  # 低估
        elif mvrv > 4.0:
            scores.append(-0.6) # 严重高估
        elif mvrv > 3.0:
            scores.append(-0.2) # 高估

        # 交易所流动态势
        flow = derived.get("exchange_flow_ratio", 0.0)
        # 净流出(+1) = 币在流向冷钱包 = 上涨信号
        # 净流入(-1) = 币在流向交易所 = 下跌信号
        scores.append(flow * 0.25)

        # 资金费率
        if funding_rate > 0.01:    # > 1% 多头被做空者支付
            scores.append(-0.3)
        elif funding_rate < -0.01: # 空头被多头支付
            scores.append(0.3)

        # 波动率状态
        vol_regime = derived.get("vol_regime", 1.0)
        if vol_regime > 2.0:
            scores.append(-0.2)   # 短期波动剧烈，可能反转
        elif vol_regime < 0.6:
            scores.append(0.15)  # 低波动积蓄

        # 30天市值变化
        mkt_cap_change = market_data.get("mkt_cap_change_30d", 0.0)
        if mkt_cap_change > 0.3:
            scores.append(0.3)
        elif mkt_cap_change < -0.3:
            scores.append(-0.3)

        # 成交量/市值比（链上活跃度）
        vol_ratio = market_data.get("vol_to_mcap", 0.0)
        if vol_ratio > 0.15:
            scores.append(0.2)  # 高活跃度
        elif vol_ratio < 0.03:
            scores.append(-0.1)

        # 鲸鱼活动
        whale = derived.get("whale_activity", "LOW")
        if whale == "HIGH":
            scores.append(-0.15)  # 鲸鱼活跃往往是见顶信号

        if not scores:
            return 0.0
        raw = sum(scores) / len(scores)
        return float(max(-1.0, min(1.0, raw)))

    def _signal_label(self, score: float) -> str:
        if score >= 0.5: return "STRONG_ONCHAIN_BUY"
        elif score >= 0.2: return "ONCHAIN_BUY"
        elif score >= -0.2: return "ONCHAIN_NEUTRAL"
        elif score >= -0.5: return "ONCHAIN_SELL"
        else: return "STRONG_ONCHAIN_SELL"
