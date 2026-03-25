"""
===================================================================
高级分析模块 (Advanced Analytics)
包含: 期权分析、风险分析、组合优化、订单簿分析
===================================================================
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 第 五 部 分 ： 期 权 分 析
# =============================================================================

class OptionsAnalyzer:
    """期权希腊值与波动率分析"""

    @staticmethod
    def black_scholes(S: float, K: float, T: float, r: float, sigma: float,
                      option_type: str = "call") -> Dict[str, float]:
        if T <= 0 or sigma <= 0:
            if option_type == "call": return {"price": max(0.0, S-K), "delta": 1.0 if S>K else 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
            else: return {"price": max(0.0, K-S), "delta": -1.0 if S<K else 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        from scipy.stats import norm
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T) + 1e-10)
        d2 = d1 - sigma*np.sqrt(T)
        nd1, nd2 = norm.cdf(d1), norm.cdf(d2)
        nnd1, nnd2 = norm.cdf(-d1), norm.cdf(-d2)
        if option_type == "call":
            price = S*nd1 - K*np.exp(-r*T)*nd2
            delta, rho = nd1, K*T*np.exp(-r*T)*nd2/100
        else:
            price = K*np.exp(-r*T)*nnd2 - S*nnd1
            delta, rho = nd1-1, -K*T*np.exp(-r*T)*nnd2/100
        gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T) + 1e-10)
        vega = S*norm.pdf(d1)*np.sqrt(T) / 100
        theta_call = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*nd2) / 365
        theta_put  = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*nnd2) / 365
        theta = theta_call if option_type == "call" else theta_put
        return {"price": float(price), "delta": float(delta), "gamma": float(gamma),
                "vega": float(vega), "theta": float(theta), "rho": float(rho),
                "d1": float(d1), "d2": float(d2)}

    @staticmethod
    def implied_volatility(price_market: float, S: float, K: float, T: float,
                          r: float, option_type: str = "call",
                          tol: float = 1e-6, max_iter: int = 100) -> float:
        """Newton-Raphson求解隐含波动率"""
        sigma = 0.20  # 初始猜测
        for _ in range(max_iter):
            result = OptionsAnalyzer.black_scholes(S, K, T, r, sigma, option_type)
            price_model = result["price"]
            vega = result["vega"] * 100
            if abs(vega) < 1e-10: break
            diff = price_market - price_model
            if abs(diff) < tol: break
            sigma += diff / vega
            sigma = max(0.01, min(sigma, 5.0))
        return float(sigma)

    @staticmethod
    def option_strategies(S: float, K: float, T: float, r: float, sigma: float
                          ) -> Dict[str, Dict[str, Any]]:
        """主要期权策略分析"""
        results = {}
        for ot in ["call", "put"]:
            bs = OptionsAnalyzer.black_scholes(S, K, T, r, sigma, ot)
            # 盈亏平衡点
            if ot == "call": breakeven = K + bs["price"]
            else: breakeven = K - bs["price"]
            results[ot] = {
                **bs, "breakeven": float(breakeven),
                "intrinsic_value": float(max(0, S-K) if ot=="call" else max(0, K-S)),
                "time_value": float(bs["price"] - max(0, S-K) if ot=="call" else bs["price"] - max(0, K-S))
            }
        # Straddle
        straddle_price = results["call"]["price"] + results["put"]["price"]
        results["straddle"] = {"price": float(straddle_price), "breakeven_up": float(S+straddle_price), "breakeven_down": float(S-straddle_price)}
        # Strangle
        otm_call = OptionsAnalyzer.black_scholes(S, K*1.05, T, r, sigma, "call")
        otm_put  = OptionsAnalyzer.black_scholes(S, K*0.95, T, r, sigma, "put")
        strangle_price = otm_call["price"] + otm_put["price"]
        results["strangle"] = {"price": float(strangle_price)}
        return results

    @staticmethod
    def volatility_smile(S: float, T: float, r: float,
                          strikes: List[float], prices: List[float]
                          ) -> Dict[str, Any]:
        """波动率微笑分析"""
        ivs = []
        for K, p in zip(strikes, prices):
            try: iv = OptionsAnalyzer.implied_volatility(p, S, K, T, r)
            except: iv = 0.30
            ivs.append(float(iv))
        skew = float(ivs[-1] - ivs[0]) if len(ivs) >= 2 else 0.0  # 斜率
        return {"implied_vols": dict(zip([str(k) for k in strikes], ivs)), "skew": skew}

    @staticmethod
    def delta_hedge(S: float, K: float, T: float, r: float, sigma: float,
                     option_type: str = "call") -> Dict[str, float]:
        """Delta中性对冲分析"""
        bs = OptionsAnalyzer.black_scholes(S, K, T, r, sigma, option_type)
        delta = bs["delta"]
        gamma = bs["gamma"]
        vega  = bs["vega"]
        shares_needed = -delta  # 对冲所需股票数量
        return {"shares_to_hedge": float(shares_needed), "delta": float(delta),
                "gamma_risk": float(gamma), "vega_risk": float(vega),
                "hedge_cost": float(abs(shares_needed * S))}


# =============================================================================
# 第 六 部 分 ： 风 险 分 析
# =============================================================================

class RiskAnalyzer:
    """风险度量与管理系统"""

    @staticmethod
    def historical_var(returns: np.ndarray, confidence: float = 0.95,
                       horizon: int = 1) -> float:
        """历史VaR"""
        ret = returns[~np.isnan(returns)]
        if len(ret) < 20: return 0.0
        var = float(np.percentile(ret, (1-confidence)*100))
        return var * np.sqrt(horizon)

    @staticmethod
    def parametric_var(returns: np.ndarray, confidence: float = 0.95,
                        horizon: int = 1) -> float:
        """参数化VaR (方差-协方差法)"""
        ret = returns[~np.isnan(returns)]
        if len(ret) < 5: return 0.0
        mu, sigma = np.mean(ret), np.std(ret)
        z = stats.norm.ppf(1 - confidence)
        var = -(mu + z * sigma) * np.sqrt(horizon)
        return float(max(0.0, var))

    @staticmethod
    def monte_carlo_var(returns: np.ndarray, confidence: float = 0.95,
                         n_simulations: int = 10000, horizon: int = 1) -> Dict[str, float]:
        """Monte Carlo VaR"""
        ret = returns[~np.isnan(returns)]
        if len(ret) < 10:
            return {"var_95": 0.0, "var_99": 0.0, "cvar_95": 0.0, "cvar_99": 0.0}
        mu, sigma = np.mean(ret), np.std(ret)
        sim_returns = np.random.normal(mu, sigma, n_simulations * horizon)
        var_95 = float(-np.percentile(sim_returns, 5))
        var_99 = float(-np.percentile(sim_returns, 1))
        cvar_95 = float(-np.mean(sim_returns[sim_returns <= -var_95]))
        cvar_99 = float(-np.mean(sim_returns[sim_returns <= -var_99]))
        return {"var_95": var_95, "var_99": var_99, "cvar_95": cvar_95, "cvar_99": cvar_99}

    @staticmethod
    def cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
        """条件VaR (Expected Shortfall)"""
        ret = returns[~np.isnan(returns)]
        if len(ret) < 10: return 0.0
        var = np.percentile(ret, (1-confidence)*100)
        cvar = float(-np.mean(ret[ret <= var]))
        return cvar

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
        """夏普比率"""
        ret = returns[~np.isnan(returns)]
        if len(ret) < 5 or np.std(ret) < 1e-10: return 0.0
        excess = np.mean(ret) - risk_free / 252
        return float(excess / np.std(ret) * np.sqrt(252))

    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0,
                       target_return: float = 0.0) -> float:
        """索提诺比率"""
        ret = returns[~np.isnan(returns)]
        if len(ret) < 5: return 0.0
        excess = np.mean(ret) - risk_free / 252
        downside = ret[ret < target_return]
        downside_std = np.std(downside) if len(downside) > 2 else 1e-10
        return float(excess / downside_std * np.sqrt(252))

    @staticmethod
    def max_drawdown(prices: np.ndarray) -> Tuple[float, int, int]:
        """最大回撤"""
        prices = np.array(prices)
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / (peak + 1e-10)
        mdd = float(np.min(drawdown))
        end_idx = int(np.argmin(drawdown))
        start_idx = int(np.argmax(prices[:end_idx])) if end_idx > 0 else 0
        return mdd, start_idx, end_idx

    @staticmethod
    def calmar_ratio(returns: np.ndarray, prices: np.ndarray) -> float:
        """卡尔马比率"""
        annual_return = float(np.mean(returns) * 252)
        mdd, _, _ = RiskAnalyzer.max_drawdown(prices)
        if abs(mdd) < 1e-10: return 0.0
        return float(annual_return / abs(mdd))

    @staticmethod
    def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
        """Omega比率"""
        ret = returns[~np.isnan(returns)]
        if len(ret) < 5: return 1.0
        gains = ret[ret > threshold]
        losses = ret[ret < threshold]
        total_gains = np.sum(gains - threshold) if len(gains) > 0 else 0.0
        total_losses = np.sum(threshold - losses) if len(losses) > 0 else 1e-10
        return float(total_gains / total_losses)

    @staticmethod
    def information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """信息比率"""
        ret = returns[~np.isnan(returns)]
        bm = benchmark_returns[~np.isnan(benchmark_returns)]
        n = min(len(ret), len(bm))
        if n < 5: return 0.0
        active_returns = ret[-n:] - bm[-n:]
        return float(np.mean(active_returns) / np.std(active_returns) * np.sqrt(252))

    @staticmethod
    def tail_ratio(returns: np.ndarray) -> float:
        """尾部比率"""
        ret = returns[~np.isnan(returns)]
        if len(ret) < 20: return 1.0
        p95 = np.percentile(ret, 95)
        p5  = abs(np.percentile(ret, 5))
        if p5 < 1e-10: return 1.0
        return float(p95 / p5)

    @staticmethod
    def kelly_criterion(returns: np.ndarray) -> Dict[str, float]:
        """Kelly方程 - 最优下注比例"""
        ret = returns[~np.isnan(returns)]
        if len(ret) < 10: return {"kelly_pct": 0.0, "half_kelly": 0.0, "quarter_kelly": 0.0}
        win_rate = np.mean(ret > 0)
        avg_win = np.mean(ret[ret > 0]) if np.any(ret > 0) else 0.0
        avg_loss = abs(np.mean(ret[ret < 0])) if np.any(ret < 0) else 1e-10
        if avg_loss < 1e-10: return {"kelly_pct": 0.0, "half_kelly": 0.0}
        win_loss_ratio = avg_win / avg_loss
        kelly = float((win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio)
        kelly = max(0.0, min(kelly, 1.0))
        return {"kelly_pct": kelly, "half_kelly": kelly/2, "quarter_kelly": kelly/4,
                "win_rate": float(win_rate), "win_loss_ratio": float(win_loss_ratio)}

    @staticmethod
    def beta_capm(returns: np.ndarray, market_returns: np.ndarray) -> Dict[str, float]:
        """CAPM Beta分析"""
        ret = returns[~np.isnan(returns)]
        mkt = market_returns[~np.isnan(market_returns)]
        n = min(len(ret), len(mkt))
        if n < 10: return {"beta": 1.0, "alpha": 0.0, "r_squared": 0.0}
        X = mkt[-n:]; y = ret[-n:]
        X_mean, y_mean = np.mean(X), np.mean(y)
        cov  = np.sum((X - X_mean) * (y - y_mean))
        var  = np.sum((X - X_mean)**2)
        beta = cov / (var + 1e-10)
        alpha = y_mean - beta * X_mean
        ss_res = np.sum((y - (alpha + beta*X))**2)
        ss_tot = np.sum((y - y_mean)**2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        return {"beta": float(beta), "alpha": float(alpha), "r_squared": float(r2)}

    @staticmethod
    def risk_metrics_all(returns: np.ndarray, prices: np.ndarray,
                          market_returns: np.ndarray = None) -> Dict[str, Any]:
        """综合风险指标"""
        ret = returns[~np.isnan(returns)]
        mc_var = RiskAnalyzer.monte_carlo_var(ret)
        kelly = RiskAnalyzer.kelly_criterion(ret)
        mdd, s_idx, e_idx = RiskAnalyzer.max_drawdown(prices)
        metrics = {
            "var_95": mc_var["var_95"], "var_99": mc_var["var_99"],
            "cvar_95": mc_var["cvar_95"], "cvar_99": mc_var["cvar_99"],
            "historical_var": RiskAnalyzer.historical_var(ret),
            "sharpe_ratio": RiskAnalyzer.sharpe_ratio(ret),
            "sortino_ratio": RiskAnalyzer.sortino_ratio(ret),
            "calmar_ratio": RiskAnalyzer.calmar_ratio(ret, prices),
            "omega_ratio": RiskAnalyzer.omega_ratio(ret),
            "tail_ratio": RiskAnalyzer.tail_ratio(ret),
            "max_drawdown": mdd, "max_drawdown_start": int(s_idx), "max_drawdown_end": int(e_idx),
            "kelly": kelly["kelly_pct"], "half_kelly": kelly["half_kelly"],
            "win_rate": kelly.get("win_rate", 0.0),
            "annual_return": float(np.mean(ret)*252),
            "annual_volatility": float(np.std(ret)*np.sqrt(252)),
        }
        if market_returns is not None:
            capm = RiskAnalyzer.beta_capm(ret, market_returns)
            metrics["beta"] = capm["beta"]; metrics["alpha"] = capm["alpha"]
            metrics["r_squared"] = capm["r_squared"]
        return metrics


# =============================================================================
# 第 七 部 分 ： 组 合 优 化
# =============================================================================

class PortfolioOptimizer:
    """投资组合优化器"""

    @staticmethod
    def markowitz(returns: np.ndarray, n_portfolios: int = 100,
                   risk_free: float = 0.0) -> Dict[str, Any]:
        """Markowitz有效前沿"""
        ret = returns[~np.isnan(returns)]
        if len(ret) < 10:
            return {"efficient_frontier": [], "optimal_weights": [], "min_var": {}, "max_sharpe": {}}
        mean_ret = np.mean(ret)
        std_ret = np.std(ret)
        n = len(ret)
        target_returns = np.linspace(mean_ret * 0.5, mean_ret * 1.5, n_portfolios)
        frontier = []
        for target in target_returns:
            if std_ret < 1e-10: continue
            sharpe = (target - risk_free/252) / std_ret
            frontier.append({"return": float(target*252), "volatility": float(std_ret*np.sqrt(252)), "sharpe": float(sharpe)})
        frontier.sort(key=lambda x: x["volatility"])
        max_sharpe = max(frontier, key=lambda x: x["sharpe"]) if frontier else {}
        min_var = min(frontier, key=lambda x: x["volatility"]) if frontier else {}
        return {"efficient_frontier": frontier, "max_sharpe": max_sharpe, "min_variance": min_var}

    @staticmethod
    def black_litterman(returns: np.ndarray, market_cap_weights: np.ndarray,
                         views: Dict[int, float] = None,
                         view_confidence: float = 0.5, risk_aversion: float = 2.5
                         ) -> Dict[str, Any]:
        """Black-Litterman资产配置模型"""
        ret = returns[~np.isnan(returns)]
        if len(ret) < 10: return {"weights": {}}
        cov = np.cov(ret.reshape(-1,1)) if len(ret.shape) == 1 else np.cov(ret.T)
        n = len(market_cap_weights)
        # 市场均衡收益率
        pi = risk_aversion * cov.dot(market_cap_weights)
        # 合并观点
        weights = market_cap_weights.copy()
        if views:
            for idx, view_return in views.items():
                if 0 <= idx < n:
                    weights[idx] = weights[idx] * (1 - view_confidence) + view_confidence * (view_return / (risk_aversion + 1e-10))
        weights = np.maximum(weights, 0)
        weights = weights / (np.sum(weights) + 1e-10)
        return {"equilibrium_returns": pi.tolist() if hasattr(pi,"tolist") else [float(pi)], "adjusted_weights": weights.tolist() if hasattr(weights,"tolist") else [float(weights)]}

    @staticmethod
    def risk_parity(returns: np.ndarray) -> Dict[str, Any]:
        """风险平价组合"""
        ret = returns[~np.isnan(returns)]
        if len(ret) < 5: return {"weights": {}}
        cov = np.cov(ret.reshape(-1,1)) if len(ret.shape) == 1 else np.cov(ret.T)
        if len(cov.shape) == 0: cov = np.array([[float(cov)**2]])
        n = cov.shape[0]
        def risk_contribution(w, cov_mat):
            port_var = w @ cov_mat @ w
            marginal = cov_mat @ w
            contrib = w * marginal / (port_var**0.5 + 1e-10)
            return contrib / (np.sum(contrib) + 1e-10)
        w = np.ones(n) / n
        for _ in range(200):
            rc = risk_contribution(w, cov)
            target_rc = np.ones(n) / n
            gradient = rc - target_rc
            w = w - 0.1 * gradient
            w = np.maximum(w, 0.001)
            w = w / (np.sum(w) + 1e-10)
        return {"weights": w.tolist() if hasattr(w,"tolist") else [float(w)]}

    @staticmethod
    def mean_variance_optimizer(expected_returns: np.ndarray, cov_matrix: np.ndarray,
                                 risk_aversion: float = 1.0,
                                 constraints: Dict = None) -> np.ndarray:
        """均值-方差优化 (解析解)"""
        n = len(expected_returns)
        ones = np.ones(n)
        try:
            cov_inv = np.linalg.inv(cov_matrix + 1e-6*np.eye(n))
            A = float(ones @ cov_inv @ ones)
            B = float(ones @ cov_inv @ expected_returns)
            C = float(expected_returns @ cov_inv @ expected_returns)
            denom = A * C - B**2
            if abs(denom) < 1e-10:
                return np.ones(n) / n
            w = cov_inv @ ((A * expected_returns - B * ones) / denom * 1/risk_aversion + B * ones / denom)
            w = np.maximum(w, 0.0)
            return w / (np.sum(w) + 1e-10)
        except:
            return np.ones(n) / n

    @staticmethod
    def minimum_variance(cov_matrix: np.ndarray) -> np.ndarray:
        """最小方差组合"""
        n = cov_matrix.shape[0]
        try:
            cov_inv = np.linalg.inv(cov_matrix + 1e-6*np.eye(n))
            ones = np.ones(n)
            w = cov_inv @ ones
            w = np.maximum(w, 0.0)
            return w / (np.sum(w) + 1e-10)
        except:
            return np.ones(n) / n

    @staticmethod
    def maximum_diversification(expected_vol: np.ndarray, cov_matrix: np.ndarray
                                 ) -> np.ndarray:
        """最大分散化组合"""
        n = len(expected_vol)
        weights = expected_vol / (expected_vol.sum() + 1e-10)
        weights = np.maximum(weights, 0.0)
        return weights / (np.sum(weights) + 1e-10)


# =============================================================================
# 第 八 部 分 ： 订 单 簿 分 析
# =============================================================================

class OrderBookAnalyzer:
    """订单簿与市场微观结构分析"""

    @staticmethod
    def analyze_orderbook(bids: List[Tuple[float, float]],
                           asks: List[Tuple[float, float]],
                           mid_price: float = None) -> Dict[str, Any]:
        """
        分析订单簿
        Args:
            bids: [(price, volume), ...] 买方深度
            asks: [(price, volume), ...] 卖方深度
        """
        if not bids or not asks:
            return {"spread": 0.0, "imbalance": 0.0, "bid_depth": 0.0, "ask_depth": 0.0, "score": 0.0}
        best_bid, best_ask = bids[0][0], asks[0][0]
        spread = (best_ask - best_bid) / ((best_bid + best_ask)/2 + 1e-10)
        bid_vols = [b[1] for b in bids[:10]]
        ask_vols = [a[1] for a in asks[:10]]
        bid_depth = sum(bid_vols)
        ask_depth = sum(ask_vols)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-10)
        # VWAP深度
        bid_vwap = sum(b[0]*b[1] for b in bids[:5]) / (sum(b[1] for b in bids[:5]) + 1e-10)
        ask_vwap = sum(a[0]*a[1] for a in asks[:5]) / (sum(a[1] for a in asks[:5]) + 1e-10)
        # 订单集中度
        top_bid_concentration = bid_vols[0] / (bid_depth + 1e-10)
        top_ask_concentration = ask_vols[0] / (ask_depth + 1e-10)
        score = imbalance * 0.6 + (top_bid_concentration - top_ask_concentration) * 0.4
        return {
            "spread": float(spread), "imbalance": float(imbalance),
            "bid_depth": float(bid_depth), "ask_depth": float(ask_depth),
            "bid_vwap": float(bid_vwap), "ask_vwap": float(ask_vwap),
            "top_bid_concentration": float(top_bid_concentration),
            "top_ask_concentration": float(top_ask_concentration),
            "score": float(max(-1, min(1, score))),
            "bid_levels": len(bids), "ask_levels": len(asks)
        }

    @staticmethod
    def estimate_market_impact(volume: float, avg_daily_volume: float,
                               volatility: float) -> Dict[str, float]:
        """
        估算市场冲击成本 (Almgren-Chriss简化模型)
        """
        participation_rate = volume / (avg_daily_volume + 1e-10)
        # 冲击 = sigma * sqrt(participation_rate)
        impact = volatility * np.sqrt(max(0, participation_rate))
        # 临时冲击（立即恢复）
        temp_impact = volatility * participation_rate * 0.5
        # 永久冲击（不会恢复）
        perm_impact = volatility * participation_rate * 0.1
        return {
            "total_impact": float(impact),
            "temporary_impact": float(temp_impact),
            "permanent_impact": float(perm_impact),
            "participation_rate": float(participation_rate),
            "estimated_spread_cost": float(volatility * 0.01 * participation_rate)
        }

    @staticmethod
    def microstrucutre_index(df: pd.DataFrame) -> Dict[str, Any]:
        """
        基于K线数据的简化微观结构指标
        """
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        volume = df["volume"].values if "volume" in df.columns else np.ones_like(close)
        n = len(close)
        if n < 5: return {"amihud_illiquidity": 0.0, "micheleyski_prob": 0.0}
        ret = np.diff(close) / (close[:-1] + 1e-10)
        # Amihud非流动性
        abs_ret = np.abs(ret)
        vol = volume[:-1] + 1e-10
        amihud = np.mean(abs_ret / vol)
        # Roll派恩价差估算
        if n >= 20:
            autocov = np.cov(ret[-20:-1], ret[-19:])[0,1]
            roll_spread = 2 * np.sqrt(max(0, -autocov))
        else:
            roll_spread = 0.0
        return {
            "amihud_illiquidity": float(amihud),
            "roll_estimated_spread": float(roll_spread),
            "price_impact": float(amihud * np.mean(vol)),
        }
