"""
Statistical Models & Game Theory Analyzer
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple


class StatisticalModels:

    @staticmethod
    def hurst_exponent(c: np.ndarray, ml=100) -> float:
        lags = range(2, min(ml, len(c) // 2))
        tau_list, msd = [], []
        for lg in lags:
            d = c[lg:] - c[:-lg]
            if len(d) > 0:
                tau_list.append(lg)
                msd.append(np.mean(d ** 2))
        if not msd:
            return 0.5
        poly = np.polyfit(np.log(tau_list), np.log(msd), 1)
        return float(max(0.0, min(1.0, poly[0] / 2)))

    @staticmethod
    def half_life(c: np.ndarray) -> float:
        if len(c) < 5:
            return 100.0
        lag = c[:-1]
        delta = c[1:] - lag
        lc = lag - np.mean(lag)
        theta = np.sum(lc * delta) / (np.sum(lc ** 2) + 1e-10)
        return max(0.0, min(-np.log(2) / theta if theta < 0 else 500.0, 500.0))

    @staticmethod
    def z_score(c: np.ndarray, p=20) -> float:
        ma = pd.Series(c).rolling(p).mean().values
        std = pd.Series(c).rolling(p).std().values
        z = (c[-1] - ma[-1]) / (std[-1] + 1e-10)
        return float(z) if not np.isnan(z) else 0.0

    @staticmethod
    def mean_reversion_signal(c: np.ndarray, p=20, z_entry=2.0):
        z = StatisticalModels.z_score(c, p)
        h = StatisticalModels.hurst_exponent(c)
        if h > 0.6:
            return "TREND_FOLLOWING", float(h)
        if abs(z) > z_entry:
            return "BUY" if z < 0 else "SELL", min(1.0, (abs(z) - z_entry) / 2)
        return "HOLD", 0.0

    @staticmethod
    def _adf_test(s: np.ndarray, ml=12) -> Dict[str, float]:
        s = np.asarray(s, dtype=float).ravel()
        n = len(s)
        if n < 15:
            return {"adf_statistic": 0.0, "p_value": 1.0}
        y = s[1:]
        t_vec = np.arange(1, len(y) + 1)
        X = np.column_stack([np.ones(len(y)), t_vec])
        coefs = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = np.asarray(y - X @ coefs, dtype=float).ravel()
        g0 = float(np.sum(residuals ** 2) / n)
        gamma_arr = np.array([
            np.sum(residuals[i:] * residuals[:len(residuals) - i]) / n
            for i in range(1, min(len(residuals) // 3, ml) + 1)
        ])
        adf_stat = float(coefs[1])
        if adf_stat < -3.96:
            pv = 0.001
        elif adf_stat < -3.41:
            pv = 0.05
        elif adf_stat < -3.12:
            pv = 0.10
        else:
            pv = 0.20
        return {"adf_statistic": adf_stat, "p_value": pv}

    @staticmethod
    def cointegration_test(s1: np.ndarray, s2: np.ndarray) -> Dict[str, Any]:
        s1_c = np.ascontiguousarray(s1).ravel()
        s2_c = np.ascontiguousarray(s2).ravel()
        X = np.column_stack([np.ones(len(s1_c)), s2_c])
        beta = np.linalg.lstsq(X, s1_c, rcond=None)[0]
        res = np.ascontiguousarray(s1_c - (beta[0] + beta[1] * s2_c)).ravel()
        adf = StatisticalModels._adf_test(res)
        h = StatisticalModels.hurst_exponent(res)
        return {
            "beta": float(beta[1]), "intercept": float(beta[0]),
            "spread_mean": float(np.mean(res)), "spread_std": float(np.std(res)),
            "adf_statistic": float(adf["adf_statistic"]),
            "p_value": float(adf["p_value"]),
            "is_cointegrated": bool(adf["p_value"] < 0.05),
            "hurst": h
        }

    @staticmethod
    def granger_causality(s1: np.ndarray, s2: np.ndarray, p=5) -> Dict[str, Any]:
        s1_c = np.ascontiguousarray(s1).ravel()
        s2_c = np.ascontiguousarray(s2).ravel()
        if len(s1_c) < p + 10:
            return {"f_statistic": 0.0, "p_value": 1.0, "granger_causes": False}
        y = s1_c[p:]
        X_r = np.column_stack([s1_c[p - i:len(s1_c) - i] for i in range(1, p + 1)])
        rss_r = np.sum((y - np.linalg.lstsq(X_r, y, rcond=None)[0]) ** 2)
        X_ur = np.column_stack([
            *[s1_c[p - i:len(s1_c) - i] for i in range(1, p + 1)],
            *[s2_c[p - i:len(s2_c) - i] for i in range(1, p + 1)]
        ])
        if X_ur.shape[0] < X_ur.shape[1]:
            return {"f_statistic": 0.0, "p_value": 1.0, "granger_causes": False}
        rss_ur = np.sum((y - np.linalg.lstsq(X_ur, y, rcond=None)[0]) ** 2)
        if rss_ur >= rss_r:
            return {"f_statistic": 0.0, "p_value": 1.0, "granger_causes": False}
        n_obs, k = len(y), p
        f_stat = ((rss_r - rss_ur) / k) / (rss_ur / (n_obs - 2 * k + 1))
        p_value = float(1.0 - stats.f.cdf(f_stat, k, n_obs - 2 * k + 1))
        return {"f_statistic": float(f_stat), "p_value": p_value, "granger_causes": bool(p_value < 0.05)}

    @staticmethod
    def kalman_spread(s1: np.ndarray, s2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s1 = np.ascontiguousarray(s1).ravel()
        s2 = np.ascontiguousarray(s2).ravel()
        n = len(s1)
        x = np.zeros((2, n))
        P = np.zeros((2, 2, n))
        x[0, 0], x[1, 0] = 1.0, 0.0
        P[:, :, 0] = np.eye(2)
        for t in range(1, n):
            x[:, t] = x[:, t - 1]
            P[:, :, t] = P[:, :, t - 1] + 0.001 * np.eye(2)
            z_vec = np.array([s2[t], 1.0])
            S = float(z_vec @ P[:, :, t] @ z_vec + 0.01)
            K = (P[:, :, t] @ z_vec) / S
            innovation = s1[t] - float(z_vec @ x[:, t])
            x[:, t] = x[:, t] + K * innovation
            P[:, :, t] = (np.eye(2) - np.outer(K, z_vec)) @ P[:, :, t]
        spread = s1 - (x[0, :] * s2 + x[1, :])
        return x[0, :], np.ascontiguousarray(spread).ravel()

    @staticmethod
    def garch_volatility(r: np.ndarray) -> Tuple[float, float]:
        r = np.asarray(r, dtype=float).ravel()
        n = len(r)
        if n < 30:
            return float(np.std(r) * np.sqrt(252)), float(np.std(r))
        sq = r ** 2
        var = np.zeros(n)
        var[0] = float(np.var(r))
        for t in range(1, n):
            var[t] = 0.01 + 0.1 * sq[t - 1] + 0.85 * var[t - 1]
        return float(np.sqrt(var[-1]) * np.sqrt(252)), float(np.sqrt(var[-1]))

    @staticmethod
    def serial_correlation(r: np.ndarray, ml=5) -> Dict[str, Any]:
        r = np.ascontiguousarray(r).ravel()
        acfs = []
        for lag in range(1, ml + 1):
            if len(r) - lag < 5:
                break
            valid = r[:-lag]
            valid2 = r[lag:]
            if len(valid) > 2:
                try:
                    r_val = float(stats.pearsonr(valid, valid2)[0])
                except Exception:
                    r_val = 0.0
                if not np.isnan(r_val):
                    acfs.append(r_val)
        n_obs = len(acfs)
        if n_obs == 0:
            return {"acf": [], "ljung_box_stat": 0.0, "p_value": 1.0, "has_autocorrelation": False}
        lb_stat = float(n_obs * (n_obs + 1) * sum(
            acfs[i] ** 2 / (n_obs - i - 1) if (n_obs - i - 1) > 0 else 0
            for i in range(n_obs)
        ))
        p_value = float(1.0 - stats.chi2.cdf(lb_stat, df=n_obs))
        return {"acf": acfs, "ljung_box_stat": lb_stat, "p_value": p_value, "has_autocorrelation": bool(p_value < 0.05)}


class GameTheoryAnalyzer:

    def analyze_market_game(self, df, orderbook=None):
        def safe_arr(val):
            if hasattr(val, "__iter__") and not isinstance(val, str):
                a = np.ascontiguousarray(val, dtype=float).ravel()
                return a
            return np.array([float(val)], dtype=float)
        c = safe_arr(df["close"])
        v = safe_arr(df["volume"]) if "volume" in df.columns else np.ones_like(c)
        h = safe_arr(df["high"]) if hasattr(df["high"], "values") else safe_arr(df.get("high", c * 1.01))
        lo = safe_arr(df["low"]) if hasattr(df["low"], "values") else safe_arr(df.get("low", c * 0.99))
        n = len(c)
        result = {}

        # Whale detection
        vol_mean = float(np.mean(v[-20:])) if n >= 20 else float(np.mean(v))
        vol_std = float(np.std(v[-20:])) if n >= 20 else 1.0
        z_vol = float((v[-1] - vol_mean) / (vol_std + 1e-10)) if vol_std > 0 else 0.0
        pc = float(abs(c[-1] - c[-2]) / (c[-2] + 1e-10)) if n >= 2 else 0.0
        is_whale = bool(z_vol > 2.0 and pc > 0.01)
        direction = "BUY" if c[-1] > c[-2] else ("SELL" if c[-1] < c[-2] else "NEUTRAL")
        whale_score = 0.0
        if is_whale:
            whale_score += 0.5
        whale_days = 0
        if vol_std > 0:
            for x in v[-5:]:
                if (x - vol_mean) / (vol_std + 1e-10) > 2.0:
                    whale_days += 1
        if whale_days >= 3:
            whale_score += 0.3
        if direction == "BUY":
            whale_score += 0.2
        elif direction == "SELL":
            whale_score -= 0.2
        result["whale_detection"] = {"is_whale": is_whale, "z_score": z_vol, "direction": direction, "score": float(max(-1, min(1, whale_score)))}

        # Market maker pressure
        spreads = h - lo
        avg_spread = float(np.mean(spreads[-20:]) / (c[-1] + 1e-10)) if n >= 20 else 0.01
        price_range = h - lo
        lower_shadow_arr = np.asarray((h + lo) / 2 - lo).ravel()
        upper_shadow_arr = np.asarray(h - (h + lo) / 2).ravel()
        pr_arr = np.asarray(price_range).ravel()
        if n >= 10:
            lsr = float(np.mean(lower_shadow_arr[-10:]) / (float(np.mean(pr_arr[-10:])) + 1e-10))
            usr = float(np.mean(upper_shadow_arr[-10:]) / (float(np.mean(pr_arr[-10:])) + 1e-10))
        else:
            lsr, usr = 0.5, 0.5
        vr = float(v[-1] / (vol_mean + 1e-10))
        pressure = 0.0
        if avg_spread > 0.02:
            pressure -= 0.3
        elif avg_spread < 0.005:
            pressure += 0.2
        if lsr > 0.4:
            pressure += 0.4
        elif usr > 0.4:
            pressure -= 0.4
        if vr > 1.5:
            pressure += 0.2 if c[-1] > c[-2] else -0.2
        result["market_maker_pressure"] = {"spread_pct": avg_spread, "pressure": float(pressure), "score": float(max(-1, min(1, pressure))), "reasoning": []}

        # Liquidity game
        if orderbook and "bids" in orderbook and "asks" in orderbook:
            bids, asks = orderbook["bids"], orderbook["asks"]
            if bids and asks:
                spread = float((asks[0][0] - bids[0][0]) / ((asks[0][0] + bids[0][0]) / 2 + 1e-10))
                bv = sum(b[1] for b in bids[:5])
                av = sum(a[1] for a in asks[:5])
                imbalance = float((bv - av) / (bv + av + 1e-10))
                lscore = 0.0
                if spread > 0.01:
                    lscore -= 0.3
                if imbalance > 0.3:
                    lscore += 0.4
                elif imbalance < -0.3:
                    lscore -= 0.4
                gt = "BID_SIDE" if imbalance > 0.2 else ("ASK_SIDE" if imbalance < -0.2 else "BALANCED")
                result["liquidity_game"] = {"spread": spread, "imbalance": imbalance, "score": float(max(-1, min(1, lscore))), "game_type": gt}
            else:
                result["liquidity_game"] = {"spread": 0.0, "imbalance": 0.0, "score": 0.0, "game_type": "UNKNOWN"}
        else:
            result["liquidity_game"] = {"spread": 0.0, "imbalance": 0.0, "score": 0.0, "game_type": "NO_DATA"}

        # Price manipulation
        manip_flags = []
        manip_prob = 0.0
        if n >= 5:
            pchg = float(abs(c[-1] - c[-5]) / (c[-5] + 1e-10))
            vr10 = float(v[-1] / (np.mean(v[-10:]) + 1e-10)) if n >= 10 else 1.0
            if pchg > 0.05 and vr10 < 0.5:
                manip_flags.append("Volume-price divergence")
                manip_prob += 0.25
        if n >= 20:
            vr_recent = float(np.mean(v[-5:]) / (np.mean(v[-20:-5]) + 1e-10))
            pchg5 = float(abs(c[-1] - c[-5]) / (c[-5] + 1e-10)) if n >= 5 else 0.0
            if vr_recent > 3.0 and pchg5 < 0.02:
                manip_flags.append("Wash trading suspected")
                manip_prob += 0.30
        returns = np.diff(c) / (c[:-1] + 1e-10) if n >= 2 else np.array([0.0])
        if len(returns) >= 20:
            ret_std = float(np.std(returns[-20:]))
            ret_recent = float(abs(returns[-1]))
            if ret_recent > 3 * ret_std:
                manip_flags.append("Extreme price move")
                manip_prob += 0.20
        result["manipulation_prob"] = {"flags": manip_flags, "prob": float(min(1.0, manip_prob)), "is_manipulated": bool(manip_prob > 0.4)}

        # Nash equilibrium
        short_var = float(np.var(returns[-5:])) if len(returns) >= 5 else 0.0
        long_var = float(np.var(returns[-20:])) if len(returns) >= 20 else 0.01
        vr_ratio = short_var / (long_var + 1e-10)
        nash_state = "TRANSITION"
        nash_score = 0.0
        if vr_ratio < 0.5:
            nash_state = "QUASI_EQUILIBRIUM"
            nash_score = 0.3
        elif vr_ratio > 2.0:
            nash_state = "DISEQUILIBRIUM"
            nash_score = -0.5
        result["nash_equilibrium"] = {"state": nash_state, "score": nash_score, "variance_ratio": float(vr_ratio)}

        # Crowd behavior
        if len(returns) >= 10:
            sk = float(stats.skew(returns[-10:]))
        else:
            sk = 0.0
        pchg5d = float((c[-1] - c[-5]) / (c[-5] + 1e-10)) if n >= 5 else 0.0
        vr5 = float(v[-1] / (np.mean(v[-10:-1]) + 1e-10)) if n >= 10 else 1.0
        fomo = bool(pchg5d > 0.02 and vr5 > 1.5)
        std_recent = float(np.std(returns[-5:])) if len(returns) >= 5 else 0.0
        std_older = float(np.std(returns[-20:-5])) if len(returns) >= 20 else 1.0
        panic = bool(sk < -1.0 and std_recent > 2 * std_older)
        price_dirs = np.sign(np.diff(c)) if n >= 2 else np.array([0.0])
        herding = float(abs(sum(price_dirs[-10:]))) / 10.0 if len(price_dirs) >= 10 else 0.0
        cscore = 0.0
        if fomo:
            cscore -= 0.3
        if panic:
            cscore -= 0.4
        if herding > 0.6:
            cscore -= herding * 0.2
        if sk > 1.0:
            cscore += 0.2
        behavior = "FOMO" if fomo else ("PANIC" if panic else ("HERDING" if herding > 0.6 else "RATIONAL"))
        result["crowd_behavior"] = {"behavior": behavior, "fomo": float(fomo), "panic": float(panic), "herding": float(herding), "score": float(max(-1, min(1, cscore)))}

        # Information asymmetry
        lead_corr = 0.0
        if len(returns) >= 10:
            for lag in range(1, 5):
                if len(returns) > lag:
                    try:
                        c_val = float(abs(stats.pearsonr(returns[:-lag], np.diff(v)[lag:])[0]))
                        if not np.isnan(c_val):
                            lead_corr = max(lead_corr, c_val)
                    except Exception:
                        pass
        extreme_days = 0
        if len(returns) >= 20:
            mret = float(np.mean(returns[-20:]))
            sret = float(np.std(returns[-20:]))
            if sret > 0:
                for r in returns[-20:]:
                    if abs(r - mret) > 2.5 * sret:
                        extreme_days += 1
        insider_prob = 0.0
        if extreme_days >= 3:
            insider_prob += 0.3
        if lead_corr > 0.5:
            insider_prob += 0.3
        if std_recent > 2 * std_older:
            insider_prob += 0.2
        result["information_asymmetry"] = {"insider_probability": float(insider_prob), "has_informed_trading": bool(insider_prob > 0.5), "score": float(1.0 - insider_prob)}

        # Overall game theory score
        gt_score = (
            result["whale_detection"]["score"] * 0.15 +
            result["market_maker_pressure"]["score"] * 0.20 +
            result["liquidity_game"]["score"] * 0.15 +
            (1 - result["manipulation_prob"]["prob"]) * 0.20 +
            result["nash_equilibrium"]["score"] * 0.15 +
            result["crowd_behavior"]["score"] * 0.15
        )
        result["overall_game_score"] = float(max(-1, min(1, gt_score)))
        return result
