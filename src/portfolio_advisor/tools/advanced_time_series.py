"""Advanced time series tools — Granger causality, change points, spectral analysis, ARCH test."""

from __future__ import annotations

import json
import logging

import numpy as np
from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.tools.technical_indicators import _prices_to_series

logger = logging.getLogger(__name__)


# ── Pure computation functions ────────────────────────────────────────────────


def compute_granger_raw(
    y_returns: np.ndarray,
    x_returns: np.ndarray,
    max_lag: int = 5,
) -> dict:
    """Granger causality F-test: does lagged X improve prediction of Y?

    Compares restricted model (Y ~ Y_lags) vs unrestricted (Y ~ Y_lags + X_lags).
    Returns F-statistic, p-value, and optimal lag.
    """
    from scipy.stats import f as f_dist

    n = len(y_returns)
    if n < max_lag * 3 + 10:
        return {"error": f"Insufficient data (need {max_lag * 3 + 10}+ obs, got {n})"}

    best_result = None
    best_aic = np.inf

    for lag in range(1, max_lag + 1):
        nobs = n - lag
        if nobs < 20:
            continue

        Y = y_returns[lag:]

        # Restricted model: Y_t ~ c + Y_{t-1} + ... + Y_{t-lag}
        X_r = np.column_stack([np.ones(nobs)] + [y_returns[lag - i - 1 : n - i - 1] for i in range(lag)])

        # Unrestricted model: Y_t ~ c + Y_{t-1} + ... + Y_{t-lag} + X_{t-1} + ... + X_{t-lag}
        X_u = np.column_stack(
            [X_r] + [x_returns[lag - i - 1 : n - i - 1] for i in range(lag)]
        )

        try:
            # Fit restricted
            coeffs_r = np.linalg.lstsq(X_r, Y, rcond=None)[0]
            resid_r = Y - X_r @ coeffs_r
            sse_r = np.sum(resid_r**2)

            # Fit unrestricted
            coeffs_u = np.linalg.lstsq(X_u, Y, rcond=None)[0]
            resid_u = Y - X_u @ coeffs_u
            sse_u = np.sum(resid_u**2)

            # F-test: ((SSE_r - SSE_u) / q) / (SSE_u / (n - k))
            q = lag  # number of restrictions
            k = X_u.shape[1]  # parameters in unrestricted model
            dof = nobs - k

            if dof <= 0 or sse_u <= 0:
                continue

            f_stat = ((sse_r - sse_u) / q) / (sse_u / dof)
            p_value = 1 - f_dist.cdf(f_stat, q, dof)

            # AIC for model selection
            aic = nobs * np.log(sse_u / nobs) + 2 * k

            if aic < best_aic:
                best_aic = aic
                best_result = {
                    "lag": lag,
                    "f_statistic": round(float(f_stat), 4),
                    "p_value": round(float(p_value), 6),
                    "sse_restricted": round(float(sse_r), 8),
                    "sse_unrestricted": round(float(sse_u), 8),
                    "dof": (q, dof),
                }
        except np.linalg.LinAlgError:
            continue

    if best_result is None:
        return {"error": "All lag specifications failed"}

    is_significant = best_result["p_value"] < 0.05
    best_result["is_granger_causal"] = is_significant
    best_result["optimal_lag"] = best_result["lag"]

    return best_result


def detect_change_points_raw(
    returns: np.ndarray,
    method: str = "cusum",
) -> dict:
    """Detect structural breaks using CUSUM or binary segmentation.

    CUSUM: cumulative sum of deviations from mean. Change at max|S_k|.
    Binary segmentation: recursive splitting at maximum variance reduction.
    """
    n = len(returns)
    if n < 30:
        return {"error": "Insufficient data (need 30+ observations)"}

    change_points = []

    if method == "cusum":
        # CUSUM statistic
        mean_r = np.mean(returns)
        cumsum = np.cumsum(returns - mean_r)

        # Normalize
        std_r = np.std(returns)
        if std_r > 0:
            cumsum_normalized = cumsum / (std_r * np.sqrt(n))
        else:
            return {"error": "Zero variance in returns"}

        # Find significant change points (where CUSUM exceeds critical value)
        # Sample-size dependent critical values for standardized CUSUM at 5%
        if n < 50:
            critical = 1.63
        elif n < 100:
            critical = 1.48
        else:
            critical = 1.36
        abs_cumsum = np.abs(cumsum_normalized)

        # Find peaks above critical value
        for i in range(1, n - 1):
            if (
                abs_cumsum[i] > critical
                and abs_cumsum[i] >= abs_cumsum[i - 1]
                and abs_cumsum[i] >= abs_cumsum[i + 1]
            ):
                # Verify it's a local maximum above threshold
                change_points.append({
                    "index": int(i),
                    "cusum_value": round(float(cumsum_normalized[i]), 4),
                    "confidence": round(min(0.99, float(abs_cumsum[i]) / critical * 0.5 + 0.5), 3),
                })

        # Always include the global maximum
        global_max_idx = int(np.argmax(abs_cumsum))
        if abs_cumsum[global_max_idx] > critical * 0.8:
            # Check if already included
            if not any(cp["index"] == global_max_idx for cp in change_points):
                change_points.insert(0, {
                    "index": global_max_idx,
                    "cusum_value": round(float(cumsum_normalized[global_max_idx]), 4),
                    "confidence": round(
                        min(0.99, float(abs_cumsum[global_max_idx]) / critical * 0.5 + 0.5), 3
                    ),
                })

    elif method == "binary_segmentation":
        # Recursive binary segmentation
        def _best_split(data: np.ndarray, start: int, min_size: int = 15) -> tuple | None:
            n_seg = len(data)
            if n_seg < 2 * min_size:
                return None
            best_gain = 0.0
            best_idx = None
            total_var = np.var(data) * n_seg
            for k in range(min_size, n_seg - min_size):
                left_var = np.var(data[:k]) * k
                right_var = np.var(data[k:]) * (n_seg - k)
                gain = total_var - (left_var + right_var)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = k
            if best_idx is not None and best_gain > total_var * 0.05:
                return (start + best_idx, best_gain / total_var)
            return None

        # Find up to 5 change points
        segments = [(0, n)]
        for _ in range(5):
            best_split = None
            best_segment_idx = -1
            best_var_reduction = 0.0
            for seg_idx, (s, e) in enumerate(segments):
                result = _best_split(returns[s:e], s)
                if result and result[1] > best_var_reduction:
                    best_split = result
                    best_segment_idx = seg_idx
                    best_var_reduction = result[1]
            if best_split is None:
                break
            cp_idx, var_red = best_split
            change_points.append({
                "index": int(cp_idx),
                "variance_reduction": round(float(var_red), 4),
                "confidence": round(min(0.99, float(var_red) * 2 + 0.5), 3),
            })
            # Split the segment
            s, e = segments[best_segment_idx]
            segments[best_segment_idx] = (s, cp_idx)
            segments.insert(best_segment_idx + 1, (cp_idx, e))

    # Sort by index
    change_points.sort(key=lambda x: x["index"])

    # Segment statistics
    segments_stats = []
    all_cps = [0] + [cp["index"] for cp in change_points] + [n]
    for i in range(len(all_cps) - 1):
        seg = returns[all_cps[i] : all_cps[i + 1]]
        if len(seg) > 0:
            segments_stats.append({
                "start_idx": all_cps[i],
                "end_idx": all_cps[i + 1],
                "length": len(seg),
                "mean_return_pct": round(float(np.mean(seg)) * 100, 4),
                "vol_pct": round(float(np.std(seg)) * np.sqrt(252) * 100, 2),
            })

    return {
        "method": method,
        "n_change_points": len(change_points),
        "change_points": change_points[:10],
        "segments": segments_stats,
        "observations": n,
    }


def compute_spectral_raw(returns: np.ndarray) -> dict:
    """FFT-based spectral analysis to identify dominant cyclical patterns.

    Returns periodogram with dominant frequencies and their corresponding periods.
    """
    n = len(returns)
    if n < 60:
        return {"error": "Insufficient data (need 60+ observations)"}

    # Detrend (remove mean)
    centered = returns - np.mean(returns)

    # Apply Hann window to reduce spectral leakage
    window = np.hanning(n)
    windowed = centered * window

    # FFT
    fft_vals = np.fft.rfft(windowed)
    power = np.abs(fft_vals) ** 2 / n
    freqs = np.fft.rfftfreq(n, d=1.0)  # d=1 trading day

    # Skip DC component (freq=0)
    power = power[1:]
    freqs = freqs[1:]

    # Find dominant frequencies (top 5)
    top_indices = np.argsort(power)[-5:][::-1]
    dominant = []
    for idx in top_indices:
        freq = float(freqs[idx])
        period = 1.0 / freq if freq > 0 else float("inf")
        dominant.append({
            "frequency": round(freq, 6),
            "period_days": round(period, 1),
            "power": round(float(power[idx]), 8),
            "interpretation": (
                "weekly" if 4 <= period <= 6 else
                "bi-weekly" if 9 <= period <= 11 else
                "monthly" if 18 <= period <= 25 else
                "quarterly" if 55 <= period <= 70 else
                f"~{period:.0f} day cycle"
            ),
        })

    # Total power and concentration
    total_power = float(np.sum(power))
    top_power = sum(d["power"] for d in dominant[:3])
    concentration = top_power / total_power if total_power > 0 else 0.0

    has_cycles = concentration > 0.3

    return {
        "dominant_frequencies": dominant,
        "total_spectral_power": round(total_power, 8),
        "top3_concentration": round(concentration, 3),
        "has_significant_cycles": has_cycles,
        "interpretation": (
            f"{'Strong' if concentration > 0.5 else 'Moderate' if concentration > 0.3 else 'Weak'} "
            f"cyclical patterns. Dominant period: {dominant[0]['period_days']:.0f} days "
            f"({dominant[0]['interpretation']})."
        ) if dominant else "No dominant cycles detected.",
        "observations": n,
    }


def test_arch_effects_raw(returns: np.ndarray, max_lag: int = 5) -> dict:
    """Engle's ARCH LM test: test for autoregressive conditional heteroskedasticity.

    Regress squared residuals on lagged squared residuals.
    LM stat = T * R^2 ~ chi-squared(q).
    Pre-condition for GARCH modeling.
    """
    from scipy.stats import chi2

    n = len(returns)
    if n < max_lag + 30:
        return {"error": f"Insufficient data (need {max_lag + 30}+ obs)"}

    # Demean returns
    resid = returns - np.mean(returns)
    resid_sq = resid**2

    best_result = None
    for lag in range(1, max_lag + 1):
        nobs = n - lag
        Y = resid_sq[lag:]
        X = np.column_stack(
            [np.ones(nobs)] + [resid_sq[lag - i - 1 : n - i - 1] for i in range(lag)]
        )

        try:
            coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
            y_pred = X @ coeffs
            ss_res = np.sum((Y - y_pred) ** 2)
            ss_tot = np.sum((Y - np.mean(Y)) ** 2)
            r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            lm_stat = nobs * r_sq
            p_value = 1 - chi2.cdf(lm_stat, lag)

            if best_result is None or p_value < best_result["p_value"]:
                best_result = {
                    "lag": lag,
                    "lm_statistic": round(float(lm_stat), 4),
                    "p_value": round(float(p_value), 6),
                    "r_squared": round(float(r_sq), 4),
                }
        except np.linalg.LinAlgError:
            continue

    if best_result is None:
        return {"error": "All lag specifications failed"}

    has_arch = best_result["p_value"] < 0.05
    best_result["has_arch_effects"] = has_arch
    best_result["garch_applicable"] = has_arch
    best_result["interpretation"] = (
        f"ARCH effects {'detected' if has_arch else 'not detected'} "
        f"(LM={best_result['lm_statistic']:.2f}, p={best_result['p_value']:.4f}). "
        f"{'GARCH modeling is appropriate.' if has_arch else 'Constant volatility may suffice.'}"
    )

    return best_result


# ── @function_tool wrappers ──────────────────────────────────────────────────


@function_tool
async def compute_granger_causality(
    ctx: RunContextWrapper[AppContext],
    ticker1: str,
    ticker2: str,
    prices_json: str,
    max_lag: int = 5,
) -> str:
    """Granger causality test between two assets. Tests whether lagged returns of one asset help predict the other. prices_json: {ticker: [bars]}."""
    prices_dict = json.loads(prices_json)

    df1 = _prices_to_series(json.dumps(prices_dict.get(ticker1, [])))
    df2 = _prices_to_series(json.dumps(prices_dict.get(ticker2, [])))

    ret1 = df1["close"].pct_change().dropna()
    ret2 = df2["close"].pct_change().dropna()

    common = ret1.index.intersection(ret2.index)
    if len(common) < max_lag * 3 + 10:
        return json.dumps({"error": "Insufficient overlapping data"})

    r1 = ret1.loc[common].values
    r2 = ret2.loc[common].values

    # Test both directions
    result_1to2 = compute_granger_raw(r2, r1, max_lag)
    result_2to1 = compute_granger_raw(r1, r2, max_lag)

    return json.dumps({
        "test_1": {
            "cause": ticker1,
            "effect": ticker2,
            **result_1to2,
        },
        "test_2": {
            "cause": ticker2,
            "effect": ticker1,
            **result_2to1,
        },
        "interpretation": (
            f"{ticker1} -> {ticker2}: {'significant' if result_1to2.get('is_granger_causal') else 'not significant'}. "
            f"{ticker2} -> {ticker1}: {'significant' if result_2to1.get('is_granger_causal') else 'not significant'}."
        ),
        "observations": len(common),
    })


@function_tool
async def detect_change_points(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
    method: str = "cusum",
) -> str:
    """Detect structural breaks in return series using CUSUM or binary segmentation. Identifies regime changes and structural shifts."""
    df = _prices_to_series(prices_json)
    returns = df["close"].pct_change().dropna().values

    raw = detect_change_points_raw(returns, method)
    if "error" in raw:
        return json.dumps({"ticker": ticker, "error": raw["error"]})

    # Add dates if available
    dates = df.index[1:]  # skip first NaN
    for cp in raw.get("change_points", []):
        idx = cp["index"]
        if idx < len(dates):
            cp["date"] = dates[idx].strftime("%Y-%m-%d")
    for seg in raw.get("segments", []):
        si = seg["start_idx"]
        ei = min(seg["end_idx"] - 1, len(dates) - 1)
        if si < len(dates):
            seg["start_date"] = dates[si].strftime("%Y-%m-%d")
        if ei < len(dates):
            seg["end_date"] = dates[ei].strftime("%Y-%m-%d")

    raw["ticker"] = ticker
    return json.dumps(raw)


@function_tool
async def compute_spectral_analysis(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
) -> str:
    """FFT-based spectral analysis to identify dominant cyclical patterns in returns. Reports dominant frequencies and their corresponding periods (in trading days)."""
    df = _prices_to_series(prices_json)
    returns = df["close"].pct_change().dropna().values

    raw = compute_spectral_raw(returns)
    if "error" in raw:
        return json.dumps({"ticker": ticker, "error": raw["error"]})
    raw["ticker"] = ticker
    return json.dumps(raw)


@function_tool
async def test_arch_effects(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
    max_lag: int = 5,
) -> str:
    """Engle's ARCH LM test for volatility clustering. Pre-condition check for GARCH modeling. Tests whether squared returns exhibit autocorrelation."""
    df = _prices_to_series(prices_json)
    returns = df["close"].pct_change().dropna().values

    raw = test_arch_effects_raw(returns, max_lag)
    if "error" in raw:
        return json.dumps({"ticker": ticker, "error": raw["error"]})
    raw["ticker"] = ticker
    return json.dumps(raw)
