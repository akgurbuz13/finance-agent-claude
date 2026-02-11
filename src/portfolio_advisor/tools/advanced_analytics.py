"""Advanced analytics — PCA, clustering, style analysis, attribution, entropy, mutual info."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from agents import RunContextWrapper, function_tool

from portfolio_advisor.agents.context import AppContext
from portfolio_advisor.tools.technical_indicators import _prices_to_series

logger = logging.getLogger(__name__)


# ── Pure computation functions ────────────────────────────────────────────────


def compute_pca_raw(returns_matrix: np.ndarray, tickers: list[str], n_components: int = 5) -> dict:
    """PCA on standardized returns: eigenvalues, variance explained, loadings.

    returns_matrix: (T, N) array of aligned returns for N assets.
    """
    from sklearn.decomposition import PCA

    n_obs, n_assets = returns_matrix.shape
    if n_obs < 30:
        return {"error": "Insufficient data (need 30+ observations)"}

    n_comp = min(n_components, n_assets, n_obs)

    # Standardize
    means = returns_matrix.mean(axis=0)
    stds = returns_matrix.std(axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero
    standardized = (returns_matrix - means) / stds

    pca = PCA(n_components=n_comp)
    pca.fit(standardized)

    # Component loadings (correlation of original variables with PCs)
    loadings = {}
    for i in range(n_comp):
        component_loadings = {}
        for j, ticker in enumerate(tickers):
            component_loadings[ticker] = round(float(pca.components_[i, j]), 4)
        loadings[f"PC{i+1}"] = component_loadings

    # Identify what each PC represents
    pc_labels = []
    for i in range(n_comp):
        abs_loadings = np.abs(pca.components_[i])
        top_idx = np.argsort(abs_loadings)[-3:][::-1]
        top_tickers = [tickers[j] for j in top_idx]
        # Check if market factor (all same sign)
        signs = np.sign(pca.components_[i])
        if np.all(signs == signs[0]) or np.sum(signs == signs[0]) > 0.8 * n_assets:
            label = "market_factor"
        else:
            label = f"rotation_{'+'.join(top_tickers[:2])}"
        pc_labels.append(label)

    return {
        "n_components": n_comp,
        "eigenvalues": [round(float(v), 6) for v in pca.explained_variance_],
        "variance_explained_pct": [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_],
        "cumulative_variance_pct": [
            round(float(v) * 100, 2)
            for v in np.cumsum(pca.explained_variance_ratio_)
        ],
        "loadings": loadings,
        "pc_labels": pc_labels,
        "observations": n_obs,
    }


def compute_clustering_raw(
    returns_matrix: np.ndarray,
    tickers: list[str],
) -> dict:
    """Hierarchical clustering on correlation distance matrix.

    Distance = sqrt(2 * (1 - rho)). Ward linkage.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n_assets = returns_matrix.shape[1]
    if n_assets < 3:
        return {"error": "Need at least 3 assets for clustering"}

    # Correlation matrix
    corr = np.corrcoef(returns_matrix.T)

    # Correlation distance
    dist = np.sqrt(2.0 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)

    # Convert to condensed form
    condensed = squareform(dist, checks=False)

    # Hierarchical clustering (Ward linkage)
    Z = linkage(condensed, method="ward")

    # Cut at different levels to find optimal clusters
    best_n = 2
    best_score = -1.0
    for n_clusters in range(2, min(n_assets, 6)):
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        # Simple silhouette-like score: avg intra-cluster corr vs inter-cluster
        intra = []
        inter = []
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                if labels[i] == labels[j]:
                    intra.append(corr[i, j])
                else:
                    inter.append(corr[i, j])
        if intra and inter:
            score = np.mean(intra) - np.mean(inter)
            if score > best_score:
                best_score = score
                best_n = n_clusters

    # Final clustering
    labels = fcluster(Z, t=best_n, criterion="maxclust")

    clusters = {}
    for i, ticker in enumerate(tickers):
        c = int(labels[i])
        if c not in clusters:
            clusters[c] = []
        clusters[c].append(ticker)

    # Cluster statistics
    cluster_stats = {}
    for c, members in clusters.items():
        member_indices = [tickers.index(t) for t in members]
        if len(member_indices) > 1:
            sub_corr = corr[np.ix_(member_indices, member_indices)]
            avg_corr = float(np.mean(sub_corr[np.triu_indices(len(member_indices), k=1)]))
        else:
            avg_corr = 1.0
        cluster_stats[f"cluster_{c}"] = {
            "members": members,
            "size": len(members),
            "avg_intra_correlation": round(avg_corr, 3),
        }

    return {
        "n_clusters": best_n,
        "clusters": cluster_stats,
        "cluster_quality_score": round(float(best_score), 3) if best_score > -1 else 0.0,
        "correlation_matrix": {
            tickers[i]: {tickers[j]: round(float(corr[i, j]), 3) for j in range(n_assets)}
            for i in range(n_assets)
        },
        "observations": returns_matrix.shape[0],
    }


def compute_style_analysis_raw(
    asset_returns: np.ndarray,
    factor_returns: np.ndarray,
    factor_names: list[str],
) -> dict:
    """Sharpe (1992) Returns-Based Style Analysis.

    Constrained regression: r_i = sum(w_j * r_factor_j) + epsilon
    Subject to: sum(w_j) = 1, w_j >= 0
    """
    from scipy.optimize import minimize

    n = len(asset_returns)
    n_factors = factor_returns.shape[1]

    if n < 60:
        return {"error": "Insufficient data (need 60+ observations)"}

    # Full-sample RBSA
    def _rbsa(y, X):
        def obj(w):
            residual = y - X @ w
            return np.sum(residual**2)

        w0 = np.ones(n_factors) / n_factors
        bounds = [(0, 1) for _ in range(n_factors)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        result = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        return result.x if result.success else w0

    weights = _rbsa(asset_returns, factor_returns)

    # R-squared
    y_pred = factor_returns @ weights
    ss_res = np.sum((asset_returns - y_pred) ** 2)
    ss_tot = np.sum((asset_returns - asset_returns.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Rolling RBSA (63-day window)
    rolling_weights = []
    window = 63
    if n >= window + 30:
        for start in range(0, n - window, 21):  # step every 21 days
            end = start + window
            w = _rbsa(asset_returns[start:end], factor_returns[start:end])
            rolling_weights.append({
                "period_end_idx": end,
                "weights": {name: round(float(w[i]), 3) for i, name in enumerate(factor_names)},
            })

    # Dominant style
    style_weights = {name: round(float(weights[i]), 3) for i, name in enumerate(factor_names)}
    dominant = max(style_weights, key=style_weights.get)

    return {
        "style_weights": style_weights,
        "dominant_style": dominant,
        "r_squared": round(float(r_squared), 4),
        "selection_return_pct": round(float(asset_returns.mean() - y_pred.mean()) * 252 * 100, 2),
        "rolling_style": rolling_weights[-5:] if rolling_weights else [],
        "interpretation": (
            f"Dominant exposure: {dominant} ({style_weights[dominant]*100:.0f}%). "
            f"Style R²={r_squared:.2f} — "
            f"{'well explained' if r_squared > 0.8 else 'partially explained' if r_squared > 0.5 else 'low explanatory power'}."
        ),
        "observations": n,
    }


def compute_brinson_raw(
    portfolio_weights: dict[str, float],
    benchmark_weights: dict[str, float],
    portfolio_returns: dict[str, float],
    benchmark_returns: dict[str, float],
    sector_map: dict[str, str],
) -> dict:
    """Brinson-Fachler performance attribution.

    Total = Allocation + Selection + Interaction
    Allocation: (w_p - w_b) * (r_b_sector - r_b_total)
    Selection: w_b * (r_p_sector - r_b_sector)
    Interaction: (w_p - w_b) * (r_p_sector - r_b_sector)
    """
    # Group by sector
    sectors = set(sector_map.values())

    # Compute sector-level weights and returns
    p_sector_w = {}
    b_sector_w = {}
    p_sector_r = {}
    b_sector_r = {}

    for sector in sectors:
        tickers_in_sector = [t for t, s in sector_map.items() if s == sector]

        pw = sum(portfolio_weights.get(t, 0) for t in tickers_in_sector)
        bw = sum(benchmark_weights.get(t, 0) for t in tickers_in_sector)

        # Weighted return within sector
        pr = sum(
            portfolio_weights.get(t, 0) * portfolio_returns.get(t, 0) for t in tickers_in_sector
        )
        pr = pr / pw if pw > 0 else 0

        br = sum(
            benchmark_weights.get(t, 0) * benchmark_returns.get(t, 0) for t in tickers_in_sector
        )
        br = br / bw if bw > 0 else 0

        p_sector_w[sector] = pw
        b_sector_w[sector] = bw
        p_sector_r[sector] = pr
        b_sector_r[sector] = br

    # Total benchmark return
    total_b_return = sum(bw * br for bw, br in zip(b_sector_w.values(), b_sector_r.values()))

    # Attribution per sector
    attribution = {}
    total_allocation = 0.0
    total_selection = 0.0
    total_interaction = 0.0

    for sector in sectors:
        pw = p_sector_w.get(sector, 0)
        bw = b_sector_w.get(sector, 0)
        pr = p_sector_r.get(sector, 0)
        br = b_sector_r.get(sector, 0)

        alloc = (pw - bw) * (br - total_b_return)
        select = bw * (pr - br)
        interact = (pw - bw) * (pr - br)

        total_allocation += alloc
        total_selection += select
        total_interaction += interact

        attribution[sector] = {
            "portfolio_weight_pct": round(pw * 100, 2),
            "benchmark_weight_pct": round(bw * 100, 2),
            "portfolio_return_pct": round(pr * 100, 2),
            "benchmark_return_pct": round(br * 100, 2),
            "allocation_pct": round(alloc * 100, 4),
            "selection_pct": round(select * 100, 4),
            "interaction_pct": round(interact * 100, 4),
        }

    total_active = total_allocation + total_selection + total_interaction

    return {
        "total_active_return_pct": round(total_active * 100, 4),
        "allocation_effect_pct": round(total_allocation * 100, 4),
        "selection_effect_pct": round(total_selection * 100, 4),
        "interaction_effect_pct": round(total_interaction * 100, 4),
        "sector_attribution": attribution,
        "interpretation": (
            f"Active return: {total_active*100:+.2f}%. "
            f"Allocation: {total_allocation*100:+.2f}%, "
            f"Selection: {total_selection*100:+.2f}%, "
            f"Interaction: {total_interaction*100:+.2f}%."
        ),
    }


def compute_entropy_raw(weights: np.ndarray) -> dict:
    """Shannon entropy and related diversification measures.

    H = -sum(w_i * ln(w_i)) for w_i > 0.
    Max entropy = ln(N) (equal weight). Normalized = H / ln(N).
    """
    w = weights[weights > 1e-10]  # filter near-zero
    n = len(w)

    if n == 0:
        return {"error": "No non-zero weights"}

    # Shannon entropy
    entropy = float(-np.sum(w * np.log(w)))
    max_entropy = float(np.log(n))
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0

    # Effective number of assets (exponential of entropy)
    eff_n = float(np.exp(entropy))

    # Herfindahl-Hirschman Index (concentration)
    hhi = float(np.sum(w**2))

    return {
        "shannon_entropy": round(entropy, 4),
        "max_entropy": round(max_entropy, 4),
        "normalized_entropy": round(normalized, 4),
        "effective_n_assets": round(eff_n, 2),
        "actual_n_assets": n,
        "herfindahl_index": round(hhi, 4),
        "concentration": "high" if hhi > 0.25 else "moderate" if hhi > 0.15 else "low",
        "interpretation": (
            f"Entropy={entropy:.3f} (normalized={normalized:.2f}). "
            f"Effective {eff_n:.1f} of {n} assets utilized. "
            f"{'Well-diversified' if normalized > 0.8 else 'Moderately diversified' if normalized > 0.6 else 'Concentrated'}."
        ),
    }


def compute_mutual_info_raw(
    returns_matrix: np.ndarray,
    tickers: list[str],
) -> dict:
    """Mutual information between asset pairs via histogram-based estimation.

    MI(X,Y) = H(X) + H(Y) - H(X,Y). Captures non-linear dependencies.
    """
    n_obs, n_assets = returns_matrix.shape
    if n_obs < 60 or n_assets < 2:
        return {"error": "Need 60+ observations and 2+ assets"}

    n_bins = max(10, int(np.sqrt(n_obs / 5)))

    def _entropy_1d(x):
        hist, _ = np.histogram(x, bins=n_bins, density=True)
        hist = hist[hist > 0]
        bin_width = (x.max() - x.min()) / n_bins if x.max() > x.min() else 1.0
        probs = hist * bin_width
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    def _entropy_2d(x, y):
        hist, _, _ = np.histogram2d(x, y, bins=n_bins, density=True)
        hist = hist[hist > 0]
        bw_x = (x.max() - x.min()) / n_bins if x.max() > x.min() else 1.0
        bw_y = (y.max() - y.min()) / n_bins if y.max() > y.min() else 1.0
        probs = hist * bw_x * bw_y
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    # Pairwise MI
    mi_matrix = {}
    notable = []
    for i in range(n_assets):
        mi_row = {}
        for j in range(n_assets):
            if i == j:
                mi_row[tickers[j]] = None
                continue
            if j < i:
                mi_row[tickers[j]] = mi_matrix[tickers[j]][tickers[i]]
                continue

            hx = _entropy_1d(returns_matrix[:, i])
            hy = _entropy_1d(returns_matrix[:, j])
            hxy = _entropy_2d(returns_matrix[:, i], returns_matrix[:, j])
            mi = max(0.0, hx + hy - hxy)

            # Normalized MI
            min_h = min(hx, hy)
            nmi = mi / min_h if min_h > 0 else 0.0

            mi_row[tickers[j]] = round(mi, 4)

            # Compare with linear correlation
            corr = float(np.corrcoef(returns_matrix[:, i], returns_matrix[:, j])[0, 1])

            if nmi > 0.1:
                notable.append({
                    "pair": f"{tickers[i]}/{tickers[j]}",
                    "mutual_information": round(mi, 4),
                    "normalized_mi": round(nmi, 4),
                    "linear_correlation": round(corr, 3),
                    "nonlinear_dependency": nmi > abs(corr) * 0.5,
                })

        mi_matrix[tickers[i]] = mi_row

    # Sort notable by MI
    notable.sort(key=lambda x: x["mutual_information"], reverse=True)

    return {
        "mi_matrix": mi_matrix,
        "notable_pairs": notable[:10],
        "n_bins": n_bins,
        "interpretation": (
            f"{len([p for p in notable if p['nonlinear_dependency']])} pair(s) with "
            f"significant non-linear dependencies (MI > correlation). "
            f"{'Non-linear risk models recommended.' if any(p['nonlinear_dependency'] for p in notable) else 'Linear correlations appear sufficient.'}"
        ),
        "observations": n_obs,
    }


# ── @function_tool wrappers ──────────────────────────────────────────────────


@function_tool
async def compute_pca_returns(
    ctx: RunContextWrapper[AppContext],
    tickers: str,
    prices_json: str,
    n_components: int = 5,
) -> str:
    """PCA on standardized returns. Extracts principal factors, variance explained, and loadings. prices_json: {ticker: [bars]}."""
    ticker_list = [t.strip() for t in tickers.split(",")]
    prices_dict = json.loads(prices_json)

    # Build aligned returns matrix
    all_returns = {}
    for t in ticker_list:
        if t not in prices_dict:
            continue
        df = _prices_to_series(json.dumps(prices_dict[t]))
        all_returns[t] = df["close"].pct_change().dropna()

    if len(all_returns) < 2:
        return json.dumps({"error": "Need data for at least 2 tickers"})

    ret_df = pd.DataFrame(all_returns).dropna()
    valid_tickers = list(ret_df.columns)

    raw = compute_pca_raw(ret_df.values, valid_tickers, n_components)
    if "error" in raw:
        return json.dumps({"error": raw["error"]})
    raw["tickers"] = valid_tickers
    return json.dumps(raw)


@function_tool
async def compute_hierarchical_clustering(
    ctx: RunContextWrapper[AppContext],
    tickers: str,
    prices_json: str,
) -> str:
    """Hierarchical clustering of assets based on correlation distance. Returns cluster assignments and intra-cluster correlations. prices_json: {ticker: [bars]}."""
    ticker_list = [t.strip() for t in tickers.split(",")]
    prices_dict = json.loads(prices_json)

    all_returns = {}
    for t in ticker_list:
        if t not in prices_dict:
            continue
        df = _prices_to_series(json.dumps(prices_dict[t]))
        all_returns[t] = df["close"].pct_change().dropna()

    if len(all_returns) < 3:
        return json.dumps({"error": "Need data for at least 3 tickers"})

    ret_df = pd.DataFrame(all_returns).dropna()
    valid_tickers = list(ret_df.columns)

    raw = compute_clustering_raw(ret_df.values, valid_tickers)
    if "error" in raw:
        return json.dumps({"error": raw["error"]})
    raw["tickers"] = valid_tickers
    return json.dumps(raw)


@function_tool
async def compute_style_analysis(
    ctx: RunContextWrapper[AppContext],
    ticker: str,
    prices_json: str,
    factor_tickers: str = "SPY,IWM,TLT,GLD",
) -> str:
    """Sharpe (1992) Returns-Based Style Analysis. Constrained regression of asset returns on factor returns (weights sum to 1, non-negative). Shows time-varying style exposures."""
    import yfinance as yf

    df = _prices_to_series(prices_json)
    asset_returns = df["close"].pct_change().dropna()

    factor_list = [t.strip() for t in factor_tickers.split(",")]
    factor_df = yf.download(factor_list, period="1y", group_by="ticker", progress=False)
    if factor_df.empty:
        return json.dumps({"ticker": ticker, "error": "Could not fetch factor data"})

    # Build aligned factor returns
    factor_returns = {}
    for ft in factor_list:
        try:
            if len(factor_list) == 1:
                factor_returns[ft] = factor_df["Close"].pct_change().dropna()
            else:
                factor_returns[ft] = factor_df[ft]["Close"].pct_change().dropna()
        except KeyError:
            continue

    if len(factor_returns) < 2:
        return json.dumps({"ticker": ticker, "error": "Insufficient factor data"})

    fr_df = pd.DataFrame(factor_returns).dropna()
    common = asset_returns.index.intersection(fr_df.index)
    if len(common) < 60:
        return json.dumps({"ticker": ticker, "error": "Insufficient overlapping data"})

    valid_factors = list(fr_df.columns)
    raw = compute_style_analysis_raw(
        asset_returns.loc[common].values,
        fr_df.loc[common].values,
        valid_factors,
    )
    if "error" in raw:
        return json.dumps({"ticker": ticker, "error": raw["error"]})
    raw["ticker"] = ticker
    raw["factors"] = valid_factors
    return json.dumps(raw)


@function_tool
async def compute_brinson_attribution(
    ctx: RunContextWrapper[AppContext],
    portfolio_json: str,
    benchmark_json: str,
    returns_json: str,
    sector_map_json: str,
) -> str:
    """Brinson-Fachler performance attribution. Decomposes active return into allocation, selection, and interaction effects. portfolio_json/benchmark_json: {ticker: weight}, returns_json: {ticker: {portfolio_return, benchmark_return}}, sector_map_json: {ticker: sector}."""
    portfolio_w = json.loads(portfolio_json)
    benchmark_w = json.loads(benchmark_json)
    returns_data = json.loads(returns_json)
    sector_map = json.loads(sector_map_json)

    # Extract returns
    p_returns = {t: r.get("portfolio_return", 0) for t, r in returns_data.items()}
    b_returns = {t: r.get("benchmark_return", 0) for t, r in returns_data.items()}

    raw = compute_brinson_raw(portfolio_w, benchmark_w, p_returns, b_returns, sector_map)
    if "error" in raw:
        return json.dumps({"error": raw["error"]})
    return json.dumps(raw)


@function_tool
async def compute_information_entropy(
    ctx: RunContextWrapper[AppContext],
    portfolio_weights_json: str,
) -> str:
    """Compute Shannon entropy and diversification measures for portfolio weights. Higher entropy = more diversified. Also reports Herfindahl index and effective number of assets."""
    weights_dict = json.loads(portfolio_weights_json)
    w = np.array(list(weights_dict.values())) / 100.0  # Convert from pct to fraction

    raw = compute_entropy_raw(w)
    if "error" in raw:
        return json.dumps({"error": raw["error"]})
    raw["tickers"] = list(weights_dict.keys())
    raw["weights_pct"] = weights_dict
    return json.dumps(raw)


@function_tool
async def compute_mutual_information(
    ctx: RunContextWrapper[AppContext],
    tickers: str,
    prices_json: str,
) -> str:
    """Compute pairwise mutual information between assets. Captures non-linear dependencies beyond correlation. prices_json: {ticker: [bars]}."""
    ticker_list = [t.strip() for t in tickers.split(",")]
    prices_dict = json.loads(prices_json)

    all_returns = {}
    for t in ticker_list:
        if t not in prices_dict:
            continue
        df = _prices_to_series(json.dumps(prices_dict[t]))
        all_returns[t] = df["close"].pct_change().dropna()

    if len(all_returns) < 2:
        return json.dumps({"error": "Need data for at least 2 tickers"})

    ret_df = pd.DataFrame(all_returns).dropna()
    valid_tickers = list(ret_df.columns)

    raw = compute_mutual_info_raw(ret_df.values, valid_tickers)
    if "error" in raw:
        return json.dumps({"error": raw["error"]})
    raw["tickers"] = valid_tickers
    return json.dumps(raw)
