import math
import sys
from datetime import datetime
from typing import Optional, Tuple, Union
from zoneinfo import ZoneInfo
 
import numpy as np
import pandas as pd
 
ArrayLike = Union[float, np.ndarray]
 
# =====================================================================
# MODULE 1: NORMAL DISTRIBUTION
# =====================================================================
 
_erf = np.vectorize(math.erf, otypes=[float])  # built once, not at every call
 
 
def standard_normal_cdf(x: ArrayLike) -> ArrayLike:
    """Standard normal CDF. Returns a float for scalar input, else an array."""
    out = 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / np.sqrt(2.0)))
    return float(out) if out.ndim == 0 else out
 
 
def standard_normal_pdf(x: ArrayLike) -> ArrayLike:
    """Standard normal PDF."""
    x = np.asarray(x, dtype=float)
    out = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
    return float(out) if out.ndim == 0 else out
 
 
# =====================================================================
# MODULE 2: ANALYTICAL BLACK-SCHOLES-MERTON
# =====================================================================
 
def calculate_d1_d2(
    S: ArrayLike, K: ArrayLike, tau: ArrayLike, r: float, sigma: ArrayLike, q: float = 0.0
) -> Tuple[ArrayLike, ArrayLike]:
    """d1 and d2 of the BSM model (vectorised)."""
    tau = np.maximum(tau, 1e-10)
    sigma = np.maximum(sigma, 1e-10)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return d1, d2
 
 
def bsm_price(
    S: ArrayLike, K: ArrayLike, tau: ArrayLike, r: float, sigma: ArrayLike,
    q: float = 0.0, option_type: str = "call",
) -> ArrayLike:
    """European option price, BSM closed form with continuous dividend yield q."""
    opt = option_type.lower()
    if opt not in ("call", "put"):
        raise ValueError("Option type must be 'call' or 'put'")
 
    tau = np.maximum(tau, 1e-10)
    d1, d2 = calculate_d1_d2(S, K, tau, r, sigma, q)
    disc_S = S * np.exp(-q * tau)
    disc_K = K * np.exp(-r * tau)
    if opt == "call":
        return disc_S * standard_normal_cdf(d1) - disc_K * standard_normal_cdf(d2)
    return disc_K * standard_normal_cdf(-d2) - disc_S * standard_normal_cdf(-d1)
 
 
def bsm_vega(
    S: ArrayLike, K: ArrayLike, tau: ArrayLike, r: float, sigma: ArrayLike, q: float = 0.0
) -> ArrayLike:
    """Vega (dPrice/dSigma, per 1.00 of volatility)."""
    tau = np.maximum(tau, 1e-10)
    d1, _ = calculate_d1_d2(S, K, tau, r, sigma, q)
    return S * np.exp(-q * tau) * standard_normal_pdf(d1) * np.sqrt(tau)
 
 
def _bsm_price_mixed(S, K, tau, r, sigma, q, is_call) -> np.ndarray:
    """BSM price where each element can be a call or a put (via put-call parity)."""
    tau = np.maximum(tau, 1e-10)
    call = bsm_price(S, K, tau, r, sigma, q, "call")
    parity = S * np.exp(-q * tau) - K * np.exp(-r * tau)  # C - P
    return np.where(is_call, call, call - parity)
 
 
# =====================================================================
# MODULE 3: NUMERICAL ENGINES
# =====================================================================
 
def binomial_tree_price(
    S: float, K: float, tau: float, r: float, sigma: float, q: float = 0.0,
    option_type: str = "call", exercise_style: str = "european", steps: int = 1000,
) -> float:
    """CRR binomial tree, European or American."""
    opt = option_type.lower()
    style = exercise_style.lower()
    if opt not in ("call", "put"):
        raise ValueError("Option type must be 'call' or 'put'")
    if style not in ("european", "american"):
        raise ValueError("Exercise style must be 'european' or 'american'")
    if steps < 1:
        raise ValueError("steps must be >= 1")
 
    tau = max(tau, 1e-10)
    dt = tau / steps
    vs = sigma * np.sqrt(dt)
    u = np.exp(vs)
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Risk-neutral probability {p:.4f} outside [0, 1]: increase steps or check inputs")
    discount = np.exp(-r * dt)
 
    j = np.arange(steps + 1)
 
    def prices_at(n: int) -> np.ndarray:
        # node with k ups at level n: S * u^k * d^(n-k) = S * exp(vs * (2k - n))
        return S * np.exp(vs * (2 * j[: n + 1] - n))
 
    def payoff(prices: np.ndarray) -> np.ndarray:
        return np.maximum(prices - K, 0.0) if opt == "call" else np.maximum(K - prices, 0.0)
 
    values = payoff(prices_at(steps))
    for n in range(steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        if style == "american":
            values = np.maximum(values, payoff(prices_at(n)))
    return float(values[0])
 
 
def monte_carlo_price(
    S: float, K: float, tau: float, r: float, sigma: float, q: float = 0.0,
    option_type: str = "call", exercise_style: str = "european",
    num_paths: int = 100_000, seed: Optional[int] = None, return_stderr: bool = False,
) -> Union[float, Tuple[float, float]]:
    """European option via Monte Carlo with antithetic variates.
 
    If return_stderr is True returns (price, standard_error).
    """
    opt = option_type.lower()
    if exercise_style.lower() == "american":
        raise NotImplementedError("Early exercise requires Longstaff-Schwartz.")
    if opt not in ("call", "put"):
        raise ValueError("Option type must be 'call' or 'put'")
 
    tau = max(tau, 1e-10)
    rng = np.random.default_rng(seed)
    half = max(num_paths // 2, 1)
    z = rng.standard_normal(half)
 
    drift = (r - q - 0.5 * sigma**2) * tau
    vol = sigma * np.sqrt(tau)
 
    def payoff(zz: np.ndarray) -> np.ndarray:
        s_t = S * np.exp(drift + vol * zz)
        return np.maximum(s_t - K, 0.0) if opt == "call" else np.maximum(K - s_t, 0.0)
 
    # average each antithetic pair: pairs are i.i.d., so the SE is computed on them
    pair = 0.5 * (payoff(z) + payoff(-z)) * np.exp(-r * tau)
    price = float(pair.mean())
    if return_stderr:
        return price, float(pair.std(ddof=1) / np.sqrt(half))
    return price
 
 
# =====================================================================
# MODULE 4: IMPLIED VOLATILITY (VECTORISED, SAFEGUARDED NEWTON)
# =====================================================================
 
def implied_volatility(
    market_price: ArrayLike, S: float, K: ArrayLike, tau: ArrayLike, r: float,
    q: float = 0.0, is_call: Union[bool, np.ndarray] = True,
    tol: float = 1e-8, max_iter: int = 100, sigma_max: float = 5.0,
) -> ArrayLike:
    """Implied volatility for arrays of European options (calls and/or puts).
 
    Newton-Raphson safeguarded by a bisection bracket [1e-6, sigma_max]: whenever
    the Newton step leaves the bracket (tiny vega, bad start) a bisection step
    is used instead. Prices violating no-arbitrage bounds return NaN.
    """
    scalar = all(np.ndim(x) == 0 for x in (market_price, K, tau, is_call))
    mp, K_, tau_, call = np.broadcast_arrays(
        np.atleast_1d(np.asarray(market_price, dtype=float)),
        np.asarray(K, dtype=float),
        np.asarray(tau, dtype=float),
        np.asarray(is_call, dtype=bool),
    )
    tau_ = np.maximum(tau_, 1e-10)
 
    disc_S = S * np.exp(-q * tau_)
    disc_K = K_ * np.exp(-r * tau_)
    intrinsic = np.where(call, np.maximum(disc_S - disc_K, 0.0), np.maximum(disc_K - disc_S, 0.0))
    upper = np.where(call, disc_S, disc_K)
    valid = np.isfinite(mp) & (mp > intrinsic) & (mp < upper)
 
    lo = np.full(mp.shape, 1e-6)
    hi = np.full(mp.shape, sigma_max)
    sigma = np.full(mp.shape, 0.3)
 
    for _ in range(max_iter):
        diff = _bsm_price_mixed(S, K_, tau_, r, sigma, q, call) - mp
        active = valid & (np.abs(diff) >= tol)
        if not active.any():
            break
        lo = np.where(diff < 0, sigma, lo)
        hi = np.where(diff >= 0, sigma, hi)
        vega = bsm_vega(S, K_, tau_, r, sigma, q)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = sigma - diff / vega
        ok = np.isfinite(newton) & (newton > lo) & (newton < hi)
        step = np.where(ok, newton, 0.5 * (lo + hi))
        sigma = np.where(active, step, sigma)
 
    final_diff = _bsm_price_mixed(S, K_, tau_, r, sigma, q, call) - mp
    result = np.where(valid & (np.abs(final_diff) < tol), sigma, np.nan)
    return float(result[0]) if scalar else result
 
 
# =====================================================================
# MODULE 5: DATA FETCHING & VOLATILITY SURFACE
# =====================================================================
 
_ET = ZoneInfo("America/New_York")
 
 
def fetch_and_clean_options_data(
    ticker_symbol: str, spot_price: float, risk_free_rate: float, dividend_yield: float,
    min_days: int = 7, max_days: int = 365, n_expirations: int = 12,
    max_rel_spread: float = 0.25, max_abs_log_moneyness: float = 0.25,
    fallback_to_last: bool = True,
) -> pd.DataFrame:
    """Download OTM calls/puts, keep liquid quotes and return a clean DataFrame.
 
    Price = bid/ask mid. If the market is closed Yahoo often returns bid = ask = 0;
    with fallback_to_last=True those rows use lastPrice (only if volume > 0).
    """
    import yfinance as yf
 
    print(f"Downloading options chain for {ticker_symbol} (Spot: {spot_price:.2f})...")
    ticker = yf.Ticker(ticker_symbol)
    now = datetime.now(_ET)
 
    expiries = []
    for exp in ticker.options:
        expiry_dt = datetime.strptime(exp, "%Y-%m-%d").replace(hour=16, tzinfo=_ET)
        tau = (expiry_dt - now).total_seconds() / (365.0 * 86400.0)
        if min_days / 365.0 <= tau <= max_days / 365.0:
            expiries.append((exp, tau))
    if not expiries:
        return pd.DataFrame()
 
    # sample expirations evenly across the whole horizon
    idx = np.unique(np.linspace(0, len(expiries) - 1, min(n_expirations, len(expiries))).round().astype(int))
 
    frames = []
    for i in idx:
        exp, tau = expiries[i]
        chain = ticker.option_chain(exp)
        fwd = spot_price * np.exp((risk_free_rate - dividend_yield) * tau)
 
        for raw, is_call in ((chain.calls, True), (chain.puts, False)):
            df = raw.copy()
            df = df[df["strike"] >= fwd] if is_call else df[df["strike"] < fwd]  # OTM only
            df["log_moneyness"] = np.log(df["strike"] / fwd)
            df = df[df["log_moneyness"].abs() <= max_abs_log_moneyness]
 
            bid, ask = df["bid"].fillna(0.0), df["ask"].fillna(0.0)
            has_quote = (bid > 0) & (ask > bid)
            mid = (bid + ask) / 2.0
            rel_spread = (ask - bid) / mid.where(mid > 0)
            good_quote = has_quote & (rel_spread <= max_rel_spread)
 
            use_last = pd.Series(False, index=df.index)
            if fallback_to_last:
                use_last = ~has_quote & (df["volume"].fillna(0) > 0) & (df["lastPrice"] > 0)
 
            keep = good_quote | use_last
            df = df[keep].copy()
            df["market_price"] = np.where(good_quote[keep], mid[keep], df["lastPrice"])
            df["price_source"] = np.where(good_quote[keep], "mid", "last")
            df["tau"] = tau
            df["is_call"] = is_call
            frames.append(df[["strike", "tau", "market_price", "is_call", "log_moneyness", "price_source"]])
 
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    n_last = int((out["price_source"] == "last").sum())
    if n_last:
        print(f"Warning: {n_last} of {len(out)} quotes use lastPrice (bid/ask unavailable, market closed?).")
    return out
 
 
def plot_volatility_surface(df: pd.DataFrame, title: str = "Volatility Surface", show: bool = True):
    """3D surface from a DataFrame with log_moneyness, tau, implied_vol."""
    import matplotlib.pyplot as plt
 
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(df["log_moneyness"], df["tau"], df["implied_vol"], cmap="viridis", edgecolor="none")
    ax.set_xlabel("Log-moneyness ln(K/F)")
    ax.set_ylabel("Time to maturity (years)")
    ax.set_zlabel("Implied volatility")
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if show:
        plt.show()
    return fig
 
 
def generate_volatility_surface(
    ticker_symbol: str = "SPY", risk_free_rate: float = 0.04, dividend_yield: float = 0.013,
) -> Optional[pd.DataFrame]:
    """Download the chain, invert BSM to IV and plot the surface."""
    import yfinance as yf
 
    hist = yf.Ticker(ticker_symbol).history(period="5d")
    if hist.empty:
        print(f"Error fetching spot price for {ticker_symbol}: yfinance unavailable?")
        return None
    spot = float(hist["Close"].iloc[-1])
 
    df = fetch_and_clean_options_data(ticker_symbol, spot, risk_free_rate, dividend_yield)
    if df.empty:
        print("No valid option data retrieved.")
        return None
 
    print("Solving for implied volatility...")
    df["implied_vol"] = implied_volatility(
        df["market_price"].to_numpy(), spot, df["strike"].to_numpy(), df["tau"].to_numpy(),
        risk_free_rate, dividend_yield, df["is_call"].to_numpy(),
    )
    n0 = len(df)
    df = df.dropna(subset=["implied_vol"])
    df = df[(df["implied_vol"] > 0.02) & (df["implied_vol"] < 1.5)]  # drop solver/quote outliers
    print(f"{len(df)} of {n0} contracts kept after IV solving and outlier filtering.")
    if len(df) < 4 or df["tau"].nunique() < 2:
        print("Not enough points to build a surface.")
        return df
 
    plot_volatility_surface(df, f"Implied Volatility Surface - {ticker_symbol}")
    return df
 
 
# =====================================================================
# MODULE 6: TESTS
# =====================================================================
 
_P = dict(S=100.0, K=100.0, tau=1.0, r=0.05, q=0.02, sigma=0.2)
 
 
def test_put_call_parity() -> None:
    c = bsm_price(_P["S"], _P["K"], _P["tau"], _P["r"], _P["sigma"], _P["q"], "call")
    p = bsm_price(_P["S"], _P["K"], _P["tau"], _P["r"], _P["sigma"], _P["q"], "put")
    rhs = _P["S"] * np.exp(-_P["q"] * _P["tau"]) - _P["K"] * np.exp(-_P["r"] * _P["tau"])
    assert np.isclose(c - p, rhs, atol=1e-10), f"{c - p} != {rhs}"
 
 
def test_arbitrage_bounds() -> None:
    S, K, tau, r, q = _P["S"], _P["K"], _P["tau"], _P["r"], _P["q"]
    for sigma in (0.05, 0.2, 0.8):
        c = bsm_price(S, K, tau, r, sigma, q, "call")
        p = bsm_price(S, K, tau, r, sigma, q, "put")
        assert max(S * np.exp(-q * tau) - K * np.exp(-r * tau), 0.0) <= c <= S * np.exp(-q * tau)
        assert max(K * np.exp(-r * tau) - S * np.exp(-q * tau), 0.0) <= p <= K * np.exp(-r * tau)
 
 
def test_binomial_convergence_to_bsm() -> None:
    for opt in ("call", "put"):
        bsm = bsm_price(_P["S"], _P["K"], _P["tau"], _P["r"], _P["sigma"], _P["q"], opt)
        tree = binomial_tree_price(_P["S"], _P["K"], _P["tau"], _P["r"], _P["sigma"], _P["q"], opt, "european", 2000)
        assert np.isclose(bsm, tree, atol=0.01), f"{opt}: tree {tree} vs BSM {bsm}"
 
 
def test_american_early_exercise() -> None:
    args = (_P["S"], _P["K"], _P["tau"], _P["r"], _P["sigma"])
    eu_put = binomial_tree_price(*args, 0.0, "put", "european", 500)
    am_put = binomial_tree_price(*args, 0.0, "put", "american", 500)
    assert am_put > eu_put, f"American put {am_put} <= European put {eu_put}"
    # with q = 0 an American call is never exercised early
    eu_call = binomial_tree_price(*args, 0.0, "call", "european", 500)
    am_call = binomial_tree_price(*args, 0.0, "call", "american", 500)
    assert np.isclose(eu_call, am_call, atol=1e-9), f"{am_call} != {eu_call}"
 
 
def test_monte_carlo_convergence() -> None:
    for opt in ("call", "put"):
        bsm = bsm_price(_P["S"], _P["K"], _P["tau"], _P["r"], _P["sigma"], _P["q"], opt)
        mc, se = monte_carlo_price(
            _P["S"], _P["K"], _P["tau"], _P["r"], _P["sigma"], _P["q"], opt,
            num_paths=400_000, seed=42, return_stderr=True,
        )
        assert abs(mc - bsm) < 4 * se, f"{opt}: MC {mc:.4f} vs BSM {bsm:.4f} (SE {se:.4f})"
 
 
def test_implied_vol_round_trip() -> None:
    S, r, q = 100.0, 0.04, 0.015
    K = np.array([70, 85, 95, 100, 105, 120, 130], dtype=float)
    tau = np.array([0.05, 0.25, 0.5, 1.0, 1.0, 0.1, 0.1])
    sigma_true = np.array([0.35, 0.25, 0.22, 0.2, 0.19, 0.3, 0.4])
    is_call = K >= S * np.exp((r - q) * tau)
    prices = _bsm_price_mixed(S, K, tau, r, sigma_true, q, is_call)
    iv = implied_volatility(prices, S, K, tau, r, q, is_call)
    assert np.all(np.isfinite(iv)), f"NaN in IV: {iv}"
    assert np.allclose(iv, sigma_true, atol=1e-4), f"{iv} vs {sigma_true}"
 
 
def test_implied_vol_rejects_arbitrage_prices() -> None:
    S, K, tau, r, q = 100.0, 90.0, 1.0, 0.05, 0.0
    intrinsic = S - K * np.exp(-r * tau)
    assert np.isnan(implied_volatility(intrinsic - 0.5, S, K, tau, r, q, True))  # below lower bound
    assert np.isnan(implied_volatility(S + 1.0, S, K, tau, r, q, True))          # above upper bound
 
 
def run_all_tests() -> bool:
    tests = [
        ("Put-Call Parity (q != 0)", test_put_call_parity),
        ("Arbitrage Bounds", test_arbitrage_bounds),
        ("Binomial Convergence to BSM", test_binomial_convergence_to_bsm),
        ("American Early Exercise", test_american_early_exercise),
        ("Monte Carlo Convergence", test_monte_carlo_convergence),
        ("Implied Vol Round Trip", test_implied_vol_round_trip),
        ("Implied Vol Rejects Arbitrage Prices", test_implied_vol_rejects_arbitrage_prices),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except AssertionError as err:
            print(f"  [FAIL] {name}: {err}")
        except Exception as err:  # noqa: BLE001
            print(f"  [ERROR] {name}: {err!r}")
    print(f"\nResult: {passed}/{len(tests)} tests passed.")
    return passed == len(tests)
 
 
# =====================================================================
# EXECUTION
# =====================================================================
 
if __name__ == "__main__":
    print("=" * 49)
    print("1. RUNNING FUNCTIONAL ENGINE TESTS")
    print("=" * 49)
    if not run_all_tests():
        sys.exit(1)
 
    print("\n" + "=" * 49)
    print("2. GENERATING VOLATILITY SURFACE")
    print("=" * 49)
    generate_volatility_surface(ticker_symbol="SPY", risk_free_rate=0.04, dividend_yield=0.013)
 
