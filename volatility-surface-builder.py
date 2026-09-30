import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple
from datetime import datetime

# =====================================================================
# MODULE 1: NORMAL DISTRIBUTION FUNCTIONS (STANDALONE / NO SCIPY)
# =====================================================================

def standard_normal_cdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Analytical cumulative distribution function for standard normal distribution."""
    # Use standard Python math.erf instead of np.math.erf (removed in NumPy 2.0)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


def standard_normal_pdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Probability density function for standard normal distribution."""
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)

# =====================================================================
# MODULE 2: ANALYTICAL BLACK-SCHOLES-MERTON ENGINE
# =====================================================================

def calculate_d1_d2(
    S: Union[float, np.ndarray], 
    K: Union[float, np.ndarray], 
    tau: Union[float, np.ndarray], 
    r: float, 
    sigma: Union[float, np.ndarray],
    q: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate d1 and d2 terms for vectorized Black-Scholes-Merton model."""
    tau = np.maximum(tau, 1e-10)
    sigma = np.maximum(sigma, 1e-10)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return d1, d2


def bsm_price(
    S: Union[float, np.ndarray], 
    K: Union[float, np.ndarray], 
    tau: Union[float, np.ndarray], 
    r: float, 
    sigma: Union[float, np.ndarray], 
    q: float = 0.0, 
    option_type: str = 'call'
) -> Union[float, np.ndarray]:
    """Price a European option using analytical BSM closed-form solution."""
    opt_type = option_type.lower()
    if opt_type not in ['call', 'put']:
        raise ValueError("Option type must be 'call' or 'put'")
        
    tau = np.maximum(tau, 1e-10)
    d1, d2 = calculate_d1_d2(S, K, tau, r, sigma, q)
    
    if opt_type == 'call':
        return S * np.exp(-q * tau) * standard_normal_cdf(d1) - K * np.exp(-r * tau) * standard_normal_cdf(d2)
    else:
        return K * np.exp(-r * tau) * standard_normal_cdf(-d2) - S * np.exp(-q * tau) * standard_normal_cdf(-d1)


def bsm_vega(
    S: Union[float, np.ndarray], 
    K: Union[float, np.ndarray], 
    tau: Union[float, np.ndarray], 
    r: float, 
    sigma: Union[float, np.ndarray], 
    q: float = 0.0
) -> Union[float, np.ndarray]:
    """Calculate Vega Greek for a BSM option."""
    tau = np.maximum(tau, 1e-10)
    d1, _ = calculate_d1_d2(S, K, tau, r, sigma, q)
    return S * np.exp(-q * tau) * standard_normal_pdf(d1) * np.sqrt(tau)

# =====================================================================
# MODULE 3: NUMERICAL ENGINES (BINOMIAL TREE & MONTE CARLO)
# =====================================================================

def binomial_tree_price(
    S: float, 
    K: float, 
    tau: float, 
    r: float, 
    sigma: float, 
    q: float = 0.0, 
    option_type: str = 'call', 
    exercise_style: str = 'european', 
    steps: int = 1000
) -> float:
    """Price an option (European or American) using the CRR binomial tree."""
    opt_type = option_type.lower()
    style = exercise_style.lower()
    
    if opt_type not in ['call', 'put']:
        raise ValueError("Option type must be 'call' or 'put'")
    if style not in ['european', 'american']:
        raise ValueError("Exercise style must be 'european' or 'american'")
        
    tau = max(tau, 1e-10)
    dt = tau / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    
    i_array = np.arange(steps + 1)
    asset_prices = S * (u ** i_array) * (d ** (steps - i_array))
    
    if opt_type == 'call':
        option_values = np.maximum(0.0, asset_prices - K)
    else:
        option_values = np.maximum(0.0, K - asset_prices)
        
    for _ in range(steps - 1, -1, -1):
        asset_prices = asset_prices[:-1] / u
        continuation_value = discount * (p * option_values[1:] + (1.0 - p) * option_values[:-1])
        
        if style == 'american':
            intrinsic_value = (np.maximum(0.0, asset_prices - K) if opt_type == 'call' 
                               else np.maximum(0.0, K - asset_prices))
            option_values = np.maximum(continuation_value, intrinsic_value)
        else:
            option_values = continuation_value
            
    return float(option_values[0])


def monte_carlo_price(
    S: float, 
    K: float, 
    tau: float, 
    r: float, 
    sigma: float, 
    q: float = 0.0, 
    option_type: str = 'call', 
    exercise_style: str = 'european', 
    num_paths: int = 100000, 
    seed: Optional[int] = None
) -> float:
    """Price a European option using Monte Carlo simulation with antithetic variates."""
    opt_type = option_type.lower()
    style = exercise_style.lower()
    
    if style == 'american':
        raise NotImplementedError("Standard Monte Carlo does not support early exercise (requires Longstaff-Schwartz).")
    if opt_type not in ['call', 'put']:
        raise ValueError("Option type must be 'call' or 'put'")
        
    tau = max(tau, 1e-10)
    if seed is not None:
        np.random.seed(seed)
        
    half_paths = num_paths // 2
    z = np.random.standard_normal(half_paths)
    z_antithetic = np.concatenate((z, -z))
    
    drift = (r - q - 0.5 * sigma**2) * tau
    diffusion = sigma * np.sqrt(tau) * z_antithetic
    s_t = S * np.exp(drift + diffusion)
    
    if opt_type == 'call':
        payoffs = np.maximum(0.0, s_t - K)
    else:
        payoffs = np.maximum(0.0, K - s_t)
        
    return float(np.exp(-r * tau) * np.mean(payoffs))

# =====================================================================
# MODULE 4: ROOT-FINDING & IMPLIED VOLATILITY
# =====================================================================

def implied_volatility_newton_raphson(
    market_price: float, 
    S: float, 
    K: float, 
    tau: float, 
    r: float, 
    q: float = 0.0, 
    option_type: str = 'call', 
    max_iter: int = 100, 
    tol: float = 1e-5
) -> float:
    """Calculate implied volatility using the Newton-Raphson root-finding algorithm."""
    sigma = 0.5
    tau = max(tau, 1e-10)
    
    for _ in range(max_iter):
        price_guess = bsm_price(S=S, K=K, tau=tau, r=r, sigma=sigma, q=q, option_type=option_type)
        diff = price_guess - market_price
        
        if abs(diff) < tol:
            return float(sigma)
            
        vega = bsm_vega(S=S, K=K, tau=tau, r=r, sigma=sigma, q=q)
        if vega < 1e-8:
            return np.nan
            
        sigma = sigma - diff / vega
        if sigma <= 0.0:
            sigma = 0.001
            
    return np.nan

# =====================================================================
# MODULE 5: DATA FETCHING & 3D VOLATILITY SURFACE
# =====================================================================

def fetch_and_clean_options_data(ticker_symbol: str, spot_price: float) -> pd.DataFrame:
    """Fetch call option chains and filter contracts based on liquidity and moneyness."""
    print(f"Downloading options chain for {ticker_symbol} (Spot: {spot_price:.2f})...")
    ticker = yf.Ticker(ticker_symbol)
    expirations = ticker.options
    options_data = []
    today = datetime.today()

    for exp in expirations[:20]:
        exp_date = datetime.strptime(exp, "%Y-%m-%d")
        tau = (exp_date - today).days / 365.0
        if tau <= 0.02:
            continue 
        
        chain = ticker.option_chain(exp)
        for _, row in chain.calls.iterrows():
            if row['volume'] > 0 and 0.8 < row['strike'] / spot_price < 1.2:
                options_data.append({
                    'strike': row['strike'],
                    'tau': tau,
                    'market_price': row['lastPrice'],
                    'moneyness': row['strike'] / spot_price
                })
                
    return pd.DataFrame(options_data)


def generate_volatility_surface(ticker_symbol: str = "SPY", risk_free_rate: float = 0.04) -> None:
    """Construct and plot the 3D implied volatility surface without SciPy."""
    ticker = yf.Ticker(ticker_symbol)
    spot_price = float(ticker.history(period="1d")['Close'].iloc[-1])
    
    df = fetch_and_clean_options_data(ticker_symbol, spot_price)
    if df.empty:
        print("No valid option data retrieved.")
        return

    print("Solving for implied volatility (Newton-Raphson)...")
    ivs = [
        implied_volatility_newton_raphson(
            market_price=row['market_price'],
            S=spot_price,
            K=row['strike'],
            tau=row['tau'],
            r=risk_free_rate,
            q=0.0,
            option_type='call'
        )
        for _, row in df.iterrows()
    ]
    
    df['implied_vol'] = ivs
    df = df.dropna()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # plot_trisurf handles unstructured (x, y, z) points without griddata interpolation
    surf = ax.plot_trisurf(df['moneyness'], df['tau'], df['implied_vol'], cmap='viridis', edgecolor='none')
    
    ax.set_xlabel('Moneyness (K/S)')
    ax.set_ylabel('Time to Maturity (Years)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(f'Volatility Surface - {ticker_symbol}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

# =====================================================================
# MODULE 6: FUNCTIONAL TESTING
# =====================================================================

def test_put_call_parity() -> None:
    """Verify European Put-Call Parity consistency."""
    S, K, tau, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2
    c = bsm_price(S, K, tau, r, sigma, q=q, option_type='call')
    p = bsm_price(S, K, tau, r, sigma, q=q, option_type='put')
    lhs = c - p
    rhs = S - K * np.exp(-r * tau)
    assert np.isclose(lhs, rhs, atol=1e-5), f"Put-Call parity failed: {lhs} != {rhs}"


def test_arbitrage_bounds() -> None:
    """Check theoretical lower bound for European call options."""
    S, K, tau, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2
    c = bsm_price(S, K, tau, r, sigma, q=q, option_type='call')
    lower_bound = max(0.0, S - K * np.exp(-r * tau))
    assert c >= lower_bound, f"Arbitrage bound violated: {c} < {lower_bound}"


def test_binomial_convergence_to_bsm() -> None:
    """Validate numerical convergence of CRR tree to analytical BSM."""
    S, K, tau, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2
    bsm_val = bsm_price(S, K, tau, r, sigma, q=q, option_type='call')
    tree_val = binomial_tree_price(S, K, tau, r, sigma, q=q, option_type='call', steps=2000)
    assert np.isclose(bsm_val, tree_val, atol=0.02), f"Binomial tree convergence error: {tree_val} vs BSM {bsm_val}"


def test_american_early_exercise_premium() -> None:
    """Confirm American put holds early exercise premium over European put."""
    S, K, tau, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2
    euro_price = binomial_tree_price(S, K, tau, r, sigma, q=q, option_type='put', exercise_style='european', steps=500)
    american_price = binomial_tree_price(S, K, tau, r, sigma, q=q, option_type='put', exercise_style='american', steps=500)
    assert american_price > euro_price, f"Early exercise premium missing: American {american_price} <= European {euro_price}"


def test_monte_carlo_convergence() -> None:
    """Verify Monte Carlo simulation price aligns with analytical BSM price."""
    S, K, tau, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2
    bsm_val = bsm_price(S, K, tau, r, sigma, q=q, option_type='call')
    mc_val = monte_carlo_price(S, K, tau, r, sigma, q=q, option_type='call', num_paths=200000, seed=42)
    assert np.isclose(bsm_val, mc_val, atol=0.05), f"Monte Carlo convergence error: {mc_val} vs BSM {bsm_val}"


def run_all_tests() -> None:
    """Run full suite of mathematical and numerical assertions."""
    tests = [
        ("Put-Call Parity", test_put_call_parity),
        ("Arbitrage Bounds", test_arbitrage_bounds),
        ("Binomial Convergence to BSM", test_binomial_convergence_to_bsm),
        ("American Early Exercise Premium", test_american_early_exercise_premium),
        ("Monte Carlo Convergence", test_monte_carlo_convergence),
    ]
    
    passed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  [PASS] {name}")
            passed += 1
        except AssertionError as err:
            print(f"  [FAIL] {name}: {err}")
        except Exception as err:
            print(f"  [ERROR] {name}: {err}")
            
    print(f"\nResult: {passed}/{len(tests)} tests passed.")

# =====================================================================
# EXECUTION
# =====================================================================

if __name__ == "__main__":
    print("=================================================")
    print("1. RUNNING FUNCTIONAL ENGINE TESTS")
    print("=================================================")
    run_all_tests()
    
    print("\n=================================================")
    print("2. GENERATING VOLATILITY SURFACE (S&P 500 ETF)")
    print("=================================================")
    generate_volatility_surface(ticker_symbol="SPY", risk_free_rate=0.04)
