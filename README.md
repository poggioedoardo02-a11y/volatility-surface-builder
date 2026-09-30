# volatility-surface-builder
A straightforward Python script for pricing financial options and visualizing a 3D Implied Volatility Surface using live market data. 

This project is built to be lightweight and does **not** rely on the `scipy` library for its mathematical models.

## Features

- **Black-Scholes-Merton (BSM):** Analytical pricing for European options and Vega calculation.
- **Binomial Tree (CRR):** Pricing for both European and American options (includes early exercise premium).
- **Monte Carlo Simulation:** Option pricing using random price paths.
- **Implied Volatility:** Calculated using the Newton-Raphson root-finding algorithm.
- **3D Volatility Surface:** Automatically fetches live market data via Yahoo Finance to plot a 3D surface of implied volatility based on moneyness and time to maturity.
- **Automated Tests:** Includes built-in tests to verify put-call parity, arbitrage bounds, and model convergence.

## Requirements

You need Python installed on your system along with the following libraries:

- `numpy`
- `pandas`
- `yfinance`
- `matplotlib`

You can install all the required dependencies using pip:

```bash
pip install numpy pandas yfinance matplotlib
