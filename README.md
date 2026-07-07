# Options Pricing & Risk Analytics Toolkit

Created by Antonello Losurdo · 2026

This project is a Python/Streamlit dashboard designed to connect option pricing theory with practical applications in derivatives pricing, volatility analysis and risk management.

## Main Features

- Black-Scholes pricing for European vanilla options
- Greeks calculation: Delta, Gamma, Vega, Theta and Rho
- Payoff and profit/loss analysis
- Binomial Tree pricing for European and American options
- Monte Carlo simulation for European options
- Simplified Heston Monte Carlo model
- Implied volatility calculator
- Scenario analysis and heatmaps
- Conceptual overview of Stochastic Volatility, Local Volatility, SABR and Hull-White models

## Implemented Models

| Model | Status | Main Use |
|---|---|---|
| Black-Scholes | Implemented | European vanilla options |
| Binomial Tree | Implemented | European and American options |
| Monte Carlo Simulation | Implemented | Simulation-based option pricing |
| Heston Model | Simplified implementation | Stochastic volatility simulation |
| Stochastic Volatility | Conceptual | General volatility modelling framework |
| Local Volatility | Conceptual | Volatility surface modelling |
| SABR | Conceptual | Rates volatility smile |
| Hull-White | Conceptual | Short-rate modelling |

## Project Scope

The dashboard is an educational and analytical project. It is not a professional pricing system and does not include market calibration.

The goal is to demonstrate understanding of option pricing, payoff logic, sensitivities, volatility modelling and risk analysis.

## Files

- `app.py`: main Streamlit application
- `requirements.txt`: Python dependencies
- `README.md`: project description

## How to Run

```bash
streamlit run app.py
