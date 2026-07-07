
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# ============================================================
# CORE PRICING FUNCTIONS
# ============================================================

def calculate_d1_d2(S, K, T, r, sigma, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_price(S, K, T, r, sigma, q, option_type):
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma, q)

    if option_type == "Call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def black_scholes_greeks(S, K, T, r, sigma, q, option_type):
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)

    gamma = (np.exp(-q * T) * pdf_d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)
    vega_per_1pct = vega / 100

    if option_type == "Call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (
            - (S * np.exp(-q * T) * pdf_d1 * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        )
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)

    else:
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
        theta = (
            - (S * np.exp(-q * T) * pdf_d1 * sigma) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        )
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Vega per 1% vol move": vega_per_1pct,
        "Theta annual": theta,
        "Theta daily": theta / 365,
        "Rho": rho,
        "Rho per 1% rate move": rho / 100
    }


def intrinsic_value(S, K, option_type):
    if option_type == "Call":
        return max(S - K, 0)
    return max(K - S, 0)


def payoff_values(ST_range, K, option_type):
    if option_type == "Call":
        return np.maximum(ST_range - K, 0)
    return np.maximum(K - ST_range, 0)


def implied_volatility_from_price(market_price, S, K, T, r, q, option_type):
    def objective(vol):
        return black_scholes_price(S, K, T, r, vol, q, option_type) - market_price

    try:
        return brentq(objective, 0.0001, 5.0)
    except ValueError:
        return None


def binomial_tree_price(S, K, T, r, sigma, q, option_type, steps, exercise_style):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    disc = np.exp(-r * dt)

    p = (np.exp((r - q) * dt) - d) / (u - d)

    if p < 0 or p > 1:
        return None, p

    j = np.arange(steps + 1)
    ST = S * (u ** j) * (d ** (steps - j))

    if option_type == "Call":
        option_values = np.maximum(ST - K, 0)
    else:
        option_values = np.maximum(K - ST, 0)

    for i in range(steps - 1, -1, -1):
        option_values = disc * (p * option_values[1:i + 2] + (1 - p) * option_values[0:i + 1])

        if exercise_style == "American":
            j = np.arange(i + 1)
            S_nodes = S * (u ** j) * (d ** (i - j))

            if option_type == "Call":
                exercise_values = np.maximum(S_nodes - K, 0)
            else:
                exercise_values = np.maximum(K - S_nodes, 0)

            option_values = np.maximum(option_values, exercise_values)

    return option_values[0], p


def monte_carlo_price(S, K, T, r, sigma, q, option_type, simulations, seed):
    rng = np.random.default_rng(seed)

    Z = rng.standard_normal(simulations)
    ST = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

    if option_type == "Call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)

    discounted_payoff = np.exp(-r * T) * payoff
    price = np.mean(discounted_payoff)
    standard_error = np.std(discounted_payoff) / np.sqrt(simulations)

    return price, standard_error, ST, payoff


def heston_monte_carlo_price(S, K, T, r, q, option_type, v0, kappa, theta, xi, rho, steps, simulations, seed):
    rng = np.random.default_rng(seed)
    dt = T / steps

    S_paths = np.full(simulations, S, dtype=float)
    v_paths = np.full(simulations, v0, dtype=float)

    for _ in range(steps):
        z1 = rng.standard_normal(simulations)
        z_independent = rng.standard_normal(simulations)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * z_independent

        v_positive = np.maximum(v_paths, 0)

        S_paths = S_paths * np.exp(
            (r - q - 0.5 * v_positive) * dt + np.sqrt(v_positive * dt) * z1
        )

        v_paths = v_paths + kappa * (theta - v_positive) * dt + xi * np.sqrt(v_positive * dt) * z2
        v_paths = np.maximum(v_paths, 0)

    if option_type == "Call":
        payoff = np.maximum(S_paths - K, 0)
    else:
        payoff = np.maximum(K - S_paths, 0)

    discounted_payoff = np.exp(-r * T) * payoff
    price = np.mean(discounted_payoff)
    standard_error = np.std(discounted_payoff) / np.sqrt(simulations)

    return price, standard_error, S_paths, payoff


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_payoff_chart(S, K, option_type, option_price):
    ST_range = np.linspace(max(0.01, S * 0.4), S * 1.6, 300)
    payoff = payoff_values(ST_range, K, option_type)
    profit_loss = payoff - option_price

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ST_range, payoff, label="Payoff at Maturity")
    ax.plot(ST_range, profit_loss, label="Profit / Loss after Premium")
    ax.axhline(0, linewidth=1)
    ax.axvline(K, linestyle="--", linewidth=1, label="Strike")
    ax.set_xlabel("Underlying Price at Maturity")
    ax.set_ylabel("Value")
    ax.set_title(f"{option_type} Payoff and Profit/Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_price_vs_underlying(S, K, T, r, sigma, q, option_type):
    S_range = np.linspace(max(0.01, S * 0.5), S * 1.5, 200)
    prices = [black_scholes_price(s, K, T, r, sigma, q, option_type) for s in S_range]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(S_range, prices)
    ax.axvline(S, linestyle="--", linewidth=1, label="Current Underlying")
    ax.axvline(K, linestyle=":", linewidth=1, label="Strike")
    ax.set_xlabel("Underlying Price")
    ax.set_ylabel("Option Price")
    ax.set_title(f"{option_type} Price vs Underlying Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_greeks_vs_underlying(S, K, T, r, sigma, q, option_type):
    S_range = np.linspace(max(0.01, S * 0.5), S * 1.5, 200)

    delta_values = []
    gamma_values = []
    vega_values = []
    theta_values = []

    for s in S_range:
        greeks = black_scholes_greeks(s, K, T, r, sigma, q, option_type)
        delta_values.append(greeks["Delta"])
        gamma_values.append(greeks["Gamma"])
        vega_values.append(greeks["Vega per 1% vol move"])
        theta_values.append(greeks["Theta daily"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(S_range, delta_values, label="Delta")
    ax.plot(S_range, gamma_values, label="Gamma")
    ax.plot(S_range, vega_values, label="Vega per 1% vol")
    ax.plot(S_range, theta_values, label="Theta daily")
    ax.axvline(S, linestyle="--", linewidth=1, label="Current Underlying")
    ax.axvline(K, linestyle=":", linewidth=1, label="Strike")
    ax.set_xlabel("Underlying Price")
    ax.set_ylabel("Greek Value")
    ax.set_title("Greeks vs Underlying Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_price_heatmap(S, K, T, r, q, option_type):
    S_range = np.linspace(max(0.01, S * 0.5), S * 1.5, 40)
    vol_range = np.linspace(0.05, 0.80, 40)

    price_matrix = np.zeros((len(vol_range), len(S_range)))

    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(S_range):
            price_matrix[i, j] = black_scholes_price(spot, K, T, r, vol, q, option_type)

    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(
        price_matrix,
        aspect="auto",
        origin="lower",
        extent=[S_range.min(), S_range.max(), vol_range.min() * 100, vol_range.max() * 100]
    )

    ax.set_xlabel("Underlying Price")
    ax.set_ylabel("Volatility (%)")
    ax.set_title(f"{option_type} Price Heatmap: Underlying vs Volatility")
    fig.colorbar(image, ax=ax, label="Option Price")
    return fig


def plot_terminal_distribution(ST, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(ST, bins=50)
    ax.set_xlabel("Terminal Underlying Price")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Options Pricing & Risk Analytics Toolkit",
    page_icon="📈",
    layout="wide"
)

st.title("Options Pricing & Risk Analytics Toolkit")
st.subheader("Pricing, Greeks, Payoff Analysis, Scenario Analysis and Model Extensions")
st.caption("Created by Antonello Losurdo · 2026")

st.markdown("""
This Streamlit dashboard is a Python-based project designed to connect option pricing theory with practical
applications in derivatives pricing, volatility analysis and risk management.

The operational part implements Black-Scholes, Binomial Tree, Monte Carlo Simulation and a simplified Heston Monte Carlo model.
The advanced part introduces Local Volatility, Stochastic Volatility, SABR and Hull-White as model extensions.
""")

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("Navigation")

dashboard_area = st.sidebar.radio(
    "Dashboard Area",
    [
        "Project Overview",
        "Pricing Models",
        "Advanced Models",
        "Risk Analytics Notes"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Created by Antonello Losurdo · 2026")


# ============================================================
# PROJECT OVERVIEW
# ============================================================

if dashboard_area == "Project Overview":

    st.write("## Project Overview")

    st.markdown("""
    The project is structured as a technical dashboard for option pricing and risk analytics.

    The objective is to show clear understanding of:

    - option pricing
    - payoff logic
    - Greeks and sensitivities
    - volatility and interest rate models
    - Monte Carlo simulation
    - scenario analysis
    - risk interpretation
    """)

    roadmap = pd.DataFrame({
        "Area": [
            "Black-Scholes",
            "Greeks",
            "Payoff Analysis",
            "Binomial Tree",
            "Monte Carlo Simulation",
            "Heston Model",
            "Advanced Models",
            "Risk Analytics"
        ],
        "Description": [
            "Closed-form pricing for European vanilla options",
            "Delta, Gamma, Vega, Theta and Rho",
            "Payoff and profit/loss visualization",
            "Discrete-time pricing for European and American options",
            "Simulation-based option pricing",
            "Simplified stochastic volatility simulation",
            "Conceptual explanation of Local Volatility, SABR and Hull-White",
            "Scenario analysis, heatmaps and sensitivity interpretation"
        ],
        "Status": [
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented",
            "Simplified implementation",
            "Conceptual",
            "Implemented"
        ]
    })

    st.dataframe(roadmap, use_container_width=True)

    st.write("## Implemented vs Conceptual Models")

    scope = pd.DataFrame({
        "Model": [
            "Black-Scholes",
            "Binomial Tree",
            "Monte Carlo Simulation",
            "Heston Model",
            "Stochastic Volatility",
            "Local Volatility",
            "SABR",
            "Hull-White"
        ],
        "Status in this project": [
            "Implemented",
            "Implemented",
            "Implemented",
            "Simplified Monte Carlo implementation",
            "Conceptual framework",
            "Conceptual extension",
            "Conceptual extension",
            "Conceptual extension"
        ],
        "Main Use": [
            "European vanilla options",
            "European and American options",
            "European options and simulation logic",
            "Equity/FX stochastic volatility",
            "General class of models with random volatility",
            "Volatility surface and exotic options",
            "Rates volatility smile, cap/floor, swaptions",
            "Short-rate modelling and rates derivatives"
        ]
    })

    st.dataframe(scope, use_container_width=True)


# ============================================================
# PRICING MODELS
# ============================================================

elif dashboard_area == "Pricing Models":

    st.sidebar.header("Market Inputs")

    S = st.sidebar.number_input("Underlying Price", min_value=0.01, value=100.0, step=1.0)
    K = st.sidebar.number_input("Strike Price", min_value=0.01, value=100.0, step=1.0)
    T = st.sidebar.number_input("Time to Maturity (years)", min_value=0.01, value=1.0, step=0.25)
    r = st.sidebar.number_input("Risk-free Rate", value=0.03, step=0.005, format="%.4f")
    sigma = st.sidebar.number_input("Volatility", min_value=0.01, value=0.20, step=0.01, format="%.4f")
    q = st.sidebar.number_input("Dividend Yield", value=0.00, step=0.005, format="%.4f")

    option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])

    pricing_model = st.sidebar.selectbox(
        "Pricing Model",
        [
            "Black-Scholes",
            "Binomial Tree",
            "Monte Carlo Simulation",
            "Heston Model - Simplified Monte Carlo"
        ]
    )

    exercise_style = "European"
    steps = 200
    simulations = 20000
    seed = 42

    if pricing_model == "Binomial Tree":
        exercise_style = st.sidebar.selectbox("Exercise Style", ["European", "American"])
        steps = st.sidebar.slider("Number of Tree Steps", min_value=10, max_value=1000, value=300, step=10)

    elif pricing_model == "Monte Carlo Simulation":
        simulations = st.sidebar.slider("Number of Simulations", min_value=1000, max_value=100000, value=20000, step=1000)
        seed = st.sidebar.number_input("Random Seed", min_value=1, value=42, step=1)

    elif pricing_model == "Heston Model - Simplified Monte Carlo":
        st.sidebar.header("Heston Parameters")

        v0 = st.sidebar.number_input("Initial Variance v0", min_value=0.0001, value=0.04, step=0.005, format="%.4f")
        kappa = st.sidebar.number_input("Mean Reversion Speed kappa", min_value=0.01, value=2.0, step=0.10)
        theta = st.sidebar.number_input("Long-run Variance theta", min_value=0.0001, value=0.04, step=0.005, format="%.4f")
        xi = st.sidebar.number_input("Volatility of Variance xi", min_value=0.01, value=0.50, step=0.05)
        rho_heston = st.sidebar.slider("Correlation rho", min_value=-0.95, max_value=0.95, value=-0.50, step=0.05)
        heston_steps = st.sidebar.slider("Heston Simulation Steps", min_value=20, max_value=504, value=126, step=10)
        heston_sims = st.sidebar.slider("Heston Simulations", min_value=1000, max_value=50000, value=10000, step=1000)
        seed = st.sidebar.number_input("Random Seed", min_value=1, value=42, step=1)

    model_price = None
    model_error = None
    terminal_prices = None
    extra_details = {}

    bs_price = black_scholes_price(S, K, T, r, sigma, q, option_type)

    if pricing_model == "Black-Scholes":
        model_price = bs_price
        extra_details["Model"] = "Closed-form Black-Scholes"

    elif pricing_model == "Binomial Tree":
        model_price, risk_neutral_probability = binomial_tree_price(
            S, K, T, r, sigma, q, option_type, steps, exercise_style
        )
        extra_details["Risk-neutral probability"] = risk_neutral_probability
        extra_details["Steps"] = steps

    elif pricing_model == "Monte Carlo Simulation":
        model_price, model_error, terminal_prices, payoff = monte_carlo_price(
            S, K, T, r, sigma, q, option_type, simulations, int(seed)
        )
        extra_details["Simulations"] = simulations
        extra_details["Standard error"] = model_error

    elif pricing_model == "Heston Model - Simplified Monte Carlo":
        model_price, model_error, terminal_prices, payoff = heston_monte_carlo_price(
            S, K, T, r, q, option_type, v0, kappa, theta, xi, rho_heston, heston_steps, heston_sims, int(seed)
        )
        extra_details["Simulations"] = heston_sims
        extra_details["Steps"] = heston_steps
        extra_details["Standard error"] = model_error
        extra_details["Feller condition"] = "Satisfied" if 2 * kappa * theta >= xi ** 2 else "Not satisfied"

    tab_summary, tab_payoff, tab_greeks, tab_scenarios, tab_distribution, tab_notes = st.tabs(
        [
            "Pricing Summary",
            "Payoff Analysis",
            "Greeks",
            "Scenario Analysis",
            "Simulation Distribution",
            "Model Notes"
        ]
    )

    with tab_summary:

        st.write("## Pricing Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Selected Model", pricing_model)

        with col2:
            st.metric("Option Type", option_type)

        with col3:
            st.metric("Exercise Style", exercise_style)

        with col4:
            if model_price is not None:
                st.metric("Option Price", f"{model_price:.4f}")
            else:
                st.metric("Option Price", "N/A")

        if model_price is None:
            st.error("The selected model cannot price the option with the current parameters.")
        else:
            selected_intrinsic = intrinsic_value(S, K, option_type)
            selected_time_value = model_price - selected_intrinsic

            col5, col6, col7 = st.columns(3)

            with col5:
                st.metric("Intrinsic Value", f"{selected_intrinsic:.4f}")

            with col6:
                st.metric("Time Value", f"{selected_time_value:.4f}")

            with col7:
                st.metric("Black-Scholes Reference", f"{bs_price:.4f}")

        selected_inputs = pd.DataFrame({
            "Parameter": [
                "Underlying Price",
                "Strike Price",
                "Time to Maturity",
                "Risk-free Rate",
                "Volatility",
                "Dividend Yield",
                "Option Type",
                "Pricing Model",
                "Exercise Style"
            ],
            "Value": [
                S,
                K,
                T,
                r,
                sigma,
                q,
                option_type,
                pricing_model,
                exercise_style
            ]
        })

        st.write("### Selected Inputs")
        st.dataframe(selected_inputs, use_container_width=True)

        if extra_details:
            details_df = pd.DataFrame({
                "Detail": list(extra_details.keys()),
                "Value": list(extra_details.values())
            })
            st.write("### Model Details")
            st.dataframe(details_df, use_container_width=True)

        if pricing_model == "Black-Scholes":
            call_price = black_scholes_price(S, K, T, r, sigma, q, "Call")
            put_price = black_scholes_price(S, K, T, r, sigma, q, "Put")
            d1, d2 = calculate_d1_d2(S, K, T, r, sigma, q)

            st.write("### Black-Scholes Details")

            details = pd.DataFrame({
                "Metric": [
                    "Call Price",
                    "Put Price",
                    "d1",
                    "d2",
                    "Put-Call Parity Left Side: C + K e^(-rT)",
                    "Put-Call Parity Right Side: P + S e^(-qT)"
                ],
                "Value": [
                    call_price,
                    put_price,
                    d1,
                    d2,
                    call_price + K * np.exp(-r * T),
                    put_price + S * np.exp(-q * T)
                ]
            })

            st.dataframe(details, use_container_width=True)

        st.write("### Optional Implied Volatility Calculator")

        st.markdown("""
        This section reverses Black-Scholes: instead of using volatility to calculate price,
        it uses a market price to infer implied volatility.
        """)

        default_market_price = float(bs_price) if bs_price is not None and bs_price > 0 else 10.0

        market_price = st.number_input(
            "Market Option Price",
            min_value=0.01,
            value=default_market_price,
            step=0.10
        )

        implied_vol = implied_volatility_from_price(market_price, S, K, T, r, q, option_type)

        if implied_vol is None:
            st.warning("No valid implied volatility found for this market price and input set.")
        else:
            st.success(f"Implied Volatility: {implied_vol:.4%}")

    with tab_payoff:

        st.write("## Payoff Analysis")

        if model_price is None:
            st.warning("Payoff chart requires a valid option price.")
        else:
            fig = plot_payoff_chart(S, K, option_type, model_price)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("""
            **Interpretation**

            The payoff chart shows the value of the option at maturity before and after considering the premium paid.
            It highlights upside, downside, break-even and convexity.
            """)

    with tab_greeks:

        st.write("## Greeks")

        st.info("""
        Greeks are calculated using the Black-Scholes framework as a reference.
        For Binomial Tree, Monte Carlo and Heston, Greeks are not directly implemented here.
        They can be estimated with finite differences or model-specific methods.
        """)

        greeks = black_scholes_greeks(S, K, T, r, sigma, q, option_type)

        col_g1, col_g2, col_g3, col_g4, col_g5 = st.columns(5)

        with col_g1:
            st.metric("Delta", f"{greeks['Delta']:.4f}")

        with col_g2:
            st.metric("Gamma", f"{greeks['Gamma']:.4f}")

        with col_g3:
            st.metric("Vega / 1% vol", f"{greeks['Vega per 1% vol move']:.4f}")

        with col_g4:
            st.metric("Theta daily", f"{greeks['Theta daily']:.4f}")

        with col_g5:
            st.metric("Rho / 1% rate", f"{greeks['Rho per 1% rate move']:.4f}")

        greeks_table = pd.DataFrame({
            "Greek": [
                "Delta",
                "Gamma",
                "Vega",
                "Vega per 1% vol move",
                "Theta annual",
                "Theta daily",
                "Rho",
                "Rho per 1% rate move"
            ],
            "Value": [
                greeks["Delta"],
                greeks["Gamma"],
                greeks["Vega"],
                greeks["Vega per 1% vol move"],
                greeks["Theta annual"],
                greeks["Theta daily"],
                greeks["Rho"],
                greeks["Rho per 1% rate move"]
            ],
            "Interpretation": [
                "Sensitivity of option price to the underlying price.",
                "Sensitivity of Delta to the underlying price.",
                "Sensitivity to a 100% volatility move.",
                "Approximate price change for a 1 percentage point volatility move.",
                "Annual time decay.",
                "Approximate daily time decay.",
                "Sensitivity to a 100% interest rate move.",
                "Approximate price change for a 1 percentage point rate move."
            ]
        })

        st.dataframe(greeks_table, use_container_width=True)

    with tab_scenarios:

        st.write("## Scenario Analysis")

        st.info("""
        Scenario analysis is based on Black-Scholes because it is fast and analytically tractable.
        This section visualizes how option value and Greeks change when market inputs move.
        """)

        fig_price = plot_price_vs_underlying(S, K, T, r, sigma, q, option_type)
        st.pyplot(fig_price)
        plt.close(fig_price)

        fig_greeks = plot_greeks_vs_underlying(S, K, T, r, sigma, q, option_type)
        st.pyplot(fig_greeks)
        plt.close(fig_greeks)

        fig_heatmap = plot_price_heatmap(S, K, T, r, q, option_type)
        st.pyplot(fig_heatmap)
        plt.close(fig_heatmap)

    with tab_distribution:

        st.write("## Simulation Distribution")

        if terminal_prices is None:
            st.info("Terminal distribution is available for Monte Carlo and Heston models.")
        else:
            fig_dist = plot_terminal_distribution(
                terminal_prices,
                f"Terminal Price Distribution - {pricing_model}"
            )
            st.pyplot(fig_dist)
            plt.close(fig_dist)

            stats_df = pd.DataFrame({
                "Statistic": [
                    "Mean Terminal Price",
                    "Median Terminal Price",
                    "5th Percentile",
                    "95th Percentile",
                    "Minimum",
                    "Maximum"
                ],
                "Value": [
                    np.mean(terminal_prices),
                    np.median(terminal_prices),
                    np.percentile(terminal_prices, 5),
                    np.percentile(terminal_prices, 95),
                    np.min(terminal_prices),
                    np.max(terminal_prices)
                ]
            })

            st.dataframe(stats_df, use_container_width=True)

    with tab_notes:

        st.write("## Model Notes")

        if pricing_model == "Black-Scholes":
            st.markdown("""
            **Black-Scholes**

            Black-Scholes is a closed-form model for European vanilla options.
            It assumes constant volatility, continuous trading, lognormal underlying dynamics and frictionless markets.

            Its main limitation is that it does not generate volatility smile or skew.
            """)

        elif pricing_model == "Binomial Tree":
            st.markdown("""
            **Binomial Tree**

            The binomial model prices options by moving backwards through a discrete tree of possible underlying prices.
            It can handle both European and American options.

            Its accuracy depends on the number of steps.
            """)

        elif pricing_model == "Monte Carlo Simulation":
            st.markdown("""
            **Monte Carlo Simulation**

            Monte Carlo estimates option value by simulating many possible future outcomes and discounting the average payoff.

            It is useful for understanding simulation-based pricing and uncertainty around terminal outcomes.
            """)

        elif pricing_model == "Heston Model - Simplified Monte Carlo":
            st.markdown("""
            **Heston Model - Simplified Monte Carlo**

            Heston is a stochastic volatility model where variance follows a mean-reverting process.
            It can generate volatility smile and skew through stochastic variance and correlation between asset returns and variance.

            In this dashboard it is implemented in simplified Monte Carlo form without market calibration.
            """)


# ============================================================
# ADVANCED MODELS
# ============================================================

elif dashboard_area == "Advanced Models":

    st.sidebar.header("Advanced Models")

    advanced_model = st.sidebar.selectbox(
        "Advanced Model",
        [
            "Stochastic Volatility Overview",
            "Heston Model",
            "Local Volatility Model",
            "SABR Model",
            "Hull-White Model"
        ]
    )

    st.write(f"## {advanced_model}")

    if advanced_model == "Stochastic Volatility Overview":

        st.markdown("""
        **Stochastic Volatility** is not a single model. It is a family of models in which volatility is itself random.

        This is the key difference from Black-Scholes and Local Volatility:

        - Black-Scholes: volatility is constant.
        - Local Volatility: volatility is deterministic and depends on price and time.
        - Stochastic Volatility: volatility follows its own stochastic process.

        Heston and SABR are two important examples of stochastic volatility models.
        """)

        st.latex(r"dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^{(1)}")
        st.latex(r"dv_t = \text{stochastic process for variance}")

        comparison = pd.DataFrame({
            "Approach": [
                "Black-Scholes",
                "Local Volatility",
                "Stochastic Volatility"
            ],
            "Volatility Assumption": [
                "Constant volatility",
                "Deterministic function of price and time",
                "Random process with its own dynamics"
            ],
            "Strength": [
                "Simple and analytical",
                "Fits today's volatility surface",
                "More realistic volatility dynamics"
            ],
            "Limitation": [
                "No smile or skew",
                "Smile dynamics may be unrealistic",
                "More complex calibration and pricing"
            ]
        })

        st.dataframe(comparison, use_container_width=True)

    elif advanced_model == "Heston Model":

        st.markdown("""
        **Heston** is a stochastic volatility model mainly used for equity and FX derivatives.

        The asset price is driven by a stochastic variance process. The variance is mean-reverting,
        and the correlation between asset returns and variance helps generate volatility skew.
        """)

        st.latex(r"dS_t = (r-q)S_tdt + \sqrt{v_t}S_tdW_t^{(1)}")
        st.latex(r"dv_t = \kappa(\theta-v_t)dt + \xi\sqrt{v_t}dW_t^{(2)}")
        st.latex(r"dW_t^{(1)}dW_t^{(2)} = \rho dt")

        params = pd.DataFrame({
            "Parameter": ["v0", "kappa", "theta", "xi", "rho"],
            "Meaning": [
                "Initial variance",
                "Speed of mean reversion",
                "Long-run variance",
                "Volatility of variance",
                "Correlation between asset and variance shocks"
            ]
        })

        st.dataframe(params, use_container_width=True)

        st.markdown("""
        In this dashboard, Heston is also implemented in simplified Monte Carlo form under Pricing Models.
        It is not calibrated to market volatility surfaces.
        """)

    elif advanced_model == "Local Volatility Model":

        st.markdown("""
        **Local Volatility** assumes that volatility is not constant, but it is a deterministic function of the underlying price and time.

        This allows the model to fit the observed volatility surface of vanilla options.
        """)

        st.latex(r"dS_t = (r-q)S_tdt + \sigma_{loc}(S_t,t)S_tdW_t")

        st.markdown("""
        The model is conceptually linked to Dupire's framework, where local volatility is derived from
        the market surface of European option prices.

        It is useful for exotic options, barrier options and instruments sensitive to volatility skew.

        It is not implemented operationally here because a correct implementation requires a clean market volatility surface and calibration.
        """)

    elif advanced_model == "SABR Model":

        st.markdown("""
        **SABR** is a stochastic volatility model widely used in rates markets,
        especially for caplets, floorlets and swaptions.

        It models a forward rate and its stochastic volatility.
        """)

        st.latex(r"dF_t = \alpha_t F_t^\beta dW_t^{(1)}")
        st.latex(r"d\alpha_t = \nu \alpha_t dW_t^{(2)}")
        st.latex(r"dW_t^{(1)}dW_t^{(2)} = \rho dt")

        params = pd.DataFrame({
            "Parameter": ["alpha", "beta", "rho", "nu"],
            "Meaning": [
                "Volatility level",
                "Forward dynamics",
                "Skew",
                "Volatility of volatility"
            ]
        })

        st.dataframe(params, use_container_width=True)

    elif advanced_model == "Hull-White Model":

        st.markdown("""
        **Hull-White** is a short-rate model used for interest rate derivatives.

        It models the short rate as a mean-reverting process and can be fitted to the initial yield curve.
        """)

        st.latex(r"dr_t = [\theta(t)-ar_t]dt + \sigma dW_t")

        st.markdown("""
        Hull-White is commonly used for caps, floors, swaptions, callable bonds and Bermudan swaptions.

        It is different from equity volatility models because it focuses on the dynamics of interest rates and the yield curve.
        """)


# ============================================================
# RISK ANALYTICS NOTES
# ============================================================

elif dashboard_area == "Risk Analytics Notes":

    st.write("## Risk Analytics Notes")

    st.markdown("""
    This section summarizes how the outputs of the dashboard can be interpreted from a risk perspective.
    """)

    notes = pd.DataFrame({
        "Concept": [
            "Option Price",
            "Intrinsic Value",
            "Time Value",
            "Delta",
            "Gamma",
            "Vega",
            "Theta",
            "Rho",
            "Monte Carlo Distribution",
            "Implied Volatility",
            "Heston Variance Process"
        ],
        "Risk Interpretation": [
            "The theoretical value of the option under the selected model.",
            "The immediate exercise value of the option.",
            "The part of the option value linked to uncertainty, time and volatility.",
            "Directional exposure to the underlying asset.",
            "Convexity and sensitivity of Delta to underlying movements.",
            "Exposure to changes in volatility.",
            "Sensitivity to the passage of time.",
            "Exposure to changes in interest rates.",
            "Range of possible terminal outcomes under simulated dynamics.",
            "Market-implied level of expected volatility.",
            "Simplified representation of stochastic volatility dynamics."
        ]
    })

    st.dataframe(notes, use_container_width=True)

    st.markdown("""
    A complete risk analysis should not stop at the theoretical price.
    It should also consider sensitivities, scenario analysis, model assumptions and the stability of inputs.
    """)

st.markdown("---")
st.caption("Created by Antonello Losurdo · 2026")
