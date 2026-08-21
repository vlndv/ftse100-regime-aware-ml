# Regime-Aware Top-Quartile Machine Learning for FTSE 100 Portfolio Construction

A quantitative equity-selection and portfolio-construction framework that combines cross-sectional machine learning, probability calibration, market-regime modelling and systematic risk management to identify potential outperformers within a selected FTSE 100 stock universe.

The strategy estimates which stocks are most likely to rank among the **top 3 of a 12-stock universe over the next five trading days**, rather than attempting to predict exact future returns.

The resulting signals are converted into a weekly portfolio using confidence filtering, inverse-volatility weighting, turnover control and regime-dependent exposure.

---

## Research Question

> **Can a leakage-controlled machine-learning system identify the future top quartile of FTSE 100 stocks over a five-trading-day horizon, and can causal market-regime probabilities improve the resulting portfolio's risk-adjusted performance after transaction costs?**

---

## Strategy Overview

The complete research pipeline is:

```text
FTSE 100 Market Data
        ↓
Data Cleaning and Alignment
        ↓
Exploratory and Unsupervised Analysis
        ↓
Financial Feature Engineering
        ↓
Cross-Sectional Top-Quartile Target
        ↓
Causal Market-Regime Estimation
        ↓
Chronological Train / Calibration / Validation / Test Split
        ↓
Purged Date-Grouped Cross-Validation
        ↓
Logistic Regression
Random Forest
HistGradientBoosting
        ↓
Probability Calibration
        ↓
Calibrated ML Ensemble
        ↓
Confidence Filtering
        ↓
Top-3 Stock Selection
        ↓
Rank-Buffer Turnover Control
        ↓
Inverse-Volatility Position Sizing
        ↓
Regime-Dependent Exposure
        ↓
Next-Session Execution
        ↓
Five-Trading-Day Holding Period
        ↓
Transaction Costs
        ↓
Portfolio Performance and Robustness Analysis
```

---

## Investment Universe

The analysis uses a diversified universe of **12 FTSE 100 stocks** drawn from multiple industries, including:

- pharmaceuticals;
- banking;
- energy;
- consumer staples;
- retail;
- mining;
- aerospace;
- information services;
- utilities.

The **FTSE 100** is used as the market benchmark.

Daily adjusted market data is obtained using `yfinance`.

---

## Prediction Objective

The strategy is formulated as a **cross-sectional classification problem**.

For every trading date:

1. Calculate each stock's five-trading-day forward log return.
2. Calculate the FTSE 100 five-trading-day forward log return.
3. Calculate each stock's benchmark-relative forward return.
4. Rank all 12 stocks by their future relative performance.
5. Assign `1` to the top 3 stocks.
6. Assign `0` to the remaining 9 stocks.

Therefore:

\[
P(Y=1)=\frac{3}{12}=25\%
\]

The machine-learning problem becomes:

> **What is the probability that each stock will belong to the future top 3?**

This aligns the predictive objective directly with the portfolio-selection decision.

---

## Feature Engineering

The model uses stock-specific, cross-sectional and market-level information.

### Momentum

Multiple momentum horizons capture recent price persistence.

### Short-Term Reversal

Short-horizon returns identify potential reversal behaviour following recent price movements.

### Trend

Moving-average relationships measure whether stocks are trading above or below their recent trends.

### Volatility

Rolling realised volatility measures changing stock-level risk.

### Relative Strength

Stock returns are compared with FTSE 100 performance to measure benchmark-relative momentum.

### RSI

Relative Strength Index provides an additional measure of recent price momentum.

### Beta

Rolling beta estimates each stock's sensitivity to movements in the FTSE 100.

### Drawdown

Rolling drawdown measures the distance between the current price and recent historical highs.

### Volume

Volume-shock features identify abnormal changes in trading activity.

### Market Breadth

Breadth features measure the proportion of stocks displaying positive market behaviour.

### FTSE 100 Market Features

The system also incorporates benchmark momentum, volatility and trend information.

### Cross-Sectional Ranks

Selected features are converted into percentile ranks across the 12-stock universe.

This allows the model to evaluate not only the absolute characteristics of a stock, but also its position relative to the other available investment candidates.

---

## Machine-Learning Models

Three supervised-learning model families are evaluated:

### Logistic Regression

Provides a regularised linear probabilistic classification model.

### Random Forest

Captures nonlinear relationships and interactions through an ensemble of decision trees.

### HistGradientBoosting

Provides an additional nonlinear boosting architecture that sequentially improves classification performance.

Hyperparameters are selected using `RandomizedSearchCV` within a chronological purged cross-validation framework.

---

## Leakage-Controlled Cross-Validation

Financial machine learning requires particular attention to temporal leakage.

The project therefore uses **purged, date-grouped chronological cross-validation**.

All stocks belonging to the same trading date remain within the same fold.

A five-trading-day embargo is also applied between training and validation observations because the classification target itself uses five days of future information.

The structure therefore follows:

```text
Past Data
    ↓
Training
    ↓
5-Day Embargo
    ↓
Validation
```

rather than randomly mixing past and future observations.

---

## Probability Calibration and Ensemble

The raw probabilities generated by the individual classifiers are calibrated using a separate chronological calibration period.

The calibrated probabilities from:

- Logistic Regression;
- Random Forest;
- HistGradientBoosting;

are subsequently combined into an ensemble probability.

The ensemble is used to rank stocks according to their estimated probability of becoming future top-quartile performers.

---

## Market-Regime Modelling

A **Hidden Markov Model (HMM)** is used to estimate latent FTSE 100 market regimes.

The regime model considers information including:

- FTSE 100 returns;
- realised volatility;
- market drawdown.

The HMM is periodically estimated using an expanding historical window so that regime probabilities at each point in time are based only on information available up to that date.

Rather than relying on a binary risk-on/risk-off classification, the strategy uses posterior regime probabilities to adjust portfolio exposure continuously.

Conceptually:

```text
Lower Turbulence Probability
        ↓
Higher Permitted Exposure

Higher Turbulence Probability
        ↓
Reduced Portfolio Exposure
```

The FTSE 100 trend is also incorporated into the exposure mechanism.

The ML ensemble and HMM therefore perform different functions:

```text
Machine Learning → Which stocks?

Regime Model → How much portfolio exposure?
```

---

## Portfolio Construction

The portfolio is constructed once per week.

The process is:

1. Generate ML probabilities at the weekly signal date.
2. Apply the selected confidence threshold.
3. Rank eligible stocks by ensemble probability.
4. Select up to three stocks.
5. Apply the rank-buffer rule to reduce unnecessary turnover.
6. Calculate inverse-volatility portfolio weights.
7. Apply position limits.
8. Adjust total exposure using market-regime information.
9. Execute the portfolio on the next trading session.
10. Maintain each signal for five trading sessions.

---

## Turnover Control

A rank buffer is incorporated into the stock-selection process.

An existing holding does not necessarily need to be sold simply because it moves marginally outside the top three.

A stock may remain eligible while it remains sufficiently highly ranked and satisfies the strategy's confidence conditions.

The purpose is to reduce unnecessary portfolio churn and transaction costs caused by small changes in weekly model rankings.

---

## Execution Model

The strategy uses an explicit execution lag:

\[
Signal_t \rightarrow Execution_{t+1}
\]

Signals are generated using information available on the signal date, while portfolio execution occurs on the next available trading session.

Each weekly signal creates a five-trading-day holding sleeve.

Portfolio P&L is calculated daily throughout the holding period.

This maintains consistency between the:

\[
5\text{-day prediction horizon}
\]

and:

\[
5\text{-day portfolio holding period}
\]

---

## Transaction Costs

Transaction costs are based on actual changes in portfolio weights.

Turnover is calculated as:

\[
Turnover_t
=
\sum_i |w_{i,t}-w_{i,t-1}|
\]

and portfolio costs are:

\[
Cost_t
=
Turnover_t \times TransactionCost
\]

The primary strategy assumes:

\[
10\text{ basis points}
\]

per unit traded.

No transaction cost is charged when portfolio weights remain unchanged.

---

## Experimental Design

The dataset is divided chronologically into separate research periods:

| Stage | Period | Purpose |
|---|---|---|
| Training | 2014–2022 | Model fitting and hyperparameter selection |
| Calibration | Jan–Jun 2023 | Probability calibration |
| Validation | Jul–Dec 2023 | Strategy and confidence-threshold selection |
| Locked Test | 2024 onward | Final out-of-sample evaluation |

The locked test period is not used for hyperparameter optimisation, probability calibration or strategy-threshold selection.

---

## Unsupervised Learning

The project incorporates several unsupervised-learning techniques.

### Hierarchical Clustering

Used to examine historical relationships and behavioural similarities between stocks.

### Principal Component Analysis

Used to investigate latent factors and dimensional structure within the stock universe.

### Isolation Forest

Used to identify unusual market observations and periods of abnormal market behaviour.

### Hidden Markov Model

Used to estimate latent market regimes and their probabilities through time.

---

## Traditional Portfolio Benchmarks

Machine-learning performance is compared against traditional portfolio approaches.

The project generates **20,000 Monte Carlo portfolios** and identifies:

- Minimum Variance Portfolio;
- Maximum-Sharpe Portfolio.

Additional comparisons include:

- FTSE 100 benchmark;
- equal-weight stock portfolio;
- machine-learning portfolio.

This provides a broader comparison between traditional portfolio optimisation and the cross-sectional ML framework.

---

## Evaluation Metrics

Predictive performance is evaluated using:

- ROC-AUC;
- Precision;
- Recall;
- F1 score;
- Log Loss;
- Brier Score;
- Precision@3;
- realised top-3 benchmark-relative return.

Portfolio performance is evaluated using:

- annualised return;
- annualised volatility;
- Sharpe ratio;
- Sortino ratio;
- maximum drawdown;
- Value at Risk;
- Expected Shortfall;
- final portfolio value;
- turnover;
- transaction-cost sensitivity.

---

## Out-of-Sample Results

On the locked out-of-sample test period, the ML ensemble demonstrated predictive performance above the natural cross-sectional baseline.

### Predictive Performance

| Metric | Result |
|---|---:|
| ROC-AUC | **0.545** |
| Precision@3 | **30.99%** |
| Random Top-3 baseline | **25.00%** |
| Relative Precision@3 lift | **~24%** |

The predictive edge is deliberately interpreted as modest rather than as evidence of highly accurate market forecasting.

The objective is to determine whether a relatively small ranking advantage can be converted into economically meaningful portfolio performance.

### Portfolio Performance

| Metric | Regime-Aware ML | FTSE 100 | Equal Weight |
|---|---:|---:|---:|
| Annualised Return | **25.74%** | 13.26% | 16.25% |
| Sharpe Ratio | **1.480** | 1.162 | 1.441 |

The ML portfolio therefore generated approximately:

\[
25.74\%-13.26\%=\mathbf{12.48}
\]

percentage points of annualised return above the FTSE 100 during the locked test period.

These results represent historical out-of-sample performance within the experimental framework and should not be interpreted as a guarantee of future returns.

---

## Robustness and Sensitivity Analysis

The final strategy is evaluated using several additional diagnostics without using the locked test period for further model selection.

The robustness analysis examines:

- ML predictive skill against random and simple ranking baselines;
- confidence-threshold integrity;
- ML-only versus regime-aware portfolio variants;
- HMM and FTSE trend contributions;
- rank-buffer effectiveness;
- transaction costs from 0 to 50 basis points;
- approximate break-even transaction costs;
- annual performance stability;
- rolling Sharpe ratios;
- portfolio turnover;
- alpha efficiency;
- economic feature-group importance;
- model-family probability agreement.

These tests are designed to determine whether the observed performance is supported by broader predictive, economic and risk-management evidence.

---

## Backtest Integrity

The notebook includes explicit checks of the portfolio engine.

The audit verifies that:

- portfolio execution occurs after the corresponding signal;
- transaction costs are charged when portfolio weights change;
- signal frequency follows the weekly strategy specification;
- each signal uses the defined five-trading-day holding period;
- portfolio P&L is calculated daily.

These checks help ensure chronological consistency between prediction, execution and performance measurement.

---

## Key Research Finding

The central result of the project is not that machine learning predicts future stock returns with high accuracy.

Instead, the experiment demonstrates how a **modest cross-sectional predictive advantage** can be incorporated into a systematic portfolio architecture.

The complete process combines:

\[
\text{Prediction}
\rightarrow
\text{Ranking}
\rightarrow
\text{Confidence Filtering}
\rightarrow
\text{Risk Allocation}
\rightarrow
\text{Regime Control}
\rightarrow
\text{Execution}
\rightarrow
\text{Cost Control}
\]

The ML models determine **which stocks appear relatively attractive**, while the portfolio and regime layers determine **how that predictive information is converted into controlled market exposure**.

---

## Limitations

The results should be interpreted within the limitations of the research design.

### Survivorship Bias

The analysis uses a fixed contemporary sample of 12 FTSE 100 stocks rather than a historical point-in-time constituent database.

### Restricted Investment Universe

The 12-stock universe represents only a subset of the complete FTSE 100.

### Data Source

`yfinance` provides accessible historical market data but is not an institutional point-in-time market database.

### Transaction Costs

The backtest incorporates explicit turnover-based transaction costs but does not fully model market impact, broker-specific spreads, taxes or all intraday execution effects.

### Model Risk

Relationships identified by machine-learning models may change as financial markets evolve.

### Regime Uncertainty

HMM regimes represent probabilistic statistical estimates rather than directly observable market states.

### Non-Stationarity

Financial markets are non-stationary, meaning that relationships observed historically may weaken, disappear or reverse in future periods.

For these reasons, historical out-of-sample performance should not be interpreted as a guarantee of future profitability.

---

## Technology

The project is implemented in Python using libraries including:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `scipy`
- `yfinance`
- `hmmlearn`

---

## Repository Structure

```text
ftse100-regime-aware-ml/
│
├── README.md
├── notebooks/
│   └── regime_aware_ftse100_ml.ipynb
│
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Running the Notebook

Clone the repository:

```bash
git clone <repository-url>
cd ftse100-regime-aware-ml
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebook using Jupyter:

```bash
jupyter notebook notebooks/regime_aware_ftse100_ml.ipynb
```

The notebook can also be executed in Google Colab.

---

## Reproducibility

A fixed random state is used throughout stochastic components where supported:

```python
RANDOM_STATE = 42
```

Because market data is retrieved from an external provider, small differences may occur if the underlying historical dataset is subsequently revised by the data source.

---

## Academic Context

This project was developed as part of quantitative finance and machine-learning research into systematic portfolio construction.

The emphasis is placed on:

- chronological experimental design;
- leakage prevention;
- out-of-sample evaluation;
- interpretable financial features;
- realistic portfolio execution;
- benchmark comparison;
- risk-adjusted performance;
- robustness analysis.

---

## Disclaimer

This repository is provided for academic and research purposes. The strategy, analysis and historical results do not constitute investment advice and do not guarantee future performance.