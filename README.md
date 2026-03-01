# Monte Carlo Pricing of European Options

This project implements Monte Carlo simulation to price European call and put options under the Black–Scholes framework.

## Features

- Geometric Brownian Motion simulation under risk-neutral measure
- Black–Scholes analytical pricing
- Monte Carlo estimator with standard errors
- 95% confidence intervals
- Put–call parity validation
- Convergence analysis
- Sensitivity analysis (volatility, strike, maturity, interest rate)

## Key Results

- Monte Carlo estimates converge to analytical Black–Scholes prices.
- Pricing error decreases at rate 1/√N.
- Put–call parity holds within statistical confidence intervals.
- Sensitivity analysis confirms theoretical monotonic relationships.

## Technologies Used

- Python
- NumPy
- Matplotlib

## Author

Toby Newman