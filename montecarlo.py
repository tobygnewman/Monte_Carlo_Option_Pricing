import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """
    Black-Scholes analytical pricing of a European call option.

    Parameters:
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the underlying asset
    num_simulations : int, optional
        Number of simulated paths (default is 100,000)

    Returns:
    float
        Discounted expected payoff of the call option

    Notes:
    - The position profits if S_T > K and expires worthless if S_T <= K.
    - Uses the standard Black-Scholes formula with cumulative normal probabilities.
    """
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    """
    Black-Scholes analytical pricing of a European put option.

    Parameters:
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the underlying asset
    num_simulations : int, optional
        Number of simulated paths (default is 100,000)

    Returns:
    float
        Discounted expected payoff of the put option

    Notes:
    - The position profits if S_T < K and expires worthless if S_T >= K.
    - Uses the standard Black-Scholes formula with cumulative normal probabilities.
    """
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def monte_carlo_call(S, K, T, r, sigma, num_simulations=100000):
    """
    Monte Carlo pricing of a European call option.

    Parameters:
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the underlying asset
    num_simulations : int, optional
        Number of simulated paths (default is 100,000)

    Returns:
    float
        Discounted expected payoff of the call option

    Notes:
    - The position profits if S_T > K and expires worthless if S_T <= K.
    - Simulates end-of-period stock prices and averages discounted payoffs.
    """
    Z = np.random.standard_normal(num_simulations)
    ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(ST - K, 0)
    discounted_payoff = np.exp(-r*T) * payoff
    
    price = np.mean(discounted_payoff)
    std_error = np.std(discounted_payoff, ddof=1) / np.sqrt(num_simulations)

    return price, std_error

def monte_carlo_put(S, K, T, r, sigma, num_simulations=100000):
    """
    Monte Carlo pricing of a European put option.

    Parameters:
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the underlying asset
    num_simulations : int, optional
        Number of simulated paths (default is 100,000)

    Returns:
    float
        Discounted expected payoff of the put option

    Notes:
    - The position profits if S_T < K and expires worthless if S_T >= K.
    - Simulates end-of-period stock prices and averages discounted payoffs.
    """
    Z = np.random.standard_normal(num_simulations)
    ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(K - ST, 0)
    discounted_payoff = np.exp(-r*T) * payoff
    
    price = np.mean(discounted_payoff)
    std_error = np.std(discounted_payoff, ddof=1) / np.sqrt(num_simulations)

    return price, std_error