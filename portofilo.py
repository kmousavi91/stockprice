import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

# Step 1: Define Stock Symbols
stocks = ["AAPL", "GOOGL", "AMZN", "MSFT", "TSLA"]

# Step 2: Download Historical Stock Data
start_date = "2018-01-01"
end_date = "2023-01-01"

data = yf.download(stocks, start=start_date, end=end_date)['Close']

# Step 3: Calculate Daily Returns
returns = data.pct_change().dropna()

# Step 4: Define Portfolio Optimization Function
def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(mean_returns * weights) * 252  # Annualized return
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualized volatility
    sharpe_ratio = portfolio_return / portfolio_volatility  # Sharpe ratio
    return -sharpe_ratio  # Minimize negative Sharpe ratio

# Step 5: Calculate Mean Returns and Covariance Matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Step 6: Set Optimization Constraints
num_assets = len(stocks)
initial_weights = np.ones(num_assets) / num_assets  # Equal allocation
bounds = tuple((0, 1) for _ in range(num_assets))  # Weights between 0 and 1
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # Weights must sum to 1

# Step 7: Perform Portfolio Optimization
optimized = minimize(portfolio_performance, initial_weights, args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = optimized.x

# Step 8: Display Optimal Portfolio Allocation
print("Optimal Portfolio Weights:")
for stock, weight in zip(stocks, optimal_weights):
    print(f"{stock}: {weight:.2%}")

# Step 9: Visualize Optimal Portfolio Allocation
plt.figure(figsize=(8, 5))
plt.pie(optimal_weights, labels=stocks, autopct='%1.1f%%', startangle=140, colors=['blue', 'red', 'green', 'purple', 'orange'])
plt.title("Optimized Portfolio Allocation")
plt.show()
