import pandas as pd

#Read the file
aapl = pd.read_csv('data/aapl.csv')
amzn = pd.read_csv('data/amzn.csv')
cvs = pd.read_csv('data/cvs.csv')
nvda = pd.read_csv('data/nvda.csv')
wmt = pd.read_csv('data/wmt.csv')

# Create a new file only with data and prices:


def clean_and_order(df, startdate, enddate):

    # Convert date, filter range and order it:

    df["Date"]=pd.to_datetime(df["Date"])
    df = df[ (df["Date"]>=startdate)&(df["Date"]<=enddate)]
    df = df.sort_values("Date")

    # Remove $ from price and covert to float:

    df["Close/Last"]=df["Close/Last"].astype(str)  # lo pasamos a string
    df["Close/Last"]=df["Close/Last"].str.replace("$", "").str.replace(",", "") #eliminamos $ y ,
    df["Close/Last"]=df["Close/Last"].astype(float)  # lo pasamos a float

    return df[["Date","Close/Last"]]

stocks={"AAPL":aapl,
        "AMZN":amzn,
        "CVS":cvs,
        "NVDA":nvda,
        "WMT":wmt}

cleaned = pd.DataFrame()

#STOCKS IN YEAR 2022:

for name, df in stocks.items():
    temp = clean_and_order(df,"01/01/2022","12/31/2022")
    temp = temp.set_index('Date')
    cleaned[name]=temp["Close/Last"]

print("Stock closes:")
print(cleaned.head())

#Calculate daily returns:
returns = (cleaned - cleaned.shift(1)) / cleaned.shift(1)

#Drop the first row (is empty):
returns= returns.dropna()

print("Daily returns: ")
print(returns.head())

# Calculate daily percentage returns for all stocks

returns_mean = returns.mean()
print("Mean daily returns (per asset):")
print(returns_mean)



print("Min/Max mean daily return:")
print(returns_mean.min(), returns_mean.max())


#Calculate covariance matrix:


covariance_matrix = returns.cov()
print("Covariance matrix (Σ):")
print(covariance_matrix)


#The problem is to minimize portfolio risk: Portfolio Risk: w^T * Σ * w
# subject to the return constraint, the budget constraint and no short selling.


import numpy as np
from scipy.optimize import minimize

# Porfolio risk (objective function):

def Risk(weights, covariance_matrix):
    return weights.T @ covariance_matrix @ weights

# Return constraint:

def ReturnConstr(weights, returns_mean, min_return):
    return returns_mean @ weights - min_return

#Budget constraint:

def BudgetConstr(weights):
    return np.sum(weights) - 1

# Define return targets

min_daily = float(returns_mean.min())
max_daily = float(returns_mean.max())

# pick targets inside [min_daily, max_daily]
daily_min_returns = np.linspace(min_daily, max_daily, 40)

# Now compute annualized targets (for plotting colors)
annualized_min_returns = daily_min_returns * 252

print("Feasible daily mean range:", min_daily, "to", max_daily)
print("Daily min returns:", daily_min_returns)

n_assets = len(returns_mean)

# Initial guess, required by minimize function (equal weights)
w0 = np.ones(n_assets) / n_assets

print("initial guess w0: ",w0)

# Bounds: no short selling (0 <= wi <= 1)
bounds = [(0, 1) for _ in range(n_assets)]


# Lists to store results for the efficient frontier
portfolio_risks = []
portfolio_returns = []
portfolio_weights = []
successful_annualized_min_returns_for_plot = [] # New list to store annualized targets for successful optimizations


# Run optimization for each of the daily min returns.

for i, min_daily_return in enumerate(daily_min_returns):
    constraints = [
        {'type': 'eq',   'fun': BudgetConstr},
        {'type': 'ineq', 'fun': ReturnConstr, 'args': (returns_mean, min_daily_return)}
    ]

    # Optimization
    result = minimize(
        Risk,
        w0,
        args=(covariance_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    # Store results if optimization was successful
    if result.success:
        weights = result.x
        #annualize:
        var_daily = result.fun
        risk = np.sqrt(var_daily) * np.sqrt(252)      # annualized volatility
        ret  = (returns_mean @ weights) * 252         # annualized return

        portfolio_risks.append(risk)
        portfolio_returns.append(ret)
        portfolio_weights.append(weights)
        successful_annualized_min_returns_for_plot.append(annualized_min_returns[i]) # Store the corresponding annualized target
    else:
        print(f"Optimization failed for min_return {min_daily_return:.6f}: {result.message}")

print("Optimization for efficient frontier complete.")
print("Number of successful optimizations:", len(portfolio_risks))

# Displaying the first set of optimal weights and portfolio performance for context
if portfolio_risks:
    print("\nFirst set of optimal weights (for the lowest target return):")
    for i, w in enumerate(portfolio_weights[0]):
        print(f"Asset {i+1}: {w:.4f}")
    print(f"\nFirst portfolio return: {portfolio_returns[0]:.6f}")
    print(f"First portfolio risk:   {portfolio_risks[0]:.6f}")
else:
    print("No successful optimizations.")


import matplotlib.pyplot as plt

# Plotting the Efficient Frontier
plt.figure(figsize=(10, 6))
plt.scatter(portfolio_risks, portfolio_returns, c=successful_annualized_min_returns_for_plot, cmap='viridis')
plt.colorbar(label='Annualized Minimum Return Target')
plt.title('Efficient Frontier')
plt.xlabel("Annualized risk")
plt.ylabel("Annualized Return")
plt.grid(True)
plt.show()


