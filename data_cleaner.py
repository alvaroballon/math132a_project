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
daily_min_returns = np.linspace(min_daily, max_daily, 20)

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


##USING KKT conditions:

def active_set_kkt_boomerang(Sigma, mu, R_min, max_iter=100):
    n = len(mu)
    Q = 2 * Sigma  # Hessian of the objective function w^T Sigma w
    
    # --- MODIFICATION ---
    # Define Equality Constraints: E * w = d 
    # Row 1: Budget constraint (Sum of w = 1)
    # Row 2: Return constraint (mu^T * w = R_min)
    E = np.vstack([np.ones(n), mu])
    d = np.array([1.0, R_min])
    
    # Define Inequality Constraints: C * w >= b
    # Now this ONLY contains the no-short-selling bounds: I * w >= 0
    C = np.eye(n)
    b = np.zeros(n)
    
    # --- PHASE 1: Find an initial feasible portfolio ---
    w = np.zeros(n)
    idx_max, idx_min = np.argmax(mu), np.argmin(mu)
    
    if mu[idx_max] == mu[idx_min]:
        w[:] = 1.0 / n
    else:
        # Mix the min and max return asset to exactly hit R_min
        alpha = (R_min - mu[idx_min]) / (mu[idx_max] - mu[idx_min])
        alpha = np.clip(alpha, 0.0, 1.0)
        w[idx_max] = alpha
        w[idx_min] = 1.0 - alpha
        
    # Track "Active" inequalities (The Working Set 'W')
    W = []
    for i in range(len(b)):
        if abs(np.dot(C[i], w) - b[i]) < 1e-7:
            W.append(i)
            
    # --- PHASE 2: Main Active Set Loop ---
    m_eq = E.shape[0] # Number of equality constraints is now 2
    
    for _ in range(max_iter):
        
        # 1. Build KKT matrix for the current active constraints
        A_k = np.vstack([E, C[W, :]]) if len(W) > 0 else E
        m_active = A_k.shape[0]
        
        # Matrix form
        KKT_top = np.hstack([Q, -A_k.T])
        KKT_bot = np.hstack([A_k, np.zeros((m_active, m_active))])
        KKT_matrix = np.vstack([KKT_top, KKT_bot])
        
        # Right-hand side (Gradient of current state)
        rhs = np.concatenate([-np.dot(Q, w), np.zeros(m_active)])
        
        # Solve for direction 'p' and multipliers 'lambdas'
        try:
            sol = np.dot(np.linalg.pinv(KKT_matrix), rhs) 
        except np.linalg.LinAlgError:
            break
            
        p = sol[:n]
        lambdas = sol[n:]
        
        # 2. Are we at the optimal point for this working set?
        if np.linalg.norm(p) < 1e-7:
            if len(W) == 0:
                break # Optimal!
                
            # Check Dual Feasibility: Are any multipliers for inequalities negative?
            # IMPORTANT: The first 'm_eq' lambdas belong to equalities, we only check inequalities
            lambda_ineq = lambdas[m_eq:] 
            
            if len(lambda_ineq) == 0:
                break
                
            min_lambda_idx = np.argmin(lambda_ineq)
            
            if lambda_ineq[min_lambda_idx] >= -1e-7:
                break # Optimal! All KKT conditions met.
            else:
                # Drop the constraint that most wants to be relaxed
                W.pop(min_lambda_idx)
        
        # 3. If not optimal, move in direction 'p' but don't violate inactive constraints
        else:
            alpha_step = 1.0
            blocking_idx = -1
            
            for i in range(len(b)):
                if i not in W:
                    c_p = np.dot(C[i], p)
                    if c_p < -1e-7: # Moving towards a violation
                        ratio = (b[i] - np.dot(C[i], w)) / c_p
                        if ratio < alpha_step:
                            alpha_step = ratio
                            blocking_idx = i
                            
            # Update weights
            w = w + alpha_step * p
            
            # Primal Feasibility: Add any newly hit constraint to the working set
            if alpha_step < 1.0 and blocking_idx != -1:
                if blocking_idx not in W:
                    W.append(blocking_idx)
                    
    return np.maximum(w, 0.0) # Ensure strictly non-negative at return

# Lists to store results for the efficient frontier
portfolio_risks = []
portfolio_returns = []
portfolio_weights = []
successful_annualized_min_returns_for_plot = []

# Convert pandas to pure NumPy arrays to feed the KKT math safely
Sigma_array = covariance_matrix.values
mu_array = returns_mean.values

print("Running Custom KKT Active Set Optimization...")

for i, min_daily_return in enumerate(daily_min_returns):
    try:
        # Run custom algorithm
        weights = active_set_kkt_boomerang(Sigma_array, mu_array, min_daily_return)
        
        # Annualize and map values
        var_daily = weights.T @ Sigma_array @ weights
        risk = np.sqrt(var_daily) * np.sqrt(252)      # annualized volatility
        ret = (mu_array @ weights) * 252              # annualized return
        
        portfolio_risks.append(risk)
        portfolio_returns.append(ret)
        portfolio_weights.append(weights)
        successful_annualized_min_returns_for_plot.append(annualized_min_returns[i])
        
    except Exception as e:
        print(f"Optimization failed for min_return {min_daily_return:.6f}: {e}")

print("Optimization for efficient frontier complete.")
print("Number of successful optimizations:", len(portfolio_risks))


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


#BACKTESTING:

#Preparar los precios "out of sample":

start_date = '2023-01-01'
end_date = '2023-12-31'

cleaned = {}

for symbol, df in stocks.items():
    # Pass correct column names 'Date' and 'Close/Last'
    temp = clean_and_order(df, start_date, end_date)
    temp = temp.set_index('Date')      # index = Date
    cleaned[symbol] = temp['Close/Last']     # keep only Close/Last

out_sample_stock_closes = pd.concat(cleaned, axis=1)
out_sample_stock_closes.columns = ['AAPL', 'AMZN', 'CVS', 'NVDA', 'WMT']

# Comprobar los datos
print("out_sample_stock_closes.head():")
print(out_sample_stock_closes.head())
print("out_sample_stock_closes.info():")
out_sample_stock_closes.info()

# Check if all stocks share the same trading days
print("Rows with NaN values (should be empty if no missing data):")
print(out_sample_stock_closes[out_sample_stock_closes.isna().any(axis=1)])





def CumulativeReturn(weights, out_sample_daily_returns):
  daily_portfolio_returns = out_sample_daily_returns.dot(weights)

  return (1 + daily_portfolio_returns).prod() - 1

def SharpeRatio(weights, out_sample_daily_returns, risk_free_rate=0.04):
  daily_portfolio_returns = out_sample_daily_returns.dot(weights)

  # Calculate daily risk-free rate from annualized risk_free_rate
  daily_risk_free_rate = (1 + risk_free_rate)**(1/252) - 1

  excess_returns = daily_portfolio_returns - daily_risk_free_rate
  portfolio_std = excess_returns.std()

  if portfolio_std == 0:
      return 0 # Avoid division by zero
  return excess_returns.mean() / portfolio_std


# Calculate daily percentage returns for out-of-sample stocks
out_sample_daily_returns = out_sample_stock_closes.pct_change().dropna()


cumulative_returns = []
sharpe_ratios = []
portfolio_stds = []

for weights in portfolio_weights:
    # Calculate cumulative return for the out-of-sample period
    cr = CumulativeReturn(weights, out_sample_daily_returns)
    cumulative_returns.append(cr)

    # Calculate Sharpe Ratio for the out-of-sample period
    sr = SharpeRatio(weights, out_sample_daily_returns)
    sharpe_ratios.append(sr)

    # Calculate daily portfolio returns for standard deviation
    daily_portfolio_returns = out_sample_daily_returns.dot(weights)
    portfolio_stds.append(daily_portfolio_returns.std())

# Find the portfolio with the highest cumulative return
max_cr_idx = np.argmax(cumulative_returns)
best_cr_weights = portfolio_weights[max_cr_idx]
best_cr_value = cumulative_returns[max_cr_idx]
best_cr_sharpe = sharpe_ratios[max_cr_idx]
best_cr_std = portfolio_stds[max_cr_idx]

# Find the portfolio with the highest Sharpe ratio
max_sharpe_ratio_idx = np.argmax(sharpe_ratios)
best_sharpe_weights = portfolio_weights[max_sharpe_ratio_idx]
best_sharpe_value = sharpe_ratios[max_sharpe_ratio_idx]
best_sharpe_cr = cumulative_returns[max_sharpe_ratio_idx]
best_sharpe_std = portfolio_stds[max_sharpe_ratio_idx]

print("\nPortfolio with Highest Cumulative Return (Out-of-Sample):")
print(f"  Weights: {best_cr_weights}")
print(f"  Cumulative Return: {best_cr_value:.4f}")
print(f"  Sharpe Ratio: {best_cr_sharpe:.4f}")
print(f"  Daily Portfolio Standard Deviation: {best_cr_std:.4f}")

print("\nPortfolio with Highest Sharpe Ratio (Out-of-Sample):")
print(f"  Weights: {best_sharpe_weights}")
print(f"  Cumulative Return: {best_sharpe_cr:.4f}")
print(f"  Sharpe Ratio: {best_sharpe_value:.4f}")
print(f"  Daily Portfolio Standard Deviation: {best_sharpe_std:.4f}")