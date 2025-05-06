import numpy as np
import pandas as pd
from scipy.optimize import minimize


def fpsmsc_portfolio(historical_returns, lambda_param):
    """
    Strict implementation of FPSMSC from the paper with error correction
    """
    returns = historical_returns.values

    # 1. Fuzzification (as per paper)
    a = np.median(returns, axis=0)
    alpha = a - np.min(returns, axis=0)
    beta = np.max(returns, axis=0) - a

    # 2. Credibility measures (exact paper formulas)
    E_c = a + (beta - alpha) / 4

    V_c = np.zeros_like(a)
    for i in range(len(a)):
        alph, bet = alpha[i], beta[i]
        if alph > bet:
            V_c[i] = (33 * alph ** 3 + 11 * alph * bet ** 2 + 21 * alph ** 2 * bet - bet ** 3) / (384 * alph)
        elif alph == bet:
            V_c[i] = alph ** 2 / 24
        else:
            V_c[i] = (34 * bet ** 3 + 21 * alph * bet ** 2 + 11 * alph ** 2 * bet - alph ** 3) / (384 * bet)

    # 3. Stochastic correlation matrix (critical fix)
    corr_matrix = np.cov(returns, rowvar=False)
    sigma = np.std(returns, axis=0)
    corr_matrix = np.outer(sigma, sigma)  # Remove this line - this was the error

    # Correct correlation calculation
    corr_matrix = pd.DataFrame(returns).corr().values

    # 4. Hybrid covariance matrix (as per equation 13)
    sigma_c = np.sqrt(V_c)
    V_hybrid = corr_matrix * np.outer(sigma_c, sigma_c)  # Element-wise multiplication

    # 5. Optimization (exact paper formulation)
    n_assets = len(a)

    def objective(weights):
        risk = weights @ V_hybrid @ weights.T
        ret = weights @ E_c
        return lambda_param * risk - (1 - lambda_param) * ret

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = [(0, 1) for _ in range(n_assets)]

    res = minimize(
        objective,
        x0=np.ones(n_assets) / n_assets,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return res.x


# Correct synthetic data generation with correlations
def generate_correlated_returns(n_assets=5, n_periods=48):
    # Create positive definite correlation matrix
    base_corr = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            base_corr[i, j] = base_corr[j, i] = 0.3 + 0.5 * (i + j) / (2 * n_assets)

    # Generate correlated returns
    chol = np.linalg.cholesky(base_corr)
    uncorrelated = np.random.normal(0.02, 0.5, (n_periods, n_assets))
    return pd.DataFrame(uncorrelated @ chol.T)


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    historical_data = generate_correlated_returns(n_assets=5)

    weights = fpsmsc_portfolio(historical_data, lambda_param=0.3)

    print("Optimal Weights:")
    print(np.round(weights, 4))