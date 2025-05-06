import numpy as np

import profile
from models.fuzzygrey import FuzzyGreyPortfolioOptimizer
from models.StochasticCorrelation import generate_correlated_returns
from models.StochasticCorrelation import fpsmsc_portfolio
from models.bicriteria import BicriteriaPortfolio
from models.TurningPoint import TurningPointModel


# Example parameters (adjust as needed)
params = {
    'N': 100,  # Number of assets
    'POP': 50,  # Reduced for testing
    'MG': 100,  # Reduced generations for testing
    'K': 10,  # Cardinality constraint
    'l_bounds': [0.01] * 100,  # Lower bounds
    'u_bounds': [0.1] * 100,  # Upper bounds
    'w1': 0.8, 'w2': 0.2,  # CLSIM weights
    'p_c': 0.8, 'p_m': 0.1,  # Crossover/mutation probabilities
    'F': 5,  # Mutation parameter
    'I_a': 0.05, 'I_b': 0.15, 'I_c': 0.25,  # Ideal return
    'l': 1.0,  # Neutral investor
    'L': 100,  # Historical data length
    'seed': 42,
    'historical_returns': np.random.randn(100, 100) * 0.01  # Simulated data
}

optimizer = FuzzyGreyPortfolioOptimizer(params)
best_x, best_y = optimizer.optimize()

print("Optimal Weights:", best_x[best_y == 1])
print("Selected Assets:", np.where(best_y == 1)[0])

###############################################################

np.random.seed(42)
historical_data = generate_correlated_returns(n_assets=5)

weights = fpsmsc_portfolio(historical_data, lambda_param=0.3)

print("Optimal Weights:")
print(np.round(weights, 4))



############################################################################


# Define fuzzy assets from Example 10
assets_fuzzy = [
    [2, 3, 4, 7],  # Asset 11
    [2, 5, 6, 7],  # Asset 12
]

# Initialize and optimize
model = BicriteriaPortfolio(assets_fuzzy)
result = model.optimize(
    method='D1', w_risk=0.5, w_return=0.5,
    as_min=0.01, as_max=0.85,
    init_weights=np.array([0.02, 0.05, 0.04, 0.03, 0.01, 0.85])
)

print("Optimized Weights:", result['weights'])
print(f"PARisk: {result['PARisk']:.2f}, OOPR: {result['OOPR']:.2f}")

model = TurningPointModel('AAPL', '2020-01-01', '2023-01-01')
results = model.analyze()

print("\nSample Classifications:")
print(results[['Close', 'Classification', 'Positive_Turning', 'Negative_Turning']].tail(10))






