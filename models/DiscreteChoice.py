import numpy as np
from sklearn.cluster import KMeans
import cvxpy as cp
import matplotlib.pyplot as plt


class InvestorPortfolioModel:
    def __init__(self, choice_data, stock_data, n_attributes=4, n_levels=[4, 4, 5, 3]):
        self.choice_data = choice_data
        self.stock_data = stock_data
        self.n_respondents = choice_data.shape[0]
        self.n_attributes = n_attributes
        self.n_levels = n_levels
        self.partworths = None
        self.segments = None
        self.shares = None

    def estimate_partworths(self):
        np.random.seed(42)
        self.partworths = np.random.normal(
            0, 1,
            (self.n_respondents, self.n_attributes, max(self.n_levels)))
        return self.partworths

    def calculate_attribute_importance(self):
        ranges = np.zeros(self.n_attributes)
        for k in range(self.n_attributes):
            utility = self.partworths[:, k, :self.n_levels[k]]
            ranges[k] = np.mean(np.max(utility, axis=1) - np.min(utility, axis=1))
        return ranges / np.sum(ranges)

    def segment_respondents(self, n_clusters=4):
        flattened = self.partworths.reshape(self.n_respondents, -1)
        self.segments = KMeans(n_clusters=n_clusters).fit_predict(flattened)
        return self.segments

    def simulate_shares(self, scale_param=1.0):
        stock_levels = {
            "Nikola Tesla Airport": [1, 2, 3, 1],
            "Metalac a.d.": [0, 0, 4, 0],
            "Energoprojekt holding": [3, 2, 1, 2],
            "Jedinstvo a.d.": [2, 0, 3, 0]
        }

        utilities = {}
        for stock, levels in stock_levels.items():
            stock_utility = np.zeros(self.n_respondents)
            for i in range(4):
                stock_utility += self.partworths[:, i, levels[i]]
            utilities[stock] = stock_utility * scale_param

        exp_utilities = np.exp(np.array(list(utilities.values())))
        self.shares = exp_utilities / np.sum(exp_utilities, axis=0)
        return self.shares

    def portfolio_performance(self):
        n_stocks = len(self.stock_data)
        returns = np.array([v["return"] + v["dividend"] / 100 for v in self.stock_data.values()])
        weights = np.mean(self.shares, axis=1)

        # Fix covariance matrix dimensions
        cov_matrix = np.cov(np.random.randn(n_stocks, 100))  # Corrected to (n_stocks, samples)

        expected_return = weights @ returns
        risk = np.sqrt(weights.T @ cov_matrix @ weights)
        return expected_return, risk

    def markowitz_frontier(self, target_risk):
        returns = np.array([v["return"] + v["dividend"] / 100 for v in self.stock_data.values()])
        n_stocks = len(returns)
        cov_matrix = np.cov(np.random.randn(n_stocks, 100))  # Corrected

        w = cp.Variable(n_stocks)
        objective = cp.Maximize(returns.T @ w)
        constraints = [
            cp.quad_form(w, cov_matrix) <= target_risk ** 2,
            cp.sum(w) == 1,
            w >= 0
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        return w.value

    def plot_results(self, portfolio_return, portfolio_risk):
        returns_array = np.array([v["return"] + v["dividend"] / 100 for v in self.stock_data.values()])

        risks = np.linspace(0.1, 2, 20)
        optimal_returns = []

        for r in risks:
            try:
                weights = self.markowitz_frontier(r)
                optimal_returns.append(weights @ returns_array)
            except:
                optimal_returns.append(0)

        plt.figure(figsize=(10, 6))
        plt.plot(risks, optimal_returns, label='Efficient Frontier')
        plt.scatter(portfolio_risk, portfolio_return,
                    c='red', s=100, label='Preference Portfolio')
        plt.xlabel('Risk')
        plt.ylabel('Return')
        plt.legend()
        plt.show()


# Test the model
if __name__ == "__main__":
    np.random.seed(42)
    test_choice_data = np.random.randint(0, 4, (100, 10))
    test_stock_data = {
        "Nikola Tesla Airport": {"return": 0.12, "dividend": 4.54},
        "Metalac a.d.": {"return": 0.01, "dividend": 8.35},
        "Energoprojekt holding": {"return": 0.09, "dividend": 1.28},
        "Jedinstvo a.d.": {"return": 0.02, "dividend": 4.24}
    }

    model = InvestorPortfolioModel(test_choice_data, test_stock_data)
    model.estimate_partworths()
    print("Attribute Importance:", model.calculate_attribute_importance())

    model.segment_respondents()
    shares = model.simulate_shares(scale_param=0.8)
    print("\nAverage Portfolio Shares:")
    for stock, share in zip(test_stock_data.keys(), np.mean(shares, axis=1)):
        print(f"{stock}: {share:.2%}")

    ret, risk = model.portfolio_performance()
    print(f"\nPortfolio Return: {ret:.2%}, Risk: {risk:.2f}")
    model.plot_results(ret, risk)