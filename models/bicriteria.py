import numpy as np
from scipy.optimize import differential_evolution, LinearConstraint


class BicriteriaPortfolio:
    def __init__(self, assets, alpha_levels=np.linspace(0, 1, 11)):
        """
        Initialize with α-cut-based arithmetic.
        """
        self.assets = np.array(assets)
        self.alpha_levels = alpha_levels
        self.n_assets = len(assets)

        # Precompute global bounds across all α-cuts
        self.OPR_min = min(asset[0] for asset in self.assets)  # α=0 lower bound
        self.OPR_max = max(asset[-1] for asset in self.assets)  # α=0 upper bound

    def compute_alpha_cuts(self, weights):
        """Compute portfolio's α-cut intervals."""
        alpha_intervals = []
        for alpha in self.alpha_levels:
            # Compute α-cut bounds for each asset
            lowers, uppers = [], []
            for asset in self.assets:
                a, b, c, d = asset
                lower = a + alpha * (b - a)  # Left side of trapezoid
                upper = d - alpha * (d - c)  # Right side
                lowers.append(lower)
                uppers.append(upper)

            # Portfolio α-cut interval
            port_lower = np.dot(weights, lowers)
            port_upper = np.dot(weights, uppers)
            alpha_intervals.append((port_lower, port_upper))
        return alpha_intervals

    def compute_criteria(self, weights):
        """Calculate PARisk and OOPR using α-cut aggregation."""
        alpha_intervals = self.compute_alpha_cuts(weights)
        PARisk_total, OOPR_total = 0.0, 0.0

        for alpha, (port_lower, port_upper) in zip(self.alpha_levels, alpha_intervals):
            # PARisk: (global_max - port_lower) / (global_max - global_min)
            PARisk_alpha = 1 - (self.OPR_max - port_lower) / (self.OPR_max - self.OPR_min + 1e-9)

            # OOPR: (global_max - port_upper) / (global_max - global_min)
            OOPR_alpha = 1 - (self.OPR_max - port_upper) / (self.OPR_max - self.OPR_min + 1e-9)

            PARisk_total += alpha * PARisk_alpha
            OOPR_total += alpha * OOPR_alpha

        # Normalize by sum of α-level weights
        total_alpha = sum(self.alpha_levels)
        return PARisk_total / total_alpha, OOPR_total / total_alpha

    def optimize(self, w_risk=0.5, w_return=0.5, method='D1',
                 as_min=0.01, as_max=0.85, init_weights=None):
        # Constraints: sum(weights) = 1
        constraint = LinearConstraint(np.ones(self.n_assets), lb=1, ub=1)
        bounds = [(as_min, as_max) for _ in range(self.n_assets)]

        # Objective function
        def objective(weights):
            PARisk, OOPR = self.compute_criteria(weights)
            if method == 'D1':
                return -np.min([PARisk ** w_risk, OOPR ** w_return])
            elif method == 'D2':
                return -(PARisk ** w_risk * OOPR ** w_return)
            elif method == 'D3':
                return -(w_risk * PARisk + w_return * OOPR)
            else:
                raise ValueError("Method must be D1/D2/D3")

        # Initial population including paper's weights
        if init_weights is None:
            init_weights = np.ones(self.n_assets) / self.n_assets
        init_pop = np.vstack([
            init_weights,
            np.random.dirichlet(np.ones(self.n_assets), size=49)
        ])

        # Global optimization
        result = differential_evolution(
            objective, bounds=bounds, constraints=constraint,
            popsize=50, maxiter=2000, polish=False, init=init_pop,
            mutation=(0.5, 1.2), recombination=0.8
        )

        opt_weights = np.round(result.x, 2)
        opt_weights /= opt_weights.sum()  # Ensure sum=1 after rounding

        PARisk, OOPR = self.compute_criteria(opt_weights)
        return {
            'weights': opt_weights,
            'PARisk': PARisk,
            'OOPR': OOPR,
            'objective': -result.fun
        }


# --------------------------------------------------------------
# Example 10 Test (Table 11)
# --------------------------------------------------------------
if __name__ == "__main__":
    # Define fuzzy assets from Example 10
    assets_fuzzy = [
        [2, 3, 4, 7],  # Asset 11
        [2, 5, 6, 7],  # Asset 12
        [3, 4, 4.6, 5],  # Asset 13
        [1, 4.6, 6.2, 6.6],  # Asset 14
        [0, 3, 6, 9],  # Asset 15
        [4, 4.8, 5.2, 6]  # Asset 16
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