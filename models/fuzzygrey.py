import numpy as np
from scipy.stats import dirichlet


# ================================================
# Helper Functions
# ================================================
def generate_ideal_sequence(a, b, c, l, L, seed=None):
    np.random.seed(seed)
    sequence = []
    threshold = (l * (b - a)) / (c + b * (l - 1) - a * l)
    for _ in range(L):
        r = np.random.uniform(0, 1)
        if r <= threshold:
            term = r * (c + b * (l - 1) - a * l) * (b - a) ** (1 / l)
            value = a + (term / l) ** (l / (l + 1))
        else:
            term = (1 - r) * (c + b * (l - 1) - a * l) * (c - b) ** (1 / l)
            value = c - (term) ** (l / (l + 1))
        sequence.append(value)
    return np.array(sequence)


def penalize(x, y, l_bounds, u_bounds, K, penalty_weight=1e3):
    penalty = 0.0
    selected_assets = np.sum(y)
    if selected_assets < K:
        penalty += (K - selected_assets)
    total_investment = np.sum(x)
    penalty += abs(total_investment - 1)
    for i in range(len(x)):
        if y[i] == 1:
            if x[i] < l_bounds[i]:
                penalty += (l_bounds[i] - x[i])
            if x[i] > u_bounds[i]:
                penalty += (x[i] - u_bounds[i])
    return penalty_weight * penalty


# ================================================
# Core Algorithm Implementation
# ================================================
class FuzzyGreyPortfolioOptimizer:
    def __init__(self, params):
        self.params = params

    def initialize_population(self):
        POP = self.params['POP']
        N = self.params['N']
        K = self.params['K']
        pop_x = []
        pop_y = []
        for _ in range(POP):
            # Ensure at least K assets are selected
            y = np.zeros(N, dtype=int)
            selected = np.random.choice(N, size=K, replace=False)
            y[selected] = 1
            # Generate weights using Dirichlet distribution
            x = np.zeros(N)
            x[selected] = dirichlet.rvs(alpha=np.ones(K), size=1).flatten()
            pop_x.append(x)
            pop_y.append(y)
        return np.array(pop_x), np.array(pop_y)

    def calculate_CLSIM(self, portfolio_sequence, ideal_sequence):
        # Zero-starting point transformation
        portfolio_zero = portfolio_sequence - portfolio_sequence[0]
        ideal_zero = ideal_sequence - ideal_sequence[0]

        # Similitude degree (ε)
        s_p = np.sum(portfolio_zero[1:-1]) + 0.5 * portfolio_zero[-1]
        s_i = np.sum(ideal_zero[1:-1]) + 0.5 * ideal_zero[-1]
        epsilon = 1 / (1 + abs(s_p - s_i))

        # Closeness degree (ρ)
        S_p = np.sum(portfolio_sequence[1:-1]) + 0.5 * portfolio_sequence[-1]
        S_i = np.sum(ideal_sequence[1:-1]) + 0.5 * ideal_sequence[-1]
        rho = 1 / (1 + abs(S_p - S_i))

        return self.params['w1'] * rho + self.params['w2'] * epsilon

    def evaluate_fitness(self, pop_x, pop_y, ideal_sequence, his_returns):
        fitness = []
        for i in range(len(pop_x)):
            x = pop_x[i]
            y = pop_y[i]
            # Portfolio return sequence
            port_returns = np.sum(his_returns * x, axis=1)
            # Objectives
            CLSIM = self.calculate_CLSIM(port_returns, ideal_sequence)
            expected_return = np.mean(port_returns)  # Simplified for example
            # Penalty
            penalty = penalize(x, y, self.params['l_bounds'],
                               self.params['u_bounds'], self.params['K'])
            fitness.append((expected_return - penalty, CLSIM - penalty))
        return np.array(fitness)

    def roulette_wheel_selection(self, fitness):
        # Convert to maximization scores
        scores = np.sum(fitness, axis=1)
        scores -= np.min(scores)  # Avoid negative probabilities
        probs = scores / np.sum(scores)
        return np.random.choice(len(fitness), size=len(fitness), p=probs)

    def SBX_crossover(self, parent1, parent2, m=2):
        child1, child2 = np.zeros_like(parent1), np.zeros_like(parent2)
        for i in range(len(parent1)):
            if np.random.rand() < self.params['p_c']:
                if np.random.rand() < 0.5:
                    beta = (2 * np.random.rand()) ** (1 / (m + 1))
                else:
                    beta = (1 / (2 * (1 - np.random.rand()))) ** (1 / (m + 1))
                child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
        return child1, child2

    def non_uniform_mutation(self, x, y, gen, max_gen, F=5):
        for i in range(len(x)):
            if y[i] == 1 and np.random.rand() < self.params['p_m']:
                delta = (1 - np.random.rand() ** ((1 - gen / max_gen) ** F))
                if np.random.rand() < 0.5:
                    x[i] += delta * (self.params['u_bounds'][i] - x[i])
                else:
                    x[i] -= delta * (x[i] - self.params['l_bounds'][i])
        # Repair to ensure sum(x) = 1
        x /= np.sum(x)
        return x

    def optimize(self):
        # Initialize population and ideal sequence
        pop_x, pop_y = self.initialize_population()
        ideal_sequence = generate_ideal_sequence(
            self.params['I_a'], self.params['I_b'], self.params['I_c'],
            self.params['l'], self.params['L'], seed=self.params['seed']
        )

        # Track global best solution
        global_best_x = None
        global_best_y = None
        global_best_score = -np.inf

        for gen in range(self.params['MG']):
            # Evaluate fitness
            fitness = self.evaluate_fitness(pop_x, pop_y, ideal_sequence,
                                            self.params['historical_returns'])

            current_scores = np.sum(fitness, axis=1)

            # Update global best
            current_best_idx = np.argmax(current_scores)
            current_best_score = current_scores[current_best_idx]
            if current_best_score > global_best_score:
                global_best_score = current_best_score
                global_best_x = pop_x[current_best_idx]
                global_best_y = pop_y[current_best_idx]

            # Elitism (top 10%)
            elite_size = max(1, int(0.1 * self.params['POP']))
            elite_indices = np.argsort(np.sum(fitness, axis=1))[-elite_size:]
            elite_x = pop_x[elite_indices]
            elite_y = pop_y[elite_indices]

            # Selection
            selected_indices = self.roulette_wheel_selection(fitness)
            parents_x = pop_x[selected_indices]
            parents_y = pop_y[selected_indices]

            # Crossover and Mutation
            offspring_x, offspring_y = [], []
            for i in range(0, self.params['POP'], 2):
                if i + 1 >= len(parents_x):
                    break  # Handle odd population sizes
                p1, p2 = parents_x[i], parents_x[i + 1]
                y1, y2 = parents_y[i], parents_y[i + 1]
                c1, c2 = self.SBX_crossover(p1, p2)
                cy = np.logical_or(y1, y2).astype(int)  # Merge selected assets
                # Mutation
                c1 = self.non_uniform_mutation(c1, cy, gen, self.params['MG'])
                c2 = self.non_uniform_mutation(c2, cy, gen, self.params['MG'])
                offspring_x.extend([c1, c2])
                offspring_y.extend([cy, cy])

            # Combine populations
            combined_x = np.vstack([elite_x, np.array(offspring_x)])
            combined_y = np.vstack([elite_y, np.array(offspring_y)])
            # Select new population
            combined_fitness = self.evaluate_fitness(combined_x, combined_y,
                                                     ideal_sequence,
                                                     self.params['historical_returns'])
            top_indices = np.argsort(np.sum(combined_fitness, axis=1))[-self.params['POP']:]
            pop_x = combined_x[top_indices]
            pop_y = combined_y[top_indices]

        return global_best_x, global_best_y


# ================================================
# Example Usage (Adjusted for Correctness)
# ================================================
if __name__ == "__main__":
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