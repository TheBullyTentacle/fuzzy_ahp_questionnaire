import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import kstest
from sklearn.decomposition import FastICA
from sklearn.preprocessing import MinMaxScaler
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


class TurningPointModel:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.df = None
        self.scaler = MinMaxScaler()
        self.ica = FastICA(n_components=2)
        self._setup_fuzzy_system()

    def _setup_fuzzy_system(self):
        self.pe_ratio = ctrl.Antecedent(np.arange(0, 1, 0.01), 'pe_ratio')
        self.book_value = ctrl.Antecedent(np.arange(0, 1, 0.01), 'book_value')
        self.neutral_range = ctrl.Consequent(np.arange(0, 1, 0.01), 'neutral_range')

        names = ['low', 'medium', 'high']
        self.pe_ratio.automf(3, names=names)
        self.book_value.automf(3, names=names)
        self.neutral_range.automf(3, names=names)

        rules = [
            ctrl.Rule(self.pe_ratio['medium'] & self.book_value['medium'], self.neutral_range['medium']),
            ctrl.Rule(self.pe_ratio['low'] | self.book_value['high'], self.neutral_range['low']),
            ctrl.Rule(self.pe_ratio['high'] | self.book_value['low'], self.neutral_range['high'])
        ]

        self.neutral_ctrl = ctrl.ControlSystem(rules)
        self.neutral_sim = ctrl.ControlSystemSimulation(self.neutral_ctrl)

    def fetch_data(self):
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        data['P/E'] = np.random.uniform(10, 30, len(data))  # Synthetic data
        data['Book Value'] = np.random.uniform(50, 200, len(data))
        self.df = data.dropna()
        return self.df

    def preprocess(self):
        cols_to_scale = ['Close', 'Volume', 'P/E', 'Book Value']
        self.df[cols_to_scale] = self.scaler.fit_transform(self.df[cols_to_scale])
        # Ensure float type after scaling
        self.df[cols_to_scale] = self.df[cols_to_scale].astype(float)
        return self.df

    def extract_features(self):
        features = self.ica.fit_transform(self.df[['Close', 'Volume', 'P/E']].fillna(0))
        self.df['Feature1'], self.df['Feature2'] = features[:, 0], features[:, 1]
        return self.df

    def calculate_neutral_scores(self):
        neutral_scores = []
        # Use itertuples for safe scalar access
        for row in self.df.itertuples():
            try:
                # Explicit type conversion to float
                self.neutral_sim.input['pe_ratio'] = float(row._4)  # P/E is 4th column
                self.neutral_sim.input['book_value'] = float(row._5)  # Book Value is 5th
                self.neutral_sim.compute()
                neutral_scores.append(self.neutral_sim.output['neutral_range'])
            except Exception as e:
                print(f"Error processing row {row.Index}: {str(e)}")
                neutral_scores.append(0.5)  # Default neutral score

        self.df['Neutral_Score'] = neutral_scores
        return self.df

    def detect_turning_points(self):
        self.df['Positive_Turning'] = (self.df['P/E'] < 0.2) & (self.df['Feature1'] > 0.5)
        self.df['Negative_Turning'] = (self.df['P/E'] > 0.8) & (self.df['Feature1'] < -0.5)
        return self.df

    def calculate_metrics(self):
        self.df['Price_Change'] = self.df['Close'].pct_change().fillna(0)
        self.df['Accelerating_Speed'] = self.df['Price_Change'] * self.df['Feature1']
        self.df['Risk_Coefficient'] = abs(self.df['Feature2']) * self.df['Accelerating_Speed']
        return self.df

    def classify_ranges(self):
        conditions = [
            (self.df['Positive_Turning'] & (self.df['Risk_Coefficient'] < 0.1)),
            (self.df['Negative_Turning'] & (self.df['Risk_Coefficient'] > 0.3)),
            (self.df['Neutral_Score'].between(0.4, 0.6, inclusive='both'))
        ]
        choices = ['Positive', 'Negative', 'Neutral']
        self.df['Classification'] = np.select(conditions, choices, default='Unclassified')
        return self.df

    def analyze(self):
        self.fetch_data()
        self.preprocess()
        self.extract_features()
        self.calculate_neutral_scores()
        self.detect_turning_points()
        self.calculate_metrics()
        self.classify_ranges()
        return self.df

    def plot_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.df['Close'], label='Normalized Price')

        # Boolean indexing with .loc
        positive_points = self.df.loc[self.df['Positive_Turning']]
        negative_points = self.df.loc[self.df['Negative_Turning']]

        plt.scatter(positive_points.index, positive_points['Close'],
                    color='green', marker='^', s=100, label='Positive Turning')
        plt.scatter(negative_points.index, negative_points['Close'],
                    color='red', marker='v', s=100, label='Negative Turning')

        plt.title(f'Turning Point Analysis for {self.ticker}')
        plt.legend()
        plt.show()


# Example test
if __name__ == "__main__":
    model = TurningPointModel('AAPL', '2020-01-01', '2023-01-01')
    results = model.analyze()

    print("\nSample Classifications:")
    print(results[['Close', 'Classification', 'Positive_Turning', 'Negative_Turning']].tail(10))

    #model.plot_results()