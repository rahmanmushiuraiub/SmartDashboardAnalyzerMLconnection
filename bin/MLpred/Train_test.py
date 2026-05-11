import numpy as np


class TrainTest:
    """Basic train/test split and regression evaluation without external ML dependencies."""

    def __init__(self, model, X, y, test_size=0.2, random_state=42):
        self.model = model
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self):
        rng = np.random.default_rng(self.random_state)
        indices = np.arange(len(self.y))
        rng.shuffle(indices)
        test_count = max(1, int(len(indices) * self.test_size))
        test_idx = indices[:test_count]
        train_idx = indices[test_count:]
        return self.X[train_idx], self.X[test_idx], self.y[train_idx], self.y[test_idx]

    def train(self):
        X_train, X_test, y_train, y_test = self.split_data()
        self.model.fit(X_train, y_train)
        return X_test, y_test

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = float(np.mean((y_test - predictions) ** 2))
        rmse = float(np.sqrt(mse))
        ss_res = float(np.sum((y_test - predictions) ** 2))
        ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        return {"mse": mse, "rmse": rmse, "r2": r2}
