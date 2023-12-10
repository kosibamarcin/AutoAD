import numpy as np

class ThresholdMetric:
    def __init__(self):
        self.max_loss = 0
        self.mean_loss = 0
        
    def calculate_params(self, train_residual):
        self.max_loss = np.max(train_residual)
        self.mean_loss = np.mean(train_residual) + np.std(train_residual)
        
    def detect_anomalies(self, residual):
        return np.array(residual > self.mean_loss).astype(int)