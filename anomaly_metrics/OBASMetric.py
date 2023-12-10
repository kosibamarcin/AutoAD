import numpy as np

class OBASMetric:
    def __init__(self):
        self.min_th = 0
        self.max_th = 0
        self.width = 0

    def calculate_params(self, train_residual):
        self.min_th = np.min(train_residual)
        self.max_th = np.max(train_residual)
        self.width = self.max_th - self.min_th
        
    def detect_anomalies(self, residual):
        return np.array(np.max(np.column_stack((np.abs(residual - self.min_th), np.abs(residual - self.max_th))), axis=1)/self.width > 1).astype(int)