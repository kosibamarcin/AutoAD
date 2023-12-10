import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from anomaly_metrics.OBASMetric import OBASMetric
from anomaly_metrics.ThresholdMetric import ThresholdMetric
from anomaly_metrics.POTMetric import POTMetric


class AnomalyDetector:
    def __init__(self):
        self.obas = OBASMetric()
        self.threshold = ThresholdMetric()
        self.pot = POTMetric()
        
        self.best_metric = None
        
    def train(self, train_residual, anomaly_residual, anomaly_labels):
        self.obas.calculate_params(train_residual)
        self.threshold.calculate_params(train_residual)
        self.pot.calculate_params(train_residual)
        
        self.select_best_metric(anomaly_residual, anomaly_labels)
        
    def select_best_metric(self, anomaly_residual, anomaly_labels):
        obas_anomalies = self.obas.detect_anomalies(anomaly_residual)
        obas_acc, obas_f1 = accuracy_score(obas_anomalies, anomaly_labels), f1_score(obas_anomalies, anomaly_labels)
        
        threshold_anomalies = self.threshold.detect_anomalies(anomaly_residual)
        threshold_acc, threshold_f1 = accuracy_score(threshold_anomalies, anomaly_labels), f1_score(threshold_anomalies, anomaly_labels)
        
        pot_anomalies = self.pot.detect_anomalies(anomaly_residual)
        pot_acc, pot_f1 = accuracy_score(pot_anomalies, anomaly_labels), f1_score(pot_anomalies, anomaly_labels)
        
        detectors_scores = [
            [self.obas, 2 * obas_acc + 3 * obas_f1],
            [self.threshold, 2 * threshold_acc + 3 * threshold_f1],
            [self.pot, 2 * pot_acc + 3 * pot_f1]
        ]
        self.best_metric = sorted(detectors_scores, key=lambda x: x[1], reverse=True)[0][0]

    def detect_anomalies(self, residual):
        return self.best_metric.detect_anomalies(residual)
 