import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from ModelGenerator import *
from ModelSelector import *
from AnomalyDetector import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class AutoAD:
    def __init__(self, train_data, test_data=None, test_labels=None):
        self.scaler = MinMaxScaler().fit(train_data.iloc[:int(0.85*train_data.shape[0]), :])
        self.train_data = self.scaler.transform(train_data.iloc[:int(0.85*train_data.shape[0]), :])
        if test_data is None:
            self.test_data, self.test_labels = self.generate_anomalies(train_data.values[int(0.85*train_data.shape[0]):, :])
            self.test_data = self.scaler.transform(self.test_data)
        else:
            self.test_data, self.test_labels = self.scaler.transform(test_data), test_labels
        self.model_selector = ModelSelector(self.train_data, self.test_data, self.test_labels)
        self.anomaly_detector = AnomalyDetector()
        self.model = None
         
    @staticmethod
    def generate_anomalies(data):
        data = data.copy()
        point_anomalies_count = int(0.5 * data.shape[0])
        point_anomalies = random.sample(range(data.shape[0]), point_anomalies_count)
        anomalies = [0] * data.shape[0]
        for anomaly in point_anomalies:
            col = random.choice(range(data.shape[1]))
            mu = np.mean(data[:, col])
            sd = np.std(data[:, col])
            data[anomaly, col] += 0.6*np.random.normal(mu, sd, size=1)
            anomalies[anomaly] = 1
        test_data = data
        return test_data, np.array(anomalies)

    def select_model(self):
        self.model = self.model_selector.evaluate()
        residual = self.model.reconstruction_error(self.train_data)
        residual_anomalies = self.model.reconstruction_error(self.test_data)
        self.anomaly_detector.train(residual, residual_anomalies, self.test_labels)

    def detect_anomalies(self, data):
        residual = self.model.reconstruction_error(self.scaler.transform(data))
        return self.anomaly_detector.detect_anomalies(residual)

    