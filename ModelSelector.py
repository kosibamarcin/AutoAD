import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from ModelGenerator import ModelGenerator
from models.CONVAE import ConvAE
from models.DENSEAE import DenseAE
from models.HMM import HMM
from models.LSTM import LstmPred
from models.LSTMAE import LstmAE

class ModelSelector:
    def __init__(self, train_data, test_data, test_labels, models_templates='config.pkl'):
        self.models = []
        self.models_templates = models_templates
        self.anomaly_generator = None
        self.train_data = train_data
        self.test_data = test_data
        self.test_labels = test_labels

    def populate(self):
        print('MODEL GENERATION')
        generator = ModelGenerator(self.models_templates, self.train_data.shape[1])
        self.models = generator.get_models()
        print('MODEL GENERATION COMPLETED')

    def train_models(self):
        print('MODEL TRAINING')
        for model in self.models:
            if isinstance(model, HMM):
                model.train(self.train_data, 5)
            else:
                model.train(self.train_data)
        print('MODEL TRAINING COMPLETED')

    def score_model(self, model):
        score = 0
        anomalies = model.detect_anomalies(self.test_data)
        
        TN, FP, FN, TP = confusion_matrix(self.test_labels, anomalies).ravel()
        print(FP, FN, TP, TN)
        
        FAR = FP/(FP+TN)
        MAR = FN/(TP+FN)
        print('FAR: ', FAR, "MAR:", MAR)
        if isinstance(model, HMM):
            anomalies = model.detect_anomalies(self.test_data)
            score = 2 * accuracy_score(anomalies, self.test_labels) + 2 * f1_score(anomalies, self.test_labels) + 2 * (1-FAR) + 2 * (1-MAR)
            print(accuracy_score(anomalies, self.test_labels), f1_score(anomalies, self.test_labels))
        else:
            reconstruction_error = np.mean(model.reconstruction_error(self.train_data))
            anomalies = model.detect_anomalies(self.test_data)
            score = 2 * (1-reconstruction_error) + 1 * accuracy_score(anomalies, self.test_labels) + 2 * f1_score(anomalies, self.test_labels) + 2 * (1-FAR) + 2 * (1-MAR)
            print(1-reconstruction_error, accuracy_score(anomalies, self.test_labels), f1_score(anomalies, self.test_labels))
        print(type(model), score)
        return score

    def select_best_model(self):
        print('MODEL SCORING')
        models_scores = []
        for model in self.models:
            models_scores.append([model, self.score_model(model)])
        return sorted(models_scores, key=lambda x: x[1], reverse=True)[0][0]

    def evaluate(self):
        print('STARTING MODEL SELECTION PROCESS')
        self.populate()
        self.train_models()
        best_model = self.select_best_model()
        print('BEST MODEL SELECTED: ', type(best_model))
        return best_model
    