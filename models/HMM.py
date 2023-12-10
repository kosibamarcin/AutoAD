import pandas as pd
import numpy as np
from hmmlearn import hmm
from tqdm import tqdm

class HMM():
    def __init__(self, n_states, covar_type, back_step, *args, **kwargs):
        self.threshold = 0
        self.n_states = n_states
        self.covar_type = covar_type
        self.back_step = back_step
        self.test_abnormality = None
        self.train_abnormality = None
        self.model = None

    def train(self, train_data: np.array, fit_num: int, *args, **kwargs):
        X = train_data[:int(0.85*train_data.shape[0])].reshape(-1, train_data.shape[1])
        val_X = train_data[int(0.85*train_data.shape[0]):].reshape(-1, train_data.shape[1])
        lengths = [len(X)]

        best_score = None
        best_model = None

        for i in range(fit_num):
            model = hmm.GaussianHMM(n_components=self.n_states, covariance_type=self.covar_type, random_state=i)
            try:
                model.fit(X, lengths)
            except:
                # sometimes the algorithm is not able to converge
                print('Failure to fit')
                continue
            score = model.score(val_X)
            print(f'Model #{i}\tScore: {score}')
            if best_score is None or (score > best_score and score > 0):
                best_model = model
                best_score = score
        
        self.model = best_model
        log_prob = self.calculate_log_prob(train_data)
        self.train_abnormality = self.calculate_abnormality(log_prob)
        self.threshold = self.calculate_threshold(self.train_abnormality)

    def detect_anomalies(self, test_data: np.array, *args, **kwargs) -> np.array:
        log_prob = self.calculate_log_prob(test_data)
        self.test_abnormality = self.calculate_abnormality(log_prob)
        return np.array(self.test_abnormality > self.threshold).astype(int)

    def calculate_log_prob(self, data: np.array, *args, **kwargs) -> np.array:
        log_prob = np.array([self.model.score(data[0:i + 1].reshape(-1, 1)) if i < self.back_step
                            else self.model.score(data[i - self.back_step:i + 1].reshape(-1, 1))
                            for i in range(len(data))])
        return log_prob
    
    def reconstruction_error(self, data):
        log_prob = self.calculate_log_prob(data)
        return self.calculate_abnormality(log_prob)
    
    def anomaly_score(self, data):
        log_prob = self.calculate_log_prob(data)
        return self.calculate_abnormality(log_prob)

    @staticmethod
    def calculate_abnormality(log_prob: np.array) -> np.array:
        return -np.append(log_prob[0], np.diff(log_prob))

    @staticmethod
    def calculate_threshold(abnormality: np.array) -> np.array:
        ratio = 0.005  # Percentage of judgments as abnormal
        return np.sort(abnormality)[int((1 - ratio) * len(abnormality))]

# class HMM:
#     def __init__(self, n_states, n_fits, time_step, batch_size=120):
#         self.n_states = n_states
#         self.best_model = None
#         self.n_fits = n_fits
#         self.batch_size = batch_size
#         self.time_step = time_step

#     def train(self, train_data):
#         size_dataset, numvars = train_data.shape
#         n_batches = np.floor(size_dataset/self.batch_size).astype(int)
#         lenghts_vec = [self.batch_size] * int(n_batches)
#         truncate_here = n_batches * self.batch_size
#         train_n_batches = int(n_batches * 0.8)
#         truncate_train = train_n_batches * self.batch_size
#         truncate_test = truncate_here

#         X = train_data.values.reshape(-1, train_data.shape[1])
#         lengths = [len(train_data)]

#         best_score = None
#         model = None

#         for i in range(self.n_fits):
#             model = hmm.GaussianHMM(n_components=self.n_states, covariance_type='full', random_state=i)
#             try:
#                 model.fit(X, lengths)
#             except:
#                 # sometimes the algorithm is not able to converge
#                 print('Failure to fit')
#                 continue
#             score = model.score(X)
#             print(f'Model #{i}\tScore: {score}')
#             if best_score is None or (score > best_score and score > 0):
#                 self.best_model = model
#                 best_score = score
#         log_prob = self.calculate_log_prob(train_data)
#         self.train_abnormality = self.calculate_abnormality(log_prob)
#         self.threshold = self.calculate_threshold(self.train_abnormality)

#     def calculate_log_prob(self, data: pd.DataFrame, *args, **kwargs) -> np.array:
#         log_prob = np.array([self.best_model.score(data.values[0:i + 1].reshape(-1, 1)) if i < self.time_step
#                             else self.best_model.score(data.values[i - self.time_step:i + 1].reshape(-1, 1))
#                             for i in range(len(data.values))])
#         return log_prob

#     @staticmethod
#     def calculate_abnormality(log_prob: np.array) -> np.array:
#         return -np.append(log_prob[0], np.diff(log_prob))

#     @staticmethod
#     def calculate_threshold(abnormality: np.array) -> np.array:
#         ratio = 0.005  # Percentage of judgments as abnormal
#         return np.sort(abnormality)[int((1 - ratio) * len(abnormality))]

#     def detect_anomalies(self, test_data):
#         log_prob = self.calculate_log_prob(test_data)
#         self.test_abnormality = self.calculate_abnormality(log_prob)
#         return np.array(self.test_abnormality > self.threshold).astype(int)
