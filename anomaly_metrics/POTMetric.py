import numpy as np
from scipy.stats import genextreme
from math import log
from scipy.optimize import minimize

class POTMetric:
    def __init__(self, risk:float=1e-4, init_level:float=0.98, num_candidates:int=10, epsilon:float=1e-8):
        self.risk = risk
        self.init_level = init_level
        self.num_candidates = num_candidates
        self.epsilon = epsilon   
        self.threshold = 0
        self.quantile = 0
        
    def calculate_params(self, train_residual):
        t = np.sort(train_residual)[int(self.init_level * train_residual.size)]
        peaks = train_residual[train_residual > t] - t

        # Grimshaw
        gamma, sigma = self.grimshaw(peaks=peaks, 
                                threshold=t, 
                                num_candidates=self.num_candidates, 
                                epsilon=self.epsilon
                                )

        # Calculate Threshold
        r = train_residual.size * self.risk / peaks.size
        if gamma != 0:
            z = t + (sigma / gamma) * (pow(r, -gamma) - 1)
        else: 
            z = t - sigma * log(r)
        
        self.threshold = t
        self.quantile = z
        
    def detect_anomalies(self, residual):
        return np.array(residual > self.quantile).astype(int)
    
    def grimshaw(self, peaks:np.array, threshold:float, num_candidates:int=10, epsilon:float=1e-8):
        ''' The Grimshaw's Trick Method

        The trick of thr Grimshaw's procedure is to reduce the two variables 
        optimization problem to a signle variable equation. 

        Args:
            peaks: peak nodes from original dataset. 
            threshold: init threshold
            num_candidates: the maximum number of nodes we choose as candidates
            epsilon: numerical parameter to perform

        Returns:
            gamma: estimate
            sigma: estimate
        '''
        min = peaks.min()
        max = peaks.max()
        mean = peaks.mean()

        if abs(-1 / max) < 2 * epsilon:
            epsilon = abs(-1 / max) / num_candidates

        a = -1 / max + epsilon
        b = 2 * (mean - min) / (mean * min)
        c = 2 * (mean - min) / (min ** 2)

        candidate_gamma = self.solve(function=lambda t: self.function(peaks, threshold), 
                                dev_function=lambda t: self.dev_function(peaks, threshold), 
                                bounds=(a + epsilon, -epsilon), 
                                num_candidates=num_candidates
                                )
        candidate_sigma = self.solve(function=lambda t: self.function(peaks, threshold), 
                                dev_function=lambda t: self.dev_function(peaks, threshold), 
                                bounds=(b, c), 
                                num_candidates=num_candidates
                                )
        candidates = np.concatenate([candidate_gamma, candidate_sigma])

        gamma_best = 0
        sigma_best = mean
        log_likelihood_best = self.cal_log_likelihood(peaks, gamma_best, sigma_best)

        for candidate in candidates:
            gamma = np.log(1 + candidate * peaks).mean()
            sigma = gamma / candidate
            log_likelihood = self.cal_log_likelihood(peaks, gamma, sigma)
            if log_likelihood > log_likelihood_best:
                gamma_best = gamma
                sigma_best = sigma
                log_likelihood_best = log_likelihood

        return gamma_best, sigma_best

    @staticmethod
    def function(x, threshold):
        s = 1 + threshold * x
        u = 1 + np.log(s).mean()
        v = np.mean(1 / s)
        return u * v - 1

    @staticmethod
    def dev_function(x, threshold):
        s = 1 + threshold * x
        u = 1 + np.log(s).mean()
        v = np.mean(1 / s)
        dev_u = (1 / threshold) * (1 - v)
        dev_v = (1 / threshold) * (-v + np.mean(1 / s ** 2))
        return u * dev_v + v * dev_u

    @staticmethod
    def obj_function(x, function, dev_function):
        m = 0
        n = np.zeros(x.shape)
        for index, item in enumerate(x):
            y = function(item)
            m = m + y ** 2
            n[index] = 2 * y * dev_function(item)
        return m, n

    def solve(self, function, dev_function, bounds, num_candidates):
        step = (bounds[1] - bounds[0]) / (num_candidates + 1)
        x0 = np.arange(bounds[0] + step, bounds[1], step)
        optimization = minimize(lambda x: self.obj_function(x, function, dev_function), 
                                x0, 
                                method='L-BFGS-B', 
                                jac=True, 
                                bounds=[bounds]*len(x0)
                                )
        x = np.round(optimization.x, decimals=5)
        return np.unique(x)

    @staticmethod
    def cal_log_likelihood(peaks, gamma, sigma):
        if gamma != 0:
            tau = gamma/sigma
            log_likelihood = -peaks.size * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * peaks)).sum()
        else: 
            log_likelihood = peaks.size * (1 + log(peaks.mean()))
        return log_likelihood