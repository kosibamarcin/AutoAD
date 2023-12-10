import numpy as np
import pandas as pd
from models import *
from models.HMM import HMM
from models.DENSEAE import DenseAE
from models.CONVAE import ConvAE
from models.LSTMAE import LstmAE
from models.LSTM import LstmPred
import pickle

class ModelGenerator:
    def __init__(self, configuration_filepath, num_of_features):
        self.configurations = self.load_configuration(configuration_filepath)
        self.num_of_features = num_of_features
        self.models = []
        
    @staticmethod
    def load_configuration(filepath):
        conf = []
        with open(filepath, 'rb') as f:
            conf = pickle.load(f)
        return conf

    def generate_hmm(self, configuration):
        model = HMM(n_states=configuration[0], covar_type=configuration[1], back_step=configuration[2])
        self.models.append(model)

    def generate_denseae(self, configuration):
        model = DenseAE(self.num_of_features, configuration[0], configuration[1], configuration[2], configuration[3], configuration[4], configuration[5])
        self.models.append(model)

    def generate_convae(self, configuration):
        model = ConvAE([configuration[5], self.num_of_features], configuration[0], padding=configuration[1], activation=configuration[2], optimizer=configuration[3], loss=configuration[4], window_size=configuration[5])
        self.models.append(model)

    def generate_lstmae(self, configuration):
        model = LstmAE([configuration[4], self.num_of_features], configuration[0], configuration[1], configuration[2], configuration[3], window_size=configuration[4])
        self.models.append(model)

    def generate_lstm(self, configuration):
        model = LstmPred([configuration[4], self.num_of_features], configuration[5], configuration[0], configuration[1], configuration[2], configuration[3], history_window=configuration[4], pred_window=configuration[5])
        self.models.append(model)
    
    def get_models(self):
        for conf in self.configurations:
            if conf[0] == "HMM":
                self.generate_hmm(conf[1])
            elif conf[0] == "DENSEAE":
                self.generate_denseae(conf[1])
            elif conf[0] == "CONVAE":
                self.generate_convae(conf[1])
            elif conf[0] == "LSTMAE":
                self.generate_lstmae(conf[1])
            else:
                self.generate_lstm(conf[1])
        return self.models