from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class LstmPred(tf.keras.Model):
    def __init__(self, input_shape, output_shape, net_layers, activation, optimizer, loss, history_window, pred_window):
        super().__init__()
        layers = []

        layers.append(keras.layers.LSTM(net_layers[0], activation=activation, input_shape=input_shape, return_sequences=True))
        layers.append(keras.layers.Dropout(0.1))
        for i in range(1, len(net_layers)-1):
            layers.append(keras.layers.LSTM(net_layers[i], activation=activation, return_sequences=True))
            layers.append(keras.layers.Dropout(0.1))

        layers.append(keras.layers.LSTM(net_layers[-1], activation=activation))
        layers.append(keras.layers.Dropout(0.1))
        layers.append(keras.layers.Dense(output_shape))

        self.net = tf.keras.Sequential(layers)
        
        self.history_window = history_window
        self.pred_window = pred_window

        self.compile(optimizer=optimizer, loss=loss)

        self.threshold = None
        self.history = None

    def call(self, inputs):
        x = self.net(inputs)
        return x
    
    @staticmethod
    def _prepare_data_lstm(data, history_steps, pred_steps):
        output_history = []
        output_targets = []
        for i in range(history_steps, data.shape[0]-pred_steps+1, 1):
            output_history.append(data[i-history_steps:i])
            output_targets.append(data[i].flatten())
        return np.stack(output_history), np.stack(output_targets)

    def train(self, train_data, epochs=100, batch_size=64):
        data, targets = self._prepare_data_lstm(train_data, self.history_window, self.pred_window)
        hist = self.fit(
            data,
            targets,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=0)
            ]
            )
        self.history = {'loss': hist.history['loss'], 'val_loss': hist.history['val_loss']}
        self.calculate_parameters(train_data)

    def plot_train_history(self):  
        plt.plot(self.history["loss"], label="Training Loss")
        plt.plot(self.history["val_loss"], label="Validation Loss")
        plt.legend()
        plt.show()

    def calculate_parameters(self, data):
        #self.threshold = np.max(self.reconstruction_error(train_data))
        res = self.reconstruction_error(data)
        self.threshold = np.mean(res) + np.std(res)
        
    def reconstruction_error(self, data):
        data, targets = self._prepare_data_lstm(data, self.history_window, self.pred_window)
        residual = tf.keras.losses.mae(self.predict(data), targets)
        residual = np.array(self.history_window * [0] + list(residual))
        return residual

    def detect_anomalies(self, test_data):
        residual = self.reconstruction_error(test_data)
        return np.array((residual > self.threshold)).astype(int)