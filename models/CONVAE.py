from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class ConvAE(tf.keras.Model):
    def __init__(self, input_shape, net_layers, padding, activation, optimizer, loss, window_size):
        super().__init__()
        encoder_layers = []
        decoder_layers = []

        encoder_layers.append(layers.Input(shape=input_shape))
        for i in range(0, len(net_layers)):
            encoder_layers.append(layers.Conv1D(filters=net_layers[i][0], kernel_size=net_layers[i][1], padding=padding, strides=net_layers[i][2], activation=activation))

        for i in range(len(net_layers)-2, -1, -1):
            decoder_layers.append(layers.Conv1DTranspose(filters=net_layers[i][0], kernel_size=net_layers[i][1], padding=padding, strides=net_layers[i][2], activation=activation))
        decoder_layers.append(layers.Conv1DTranspose(filters=input_shape[1], kernel_size=net_layers[-1][1], padding=padding))
        
        self.encoder = tf.keras.Sequential(encoder_layers)
        self.decoder = tf.keras.Sequential(decoder_layers)
        
        self.window_size= window_size

        self.compile(optimizer=optimizer, loss=loss)

        self.threshold = None
        self.history = None

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
    
    @staticmethod
    def _prepare_data_window_ae(data, time_steps):
        output = []
        for i in range(len(data) - time_steps + 1):
            output.append(data[i : (i + time_steps)])
        return np.stack(output)

    def train(self, train_data, epochs=100, batch_size=64):
        data = self._prepare_data_window_ae(train_data, self.window_size)
        hist = self.fit(
            data,
            data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=0)
            ]
            )
        self.history = {'loss': hist.history['loss'], 'val_loss': hist.history['val_loss']}
        self.set_threshold(train_data)

    def plot_train_history(self):
        plt.plot(self.history["loss"], label="Training Loss")
        plt.plot(self.history["val_loss"], label="Validation Loss")
        plt.legend()
        plt.show()

    def set_threshold(self, train_data):
        #self.threshold = np.max(self.reconstruction_error(train_data))
        res = self.reconstruction_error(train_data)
        self.threshold = np.mean(res) + np.std(res)
        
    def reconstruction_error(self, data):
        data = self._prepare_data_window_ae(data, self.window_size)
        pred = keras.losses.mae(self.predict(data), data)
        residual = np.zeros((pred.shape[0]+(self.window_size-1)))
        for i in range(0, pred.shape[0], 1):
            for j in range(self.window_size):
                if residual[i+j] < pred[i][j]:
                    residual[i+j] = pred[i][j]
        return  residual

    def detect_anomalies(self, test_data):
        residual = self.reconstruction_error(test_data)
        return np.array((residual > self.threshold)).astype(int)
