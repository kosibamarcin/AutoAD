from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class DenseAE(tf.keras.Model):
    def __init__(self, input_shape, net_layers, internal_act, output_act, dropout_rate, optimizer, loss):
        super().__init__()
        encoder_layers = []
        decoder_layers = []

        encoder_layers.append(layers.Input(shape=input_shape))
        for i in range(len(net_layers)-1):
            encoder_layers.append(layers.Dense(net_layers[i], activation=internal_act))
            encoder_layers.append(layers.Dropout(dropout_rate))
        encoder_layers.append(layers.Dense(net_layers[-1], activation=internal_act))

        for i in range(len(net_layers)-2, 0, -1):
            decoder_layers.append(layers.Dense(net_layers[i], activation=internal_act))
            decoder_layers.append(layers.Dropout(dropout_rate))
        decoder_layers.append(layers.Dense(input_shape, activation=output_act))
        
        self.encoder = tf.keras.Sequential(encoder_layers)
        self.decoder = tf.keras.Sequential(decoder_layers)

        self.compile(optimizer=optimizer, loss=loss)

        self.threshold = None
        self.history = None

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    def train(self, train_data, epochs=100, batch_size=64):
        hist = self.fit(
            train_data,
            train_data,
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
        return tf.keras.losses.mae(self.predict(data), data)

    def detect_anomalies(self, test_data):
        residual = self.reconstruction_error(test_data)
        return np.array(residual > self.threshold).astype(int)
    