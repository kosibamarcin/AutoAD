import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_skab(data, true_anomalies, detected_anomalies=None):
    if detected_anomalies is None:
        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1]}, figsize=(20,8))
        a0.plot(data['Accelerometer1RMS'], label='Accelerometer1RMS')
        a0.plot(data['Accelerometer2RMS'], label='Accelerometer2RMS')
        a0.plot(data['Current'], label='Current')
        a0.plot(data['Pressure'], label='Pressure')
        a0.plot(data['Temperature'], label='Thermocouple')
        a0.plot(data['Voltage'], label='Voltage')
        a0.plot(data['Volume Flow RateRMS'], label='Volume Flow RateRMS')
        a0.legend()
        a1.plot(true_anomalies, 'r', label='Anomalies Ground Truth')
        a1.legend()
        plt.show()
    else:
        f, (a0, a1, a2) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [5, 1, 1]}, figsize=(20,8))
        a0.plot(data['Accelerometer1RMS'], label='Accelerometer1RMS')
        a0.plot(data['Accelerometer2RMS'], label='Accelerometer2RMS')
        a0.plot(data['Current'], label='Current')
        a0.plot(data['Pressure'], label='Pressure')
        a0.plot(data['Temperature'], label='Thermocouple')
        a0.plot(data['Voltage'], label='Voltage')
        a0.plot(data['Volume Flow RateRMS'], label='Volume Flow RateRMS')
        a0.legend()
        a1.plot(true_anomalies, 'r', label='Ground Truth Anomalies')
        a1.legend()
        a2.plot(detected_anomalies, 'r', label='Predicted Anomalies')
        a2.legend()
        plt.show()      