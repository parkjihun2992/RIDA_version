import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
from collections import deque
import matplotlib.pyplot as plt

class Model_module:
    def __init__(self):
        pass

    def load_model(self):
        # Model Code + Weight (나중에..)
        self.binary_LSTM = load_model('LSTM_CNS_27epcoh_saved_model.h5')
        self.multiple_LSTM = load_model('LSTM_CNS_12epcoh_saved_model.h5')
        self.autoencoder = load_model('model.h5')
        return self.binary_LSTM, self.multiple_LSTM, self.autoencoder

    def binary_prediction(self, data, dim2, binary_prediction_each, mse):
        # Binary LSTM을 활용한 분류
        binary_prediction = self.binary_LSTM.predict(data)
        binary_prediction = binary_prediction[0]
        binary_prediction_each.append(binary_prediction)
        self.binary_prediction_each = binary_prediction_each
        # AutoEncoder를 활용한 분류
        AE_error = deque(maxlen=1)
        AE_prediction = self.autoencoder.predict(dim2)
        AE_error.append(AE_prediction[0])
        error = np.mean(np.power(data[0][0] - AE_error, 2), axis=1)
        mse.append(error)
        self.mse = mse
        return binary_prediction, self.binary_prediction_each, self.mse

    def multiple_prediction(self, data, multiple_prediction_each):
        multiple_prediction = self.multiple_LSTM.predict(data)
        multiple_prediction = multiple_prediction[0]
        multiple_prediction_each.append(multiple_prediction)
        self.multiple_prediction_each = multiple_prediction_each
        return multiple_prediction, self.multiple_prediction_each


