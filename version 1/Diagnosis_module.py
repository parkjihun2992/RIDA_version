import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
from collections import deque
import matplotlib.pyplot as plt
from Plot_module import Plot_module

class Diagnosis_module:
    def __init__(self):
        self.plot_module = Plot_module()

    def diagnosis(self, binary_prediction, multiple_prediction, time, row, binary_prediction_each, multiple_prediction_each, mse, check_data, check_parameter, binary_result, multiple_result):
        if binary_prediction[0] > binary_prediction[1]: # First 예측결과가 정상일 경우
            print(f'{time[-1]}sec First Prediction Result : Normal')
            if multiple_prediction[0] > multiple_prediction[1:].max(): # Second 예측결과가 정상일 경우
                print(f'{time[-1]}sec Second Prediction Result : Normal')
                self.plot_module.SubPlotting(time, row, binary_prediction_each, multiple_prediction_each, mse, check_data, check_parameter, binary_result, multiple_result)
            else: # Second 예측결과가 비정상일 경우
                print(f'{time[-1]}sec Second Prediction Result : Abnormal')
                print('\033[31m' + 'First and Second Result Error' + '\033[0m')

        else: # First 예측결과가 비정상일 경우
            print(f'{time[-1]}sec First Prediction Result : Abnormal')
            if multiple_prediction[0] < multiple_prediction[1:].max():  # Second 예측결과가 비정상일 경우
                print(f'{time[-1]}sec Second Prediction Result : Abnormal')
                self.plot_module.SubPlotting(time, row, binary_prediction_each, multiple_prediction_each, mse, check_data, check_parameter, binary_result, multiple_result)
            else: # Second 예측결과가 정상일 경우
                print(f'{time[-1]}sec Second Prediction Result : Normal')
                print('\033[31m' + 'First and Second Result Error' + '\033[0m')
        return