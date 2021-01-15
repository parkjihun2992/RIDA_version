import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
from collections import deque
import matplotlib.pyplot as plt
import matplotlib
from Rule_module import Rule_module

class Plot_module:
    def __init__(self):
        pass

    # Dict 활용 -> Ylim 글씨로 수정
    def Plotting(self, time, row, binary_prediction_each, multiple_prediction_each, mse):
        # Binary Classification Plotting
        plt.figure(1)
        plt.rcParams['figure.figsize'] = [7, 4]  # Graph 크기 결정
        plt.plot(time[:row], np.argmax(binary_prediction_each, axis=1)[:row], 'r.', label='Prediction_Value')
        plt.pause(0.01)
        plt.title('Normal / Abnormal Classification')
        plt.xlabel('Time')
        plt.ylabel('Diagnosis Result')
        plt.grid(True)

        # AuotEncoder Classification Plotting
        plt.figure(2)
        plt.rcParams['figure.figsize'] = [7, 4]  # Graph 크기 결정
        plt.plot(time[:row], mse[:row], 'k.', label='AutoEncoder Result')
        plt.hlines(0.0035, time[0], time[-1], 'r')
        plt.pause(0.01)
        plt.title('AutoEncoder Classification')
        plt.xlabel('Time')
        plt.ylabel('AutoEncoder Result')
        plt.grid(True)

        # Multiple Classification Plotting
        plt.figure(3)
        plt.rcParams['figure.figsize'] = [7, 4]  # Graph 크기 결정
        plt.plot(time[:row], np.argmax(multiple_prediction_each, axis=1)[:row], 'r.', label='Prediction_Value')
        plt.pause(0.01)
        plt.title('Abnormal Procedure Classification')
        plt.xlabel('Time')
        plt.ylabel('Diagnosis Result')
        plt.grid(True)
        if row == 0:
            plt.close()
        return

    def SubPlotting(self, time, row, binary_prediction_each, multiple_prediction_each, mse, check_data, check_parameter, binary_result, multiple_result):
        plt.rcParams['figure.figsize'] = [16, 10]  # Graph 크기 결정

        # Binary Classification Plot
        plt.subplot(2,2,1)
        binary_result_convert = {0:'Normal', 1:'Abnormal'}
        temp1 = binary_result_convert[np.argmax(binary_prediction_each, axis=1)[-1]]
        binary_result.append(temp1)
        # plt.plot(time[:row], np.argmax(binary_prediction_each, axis=1)[:row], 'r.', label='Prediction_Value')
        plt.plot(time[:row-8], binary_result[:row], 'r.', label='Prediction_Value')
        plt.pause(0.01)
        plt.title('Normal / Abnormal Classification')
        plt.xlabel('Time')
        plt.ylabel('Diagnosis Result')
        plt.grid(True)

        # AutoEncoder Classification Plot
        plt.subplot(2,2,2)
        plt.plot(time[:row-8], mse[:row], 'k.', label='AutoEncoder Result')
        plt.hlines(0.0035, time[0], time[-1], 'r')
        plt.pause(0.01)
        plt.title('AutoEncoder Classification')
        plt.xlabel('Time')
        plt.ylabel('AutoEncoder Result')
        plt.grid(True)

        # Multiple Classification Plot
        plt.subplot(2,2,3)
        mutiple_result_convert = {0:'Normal', 1:'Ab21-01', 2:'Ab21-02', 3:'Ab20-01', 4:'Ab20-04', 5:'Ab15-07', 6:'Ab15-08', 7:'Ab63-04', 8:'Ab63-02', 9:'Ab63-03', 10:'Ab21-12', 11:'Ab19-02', 12:'Ab21-11', 13:'Ab23-03', 14:'Ab80-02', 15:'Ab60-02', 16:'Ab59-02', 17:'Ab23-01', 18:'Ab23-06', 19:'Ab59-01', 20:'Ab64-03'}
        temp2 = mutiple_result_convert[np.argmax(multiple_prediction_each, axis=1)[-1]]
        multiple_result.append(temp2)
        plt.plot(time[:row-8], multiple_result[:row], 'r.', label='Prediction_Value')
        # plt.plot(time[:row], np.argmax(multiple_prediction_each, axis=1)[:row], 'r.', label='Prediction_Value')
        plt.pause(0.01)
        plt.title('Abnormal Procedure Classification')
        plt.xlabel('Time')
        plt.ylabel('Diagnosis Result')
        plt.grid(True)
        if row == 0:
            plt.close()

        # Procedure satisfaction Histogram
        plt.subplot(2,2,4)
        rule_module = Rule_module()
        temp = rule_module.percent_procedure(check_data, check_parameter)
        plt.cla()
        plt.barh(temp.index, temp.values.reshape(-1))
        plt.pause(0.01)


        return