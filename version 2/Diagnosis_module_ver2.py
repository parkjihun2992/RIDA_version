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

    def diagnosis(self, time, row, check_data, check_parameter, threshold_train_untrain, threshold_normal_abnormal, train_untrain_reconstruction_error, normal_abnormal_reconstruction_error, abnormal_procedure_result, shap_values, abnormal_verif_reconstruction_error, verif_threshold):
        if train_untrain_reconstruction_error[row-9] <= threshold_train_untrain: # Trained Data
            print(f'{time[-1]-9}sec 해당 데이터는 Train(Known) 데이터 입니다.')
            if normal_abnormal_reconstruction_error[row-9] > threshold_normal_abnormal: # Abnormal Data
                print(f'{time[-1]-9}sec 해당 데이터는 Abnormal Condition 입니다.')
                self.plot_module.SubPlotting_abnormal(time, row, train_untrain_reconstruction_error, normal_abnormal_reconstruction_error, check_data, check_parameter, threshold_train_untrain, threshold_normal_abnormal, abnormal_procedure_result, shap_values, abnormal_verif_reconstruction_error, verif_threshold)
            else: # Normal Data
                print(f'{time[-1]-9}sec 해당 데이터는 Normal Condition 입니다.')
                # print('\033[31m' + 'First and Second Result Error' + '\033[0m')
                self.plot_module.SubPlotting_normal(time, row, train_untrain_reconstruction_error, normal_abnormal_reconstruction_error, threshold_train_untrain, threshold_normal_abnormal)

        else: # UnTrained Dataset
            print(f'{time[-1]-9}sec 해당 데이터는 Untrain(Unknown) 데이터 입니다.')
            print('Untrained Data System Diagnosis 시작')
            self.plot_module.SubPlotting_untrain(time, row, train_untrain_reconstruction_error, threshold_train_untrain )
        return