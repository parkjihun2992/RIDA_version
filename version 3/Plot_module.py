import numpy as np
import pandas as pd
import pickle
from collections import deque
import matplotlib.pyplot as plt
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from Rule_module import Rule_module

class Plot_module:
    def __init__(self):
        self.threshold = deque(maxlen=2)

    def SubPlotting_untrain(self, time, row, train_untrain_reconstruction_error, threshold_train_untrain, check_data):
        plt.rcParams['figure.figsize'] = [4, 2]  # Graph 크기 결정

        # Trained / Untrained Result Plot
        plt.subplot(2, 3, 1)
        if train_untrain_reconstruction_error[row-9] > threshold_train_untrain:  # Result : Untrain
            plt.plot(time[row-9], train_untrain_reconstruction_error[row-9], 'rx')
        else:  # Result : Train
            plt.plot(time[row - 9], train_untrain_reconstruction_error[row-9], 'g.')
        # line = plt.hlines(threshold_train_untrain, time[0], time[-1], 'k', label='Threshold')
        line = plt.axhline(threshold_train_untrain, color='k', linewidth=2, label='Threshold')
        plt.pause(0.01)
        plt.title('Trained / Untrained Classification')
        plt.xlabel('Time')
        plt.ylabel('Reconstruction Error')
        plt.legend(handles=[line])
        plt.grid(True)

        # plt.subplot(2, 3, 2)
        # rule_module = Rule_module()
        # sys_ = rule_module.system_daignosis(check_data)
        # plt.cla()
        # plt.barh(sys_.index, sys_.values.reshape(-1))
        # plt.title('System Diagnosis Result')
        # plt.pause(0.01)

        plt.subplot(2, 3, 2)
        rule_module = Rule_module()
        sys_ = rule_module.system_daignosis(check_data)
        plt.cla()
        # pp = sys_[['Satisfaction','Total']].plot(kind='barh', stacked=True)
        plt.barh(sys_.index, sys_['Satisfaction'], color='steelblue', label='Satisfaction')
        plt.barh(sys_.index, sys_['Total'], left=sys_['Satisfaction'], color='powderblue', label='Total_Satisfaction')
        # plt.barh(sys_.index, sys_.values.reshape(-1))
        plt.title('System Diagnosis Result')
        plt.legend()
        # plt.legend(handles=[satis, tot])
        plt.pause(0.01)

        plt.tight_layout()

    def SubPlotting_normal(self, time, row, train_untrain_reconstruction_error, normal_abnormal_reconstruction_error, threshold_train_untrain, threshold_normal_abnormal):
        plt.rcParams['figure.figsize'] = [16, 10]  # Graph 크기 결정

        # Trained / Untrained Result Plot
        plt.subplot(1, 2, 1)
        if train_untrain_reconstruction_error[row-9] > threshold_train_untrain:  # Result : Untrain
            plt.plot(time[row-9], train_untrain_reconstruction_error[row-9], 'rx')
        else:  # Result : Train
            plt.plot(time[row-9], train_untrain_reconstruction_error[row-9], 'g.')
        # line = plt.hlines(threshold_train_untrain, time[0], time[-1], 'k', label='Threshold')
        line = plt.axhline(threshold_train_untrain, color='k', linewidth=2, label='Threshold')
        plt.pause(0.01)
        plt.title('Trained / Untrained Classification')
        plt.xlabel('Time')
        plt.ylabel('Reconstruction Error')
        plt.legend(handles=[line])
        plt.grid(True)

        # Normal / Abnormal Result Plot
        plt.subplot(1, 2, 2)
        if normal_abnormal_reconstruction_error[row-9] > threshold_normal_abnormal:  # Result : Abnormal Condition
            plt.plot(time[row-9], normal_abnormal_reconstruction_error[row-9], 'rx')
        else:  # Result : Normal Condition
            plt.plot(time[row-9], normal_abnormal_reconstruction_error[row-9], 'g.')
        # line = plt.hlines(threshold_normal_abnormal, time[0], time[-1], 'k', label='Threshold')
        line = plt.axhline(threshold_normal_abnormal, color='k', linewidth=2, label='Threshold')
        plt.pause(0.01)
        plt.title('Normal / Abnormal Classification')
        plt.xlabel('Time')
        plt.ylabel('Reconstruction Error')
        plt.legend(handles=[line])
        plt.grid(True)

        plt.tight_layout()

    def SubPlotting_abnormal(self, time, row, train_untrain_reconstruction_error, normal_abnormal_reconstruction_error, check_data, check_parameter, threshold_train_untrain, threshold_normal_abnormal, abnormal_procedure_result, shap_values, abnormal_verif_reconstruction_error, verif_threshold):
        plt.rcParams['figure.figsize'] = [16, 10]  # Graph 크기 결정

        # Trained / Untrained Result Plot
        plt.subplot(2, 3, 1)
        if train_untrain_reconstruction_error[row-9] > threshold_train_untrain:  # Result : Untrain
            plt.plot(time[row-9], train_untrain_reconstruction_error[row-9], 'rx')
        else:  # Result : Train
            plt.plot(time[row-9], train_untrain_reconstruction_error[row-9], 'g.')
        # line = plt.hlines(threshold_train_untrain, time[0], time[-1], 'k', label='Threshold')
        line = plt.axhline(threshold_train_untrain, color='k', linewidth=2, label='Threshold')
        plt.pause(0.01)
        plt.title('Trained / Untrained Classification')
        plt.xlabel('Time')
        plt.ylabel('Reconstruction Error')
        plt.legend(handles=[line])
        plt.grid(True)

        # Normal / Abnormal Result Plot
        plt.subplot(2, 3, 2)
        if normal_abnormal_reconstruction_error[row-9] > threshold_normal_abnormal:  # Result : Abnormal Condition
            plt.plot(time[row - 9], normal_abnormal_reconstruction_error[row-9], 'rx')
        else:  # Result : Normal Condition
            plt.plot(time[row - 9], normal_abnormal_reconstruction_error[row-9], 'g.')
        # line = plt.hlines(threshold_normal_abnormal, time[0], time[-1], 'k', label='Threshold')
        line = plt.axhline(threshold_normal_abnormal, color='k', linewidth=2, label='Threshold')
        plt.pause(0.01)
        plt.title('Normal / Abnormal Classification')
        plt.xlabel('Time')
        plt.ylabel('Reconstruction Error')
        plt.legend(handles=[line])
        plt.grid(True)

        # Abnormal Procedure Result Plot
        plt.subplot(2,3,3)
        # mutiple_result_convert = {0:'Normal',1:'Ab21-01', 2:'Ab21-02', 3:'Ab20-04', 4:'Ab15-07', 5:'Ab15-08', 6:'Ab63-04', 7:'Ab63-02', 8:'Ab21-12', 9:'Ab19-02', 10:'Ab21-11', 11:'Ab23-03', 12:'Ab60-02', 13:'Ab59-02', 14:'Ab23-01', 15:'Ab23-06'}
        mutiple_result_convert = ['Normal','Ab21-01', 'Ab21-02', 'Ab20-04', 'Ab15-07', 'Ab15-08', 'Ab63-04', 'Ab63-02', 'Ab21-12', 'Ab19-02', 'Ab21-11', 'Ab23-03', 'Ab60-02', 'Ab59-02', 'Ab23-01', 'Ab23-06']
        # temp2 = mutiple_result_convert[np.argmax(abnormal_procedure_result[row-9], axis=1)[0]]
        temp1 = pd.DataFrame(np.array((abnormal_procedure_result[row - 9])*100), columns=mutiple_result_convert)
        temp2 = temp1.sort_values(by=0, ascending=True, axis=1)
        # multiple_result.append(temp2)
        plt.cla()
        plt.barh(temp2.columns, np.array(temp2)[0])
        # plt.plot(time[row-9], temp2, 'r.', label='Prediction_Value')
        # plt.plot(time[:row], np.argmax(multiple_prediction_each, axis=1)[:row], 'r.', label='Prediction_Value')
        plt.title('Abnormal Procedure Classification')
        plt.xlabel('Diagnostic Probability')
        plt.pause(0.01)
        # plt.grid(True)
        if row == 0:
            plt.close()

        # Procedure satisfaction Histogram
        plt.subplot(2,3,4)
        rule_module = Rule_module()
        temp = rule_module.percent_procedure(check_data, check_parameter)
        plt.cla()
        plt.barh(temp.index, temp['Satisfaction'], color='steelblue', label='Satisfaction')
        plt.barh(temp.index, temp['Total'], left=temp['Satisfaction'], color='powderblue', label='Total_Satisfaction')
        # plt.barh(temp.index, temp.values.reshape(-1))
        plt.title('Procedure satisfaction Histogram')
        plt.legend()
        plt.pause(0.01)

        # XAI related variable
        plt.subplot(2,3,5)
        plt.cla()
        plt.barh(shap_values.columns, np.array(shap_values)[0])
        plt.title('Impact Parameter of Diagnosis')
        plt.xlabel('Impact Factor')
        plt.pause(0.01)

        # Verification of Abnormal Procedure Classification Result
        plt.subplot(2, 3, 6)
        self.threshold.append(verif_threshold)
        if abnormal_verif_reconstruction_error[row - 9] > verif_threshold:  # Result : Abnormal Condition
            plt.plot(time[row - 9], abnormal_verif_reconstruction_error[row - 9], 'rx')
        else:  # Result : Normal Condition
            plt.plot(time[row - 9], abnormal_verif_reconstruction_error[row - 9], 'g.')
        if np.shape(self.threshold)[0] == 2 and self.threshold[0] != self.threshold[1]:
            plt.cla()
        line = plt.axhline(verif_threshold, color='k', linewidth=2, label='Threshold')
        plt.pause(0.01)
        plt.title('Abnormal Procedure Classification Verification')
        plt.xlabel('Time')
        plt.ylabel('Reconstruction Error')
        plt.legend(handles=[line])
        plt.grid(True)

        plt.tight_layout()
        return



