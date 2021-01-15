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
    # def Plotting(self, time, row, binary_prediction_each, multiple_prediction_each, mse):
    #     # Binary Classification Plotting
    #     plt.figure(1)
    #     plt.rcParams['figure.figsize'] = [7, 4]  # Graph 크기 결정
    #     plt.plot(time[:row], np.argmax(binary_prediction_each, axis=1)[:row], 'r.', label='Prediction_Value')
    #     plt.pause(0.01)
    #     plt.title('Normal / Abnormal Classification')
    #     plt.xlabel('Time')
    #     plt.ylabel('Diagnosis Result')
    #     plt.grid(True)
    #
    #     # AuotEncoder Classification Plotting
    #     plt.figure(2)
    #     plt.rcParams['figure.figsize'] = [7, 4]  # Graph 크기 결정
    #     plt.plot(time[:row], mse[:row], 'k.', label='AutoEncoder Result')
    #     plt.hlines(0.0035, time[0], time[-1], 'r')
    #     plt.pause(0.01)
    #     plt.title('AutoEncoder Classification')
    #     plt.xlabel('Time')
    #     plt.ylabel('AutoEncoder Result')
    #     plt.grid(True)
    #
    #     # Multiple Classification Plotting
    #     plt.figure(3)
    #     plt.rcParams['figure.figsize'] = [7, 4]  # Graph 크기 결정
    #     plt.plot(time[:row], np.argmax(multiple_prediction_each, axis=1)[:row], 'r.', label='Prediction_Value')
    #     plt.pause(0.01)
    #     plt.title('Abnormal Procedure Classification')
    #     plt.xlabel('Time')
    #     plt.ylabel('Diagnosis Result')
    #     plt.grid(True)
    #     if row == 0:
    #         plt.close()
    #     return

    def SubPlotting_untrain(self, time, row, train_untrain_reconstruction_error, threshold_train_untrain):
        plt.rcParams['figure.figsize'] = [4, 2]  # Graph 크기 결정

        # Trained / Untrained Result Plot
        plt.subplot(2, 3, 1)
        if train_untrain_reconstruction_error[row-9] > threshold_train_untrain:  # Result : Untrain
            plt.plot(time[row-9], train_untrain_reconstruction_error[row-9], 'rx')
        else:  # Result : Train
            plt.plot(time[row - 9], train_untrain_reconstruction_error[row-9], 'g.')
        line = plt.hlines(threshold_train_untrain, time[0], time[-1], 'k', label='Threshold')
        plt.pause(0.01)
        plt.title('Trained / Untrained Classification')
        plt.xlabel('Time')
        plt.ylabel('Reconstruction Error')
        plt.legend(handles=[line])
        plt.grid(True)

        # System Diagnosis
        # plt.subplot(1,2,2)

    def SubPlotting_normal(self, time, row, train_untrain_reconstruction_error, normal_abnormal_reconstruction_error, threshold_train_untrain, threshold_normal_abnormal):
        plt.rcParams['figure.figsize'] = [16, 10]  # Graph 크기 결정

        # Trained / Untrained Result Plot
        plt.subplot(2, 3, 1)
        if train_untrain_reconstruction_error[row-9] > threshold_train_untrain:  # Result : Untrain
            plt.plot(time[row-9], train_untrain_reconstruction_error[row-9], 'rx')
        else:  # Result : Train
            plt.plot(time[row-9], train_untrain_reconstruction_error[row-9], 'g.')
        line = plt.hlines(threshold_train_untrain, time[0], time[-1], 'k', label='Threshold')
        plt.pause(0.01)
        plt.title('Trained / Untrained Classification')
        plt.xlabel('Time')
        plt.ylabel('Reconstruction Error')
        plt.legend(handles=[line])
        plt.grid(True)

        # Normal / Abnormal Result Plot
        plt.subplot(2, 3, 2)
        if normal_abnormal_reconstruction_error[row-9] > threshold_normal_abnormal:  # Result : Abnormal Condition
            plt.plot(time[row-9], normal_abnormal_reconstruction_error[row-9], 'rx')
        else:  # Result : Normal Condition
            plt.plot(time[row-9], normal_abnormal_reconstruction_error[row-9], 'g.')
        line = plt.hlines(threshold_normal_abnormal, time[0], time[-1], 'k', label='Threshold')
        plt.pause(0.01)
        plt.title('Normal / Abnormal Classification')
        plt.xlabel('Time')
        plt.ylabel('Reconstruction Error')
        plt.legend(handles=[line])
        plt.grid(True)


    def SubPlotting_abnormal(self, time, row, train_untrain_reconstruction_error, normal_abnormal_reconstruction_error, check_data, check_parameter, threshold_train_untrain, threshold_normal_abnormal, abnormal_procedure_result, shap_values, abnormal_verif_reconstruction_error, verif_threshold):
        plt.rcParams['figure.figsize'] = [16, 10]  # Graph 크기 결정

        # Trained / Untrained Result Plot
        plt.subplot(2, 3, 1)
        if train_untrain_reconstruction_error[row-9] > threshold_train_untrain:  # Result : Untrain
            plt.plot(time[row-9], train_untrain_reconstruction_error[row-9], 'rx')
        else:  # Result : Train
            plt.plot(time[row-9], train_untrain_reconstruction_error[row-9], 'g.')
        line = plt.hlines(threshold_train_untrain, time[0], time[-1], 'k', label='Threshold')
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
        line = plt.hlines(threshold_normal_abnormal, time[0], time[-1], 'k', label='Threshold')
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
        plt.pause(0.01)
        plt.title('Abnormal Procedure Classification')
        plt.xlabel('Diagnostic Probability')
        # plt.grid(True)
        if row == 0:
            plt.close()

        # Procedure satisfaction Histogram
        plt.subplot(2,3,4)
        rule_module = Rule_module()
        temp = rule_module.percent_procedure(check_data, check_parameter)
        plt.cla()
        plt.barh(temp.index, temp.values.reshape(-1))
        plt.title('Procedure satisfaction Histogram')
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
        if abnormal_verif_reconstruction_error[row - 9] > verif_threshold:  # Result : Abnormal Condition
            plt.plot(time[row - 9], abnormal_verif_reconstruction_error[row - 9], 'rx')
        else:  # Result : Normal Condition
            plt.plot(time[row - 9], abnormal_verif_reconstruction_error[row - 9], 'g.')
        line = plt.hlines(verif_threshold, time[0], time[-1], 'k', label='Threshold')
        plt.pause(0.01)
        plt.title('Abnormal Procedure Classification Verification')
        plt.xlabel('Time')
        plt.ylabel('Reconstruction Error')
        plt.legend(handles=[line])
        plt.grid(True)

        return