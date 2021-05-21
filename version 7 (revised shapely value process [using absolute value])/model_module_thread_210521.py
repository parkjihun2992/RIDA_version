import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
from collections import deque
import time
import shap


class Model_module:
    def __init__(self):
        self.diagnosed_shap_result_abs = {'Normal':deque(maxlen=1), 'Ab21_01':deque(maxlen=1), 'Ab21_02':deque(maxlen=1), 'Ab20_04':deque(maxlen=1), 'Ab15_07':deque(maxlen=1), 'Ab15_08':deque(maxlen=1), 'Ab63_04':deque(maxlen=1), 'Ab63_02':deque(maxlen=1), 'Ab21_12':deque(maxlen=1), 'Ab19_02':deque(maxlen=1), 'Ab21_11':deque(maxlen=1), 'Ab23_03':deque(maxlen=1), 'Ab60_02':deque(maxlen=1), 'Ab59_02':deque(maxlen=1), 'Ab23_01':deque(maxlen=1), 'Ab23_06':deque(maxlen=1)}
        self.undiagnosed_shap_result_abs = {'Normal':deque(maxlen=1), 'Ab21_01':deque(maxlen=1), 'Ab21_02':deque(maxlen=1), 'Ab20_04':deque(maxlen=1), 'Ab15_07':deque(maxlen=1), 'Ab15_08':deque(maxlen=1), 'Ab63_04':deque(maxlen=1), 'Ab63_02':deque(maxlen=1), 'Ab21_12':deque(maxlen=1), 'Ab19_02':deque(maxlen=1), 'Ab21_11':deque(maxlen=1), 'Ab23_03':deque(maxlen=1), 'Ab60_02':deque(maxlen=1), 'Ab59_02':deque(maxlen=1), 'Ab23_01':deque(maxlen=1), 'Ab23_06':deque(maxlen=1)}
        self.selected_para = pd.read_csv('./DataBase/Final_parameter_200825.csv')


    def flatten(self, X):
        '''
        Flatten a 3D array.

        Input
        X            A 3D array for lstm, where the array is sample x timesteps x features.

        Output
        flattened_X  A 2D array, sample x features.
        '''
        flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
        for i in range(X.shape[0]):
            flattened_X[i] = X[i, (X.shape[1] - 1), :]
        return (flattened_X)


    def load_model(self):
        # Model Code + Weight (나중에..)
        # Compile False Option -> Very fast model load
        self.train_untrain_lstm_autoencoder = load_model('model/Train_Untrain_epoch27_[0.00225299]_acc_[0.9724685967462512].h5', compile=False)
        self.multiple_xgbclassification = pickle.load(open('model/Lightgbm_max_depth_feature_137_200825.h5', 'rb')) # multiclassova
        self.explainer = pickle.load(open('model/explainer.pkl', 'rb')) # pickle로 저장하여 로드하면 더 빠름.
        self.normal_abnormal_lstm_autoencoder = load_model('model/node32_Nor_Ab_epoch157_[0.00022733]_acc_[0.9758567350470185].h5', compile=False)
        self.ab_21_01_lstmautoencoder = load_model('model/ab21-01_epoch74_[0.00712891]_acc_[0.9653829136428816].h5', compile=False)
        self.ab_21_02_lstmautoencoder = load_model('model/ab21-02_epoch200_[0.01140293]_acc_[0.8144184191122964]_rev1.h5', compile=False)
        self.ab_20_04_lstmautoencoder = load_model('model/ab20-04_epoch62_[0.00121183]_acc_[0.9278062539244847].h5', compile=False)
        self.ab_15_07_lstmautoencoder = load_model('model/ab15-07_epoch100_[0.00048918]_acc_[0.9711231231353337].h5', compile=False)
        self.ab_15_08_lstmautoencoder = load_model('model/ab15-08_epoch97_[0.00454363]_acc_[0.9602876916386659].h5', compile=False)
        self.ab_63_04_lstmautoencoder = load_model('model/ab63-04_epoch92_[0.00015933]_acc_[0.9953701129330438].h5', compile=False)
        self.ab_63_02_lstmautoencoder = load_model('model/ab63-02_epoch100_[0.00376744]_acc_[0.8948400650463064].h5', compile=False)
        self.ab_21_12_lstmautoencoder = load_model('model/ab21-12_epoch40_[0.00236955]_acc_[0.9473147939642043].h5', compile=False)
        self.ab_19_02_lstmautoencoder = load_model('model/ab19-02_epoch200_[0.00610265]_acc_[0.8852787664918006]_rev1.h5', compile=False)
        self.ab_21_11_lstmautoencoder = load_model('model/ab21-11_epoch98_[0.00265125]_acc_[0.9907740136695732].h5', compile=False)
        self.ab_60_02_lstmautoencoder = load_model('model/ab60-02_epoch100_[0.00370811]_acc_[0.989989808320697].h5', compile=False)
        self.ab_23_03_lstmautoencoder = load_model('model/ab23-03_epoch91_[0.00017211]_acc_[0.9931771647209489].h5', compile=False)
        self.ab_59_02_lstmautoencoder = load_model('model/ab59-02_epoch98_[0.00424398]_acc_[0.9923899523150299].h5', compile=False)
        self.ab_23_01_lstmautoencoder = load_model('model/ab23-01_epoch93_[0.00314694]_acc_[0.9407692298522362].h5', compile=False)
        self.ab_23_06_lstmautoencoder = load_model('model/ab23-06_epoch96_[0.00584512]_acc_[0.8694280893287306].h5', compile=False)
        return


    def train_untrain_classifier(self, data):
        # LSTM_AutoEncoder를 활용한 Train / Untrain 분류
        train_untrain_prediction = self.train_untrain_lstm_autoencoder.predict(data) # 예측
        train_untrain_error = np.mean(np.power(self.flatten(data) - self.flatten(train_untrain_prediction), 2), axis=1)
        if train_untrain_error[0] <= 0.00225299:
            train_untrain_result = 0 # trained condition
        else:
            train_untrain_result = 1 # untrained condition
        return train_untrain_result

    def abnormal_procedure_classifier_0(self, data): # 진단 확률 및 Shapley value 출력
        self.abnormal_procedure_prediction = self.multiple_xgbclassification.predict(data) # Softmax 예측 값 출력
        self.diagnosed_scenario = np.argmax(self.abnormal_procedure_prediction, axis=1)[0]
        self.shap_value = self.explainer.shap_values(data)  # Shap_value 출력
        return self.abnormal_procedure_prediction, self.diagnosed_scenario

    def abnormal_procedure_classifier_1(self): # 진단된 시나리오의 Shapley value 처리 (변수명, 변수설명, 값, 부호)
        diagnosis_convert_text = {0: 'Normal', 1: 'Ab21_01', 2: 'Ab21_02', 3: 'Ab20_04', 4: 'Ab15_07', 5: 'Ab15_08',
                         6: 'Ab63_04', 7: 'Ab63_02', 8: 'Ab21_12',9: 'Ab19_02', 10: 'Ab21_11', 11: 'Ab23_03',
                         12: 'Ab60_02', 13: 'Ab59_02', 14: 'Ab23_01', 15: 'Ab23_06'}

        temp1 = pd.DataFrame(self.shap_value[self.diagnosed_scenario], columns=self.selected_para['0'].tolist()).T
        sign = []
        for i in range(len(temp1[0])):
            if np.sign(temp1[0][i]) == 1.0 or np.sign(temp1[0][i]) == 0.0:
                sign.append('+')
            elif np.sign(temp1[0][i]) == -1.0:
                sign.append('-')
        prob = [np.round((np.abs(temp1[0][i])/sum(np.abs(temp1[0])))*100, 2) for i in range(len(temp1[0]))]
        temp2 = pd.DataFrame([temp1.index, self.selected_para['1'].tolist(), np.abs(temp1.values), prob, sign], index=['variable', 'describe', 'value', 'probability', 'sign']).T.sort_values(by='value', ascending=False, axis=0).reset_index(drop=True)
        temp2 = temp2[temp2['value'] > 0]
        self.diagnosed_shap_result_abs[diagnosis_convert_text[self.diagnosed_scenario]].append(temp2)
        return self.diagnosed_shap_result_abs

    def abnormal_procedure_classifier_2(self, scenario): # 진단되지 않은 시나리오의 Shapley value 처리
        int_convert_text = {0:'Normal', 1:'Ab21_01', 2:'Ab21_02', 3:'Ab20_04', 4:'Ab15_07', 5:'Ab15_08',
                                  6:'Ab63_04', 7:'Ab63_02', 8:'Ab21_12', 9:'Ab19_02', 10:'Ab21_11', 11:'Ab23_03',
                                  12:'Ab60_02', 13:'Ab59_02', 14:'Ab23_01', 15:'Ab23_06'}

        temp1 = pd.DataFrame(self.shap_value[scenario], columns=self.selected_para['0'].tolist()).T
        sign = []
        for i in range(len(temp1[0])):
            if np.sign(temp1[0][i]) == 1.0 or np.sign(temp1[0][i]) == 0.0:
                sign.append('+')
            elif np.sign(temp1[0][i]) == -1.0:
                sign.append('-')
        prob = [np.round((np.abs(temp1[0][i]) / sum(np.abs(temp1[0]))) * 100, 2) for i in range(len(temp1[0]))]
        temp2 = pd.DataFrame([temp1.index, self.selected_para['1'].tolist(), np.abs(temp1.values), prob, sign], index=['variable', 'describe', 'value', 'probability', 'sign']).T.sort_values(by='value', ascending=False, axis=0).reset_index(drop=True)
        temp2 = temp2[temp2['value']>0]
        self.undiagnosed_shap_result_abs[int_convert_text[scenario]].append(temp2)
        return self.undiagnosed_shap_result_abs

    # def abnormal_procedure_classifier_1(self):
    #     diagnosis_convert_text = {0: 'Normal', 1: 'Ab21_01', 2: 'Ab21_02', 3: 'Ab20_04', 4: 'Ab15_07', 5: 'Ab15_08',
    #                      6: 'Ab63_04', 7: 'Ab63_02', 8: 'Ab21_12',9: 'Ab19_02', 10: 'Ab21_11', 11: 'Ab23_03',
    #                      12: 'Ab60_02', 13: 'Ab59_02', 14: 'Ab23_01', 15: 'Ab23_06'}
    #     sort_shap_values_positive = pd.DataFrame(self.shap_value[np.argmax(self.abnormal_procedure_prediction, axis=1)[0]], columns=self.selected_para['0'].tolist()).sort_values(by=0, ascending=False, axis=1)
    #     drop_shap_values_positive = sort_shap_values_positive[sort_shap_values_positive.iloc[:]>0].dropna(axis=1).T
    #     reset_shap_values_positive = drop_shap_values_positive.reset_index()
    #     column_positive = reset_shap_values_positive['index']
    #     var_positive = [self.selected_para['0'][self.selected_para['0'] == col_].index for col_ in column_positive]
    #     val_col_positive = [self.selected_para['1'][var_].iloc[0] for var_ in var_positive]
    #     proba_positive = [(reset_shap_values_positive[0][val_num]/sum(reset_shap_values_positive[0]))*100 for val_num in range(len(reset_shap_values_positive))]
    #     val_system_positive = [self.selected_para['2'][var_].iloc[0] for var_ in var_positive]
    #     reset_shap_values_positive['describe'] = val_col_positive
    #     reset_shap_values_positive['probability'] = proba_positive
    #     reset_shap_values_positive['system'] = val_system_positive
    #     self.shap_result_postive[diagnosis_convert_text[np.argmax(self.abnormal_procedure_prediction, axis=1)[0]]].append(reset_shap_values_positive)
    #     return self.shap_result_postive

    # def abnormal_procedure_classifier_2(self, scenario): # 진단되지 않은 시나리오의 Shapley value 처리
    #     int_convert_text = {0:'Normal', 1:'Ab21_01', 2:'Ab21_02', 3:'Ab20_04', 4:'Ab15_07', 5:'Ab15_08',
    #                               6:'Ab63_04', 7:'Ab63_02', 8:'Ab21_12', 9:'Ab19_02', 10:'Ab21_11', 11:'Ab23_03',
    #                               12:'Ab60_02', 13:'Ab59_02', 14:'Ab23_01', 15:'Ab23_06'}
    #
    #     sort_shap_values_negative = pd.DataFrame(self.shap_value[scenario], columns=self.selected_para['0'].tolist()).sort_values(by=0, ascending=True, axis=1)
    #     drop_shap_values_negative = sort_shap_values_negative[sort_shap_values_negative.iloc[:]<0].dropna(axis=1).T
    #     reset_shap_values_negative = drop_shap_values_negative.reset_index()
    #     column_negative = reset_shap_values_negative['index']
    #     val_negative = [self.selected_para['0'][self.selected_para['0'] == col_].index for col_ in column_negative]
    #     val_col_negative = [self.selected_para['1'][var_].iloc[0] for var_ in val_negative]
    #     proba_negative = [(reset_shap_values_negative[0][val_num]/sum(reset_shap_values_negative[0]))*100 for val_num in range(len(reset_shap_values_negative))]
    #     val_system_negative = [self.selected_para['2'][var_].iloc[0] for var_ in val_negative]
    #     reset_shap_values_negative['describe'] = val_col_negative
    #     reset_shap_values_negative['probability'] = proba_negative
    #     reset_shap_values_negative['system'] = val_system_negative
    #     self.shap_result_negative[int_convert_text[scenario]].append(reset_shap_values_negative)
    #     return self.shap_result_negative

    def abnormal_procedure_verification(self, data): # 만약 abnormal 예측이 실패한다면??
        global verif_prediction, verif_threshold
        if self.diagnosed_scenario == 0:
            verif_prediction = self.normal_abnormal_lstm_autoencoder.predict(data)
            verif_threshold = 0.00022733
        elif self.diagnosed_scenario == 1:
            verif_prediction = self.ab_21_01_lstmautoencoder.predict(data)
            verif_threshold = 0.00712891
        elif self.diagnosed_scenario == 2:
            verif_prediction = self.ab_21_02_lstmautoencoder.predict(data)
            verif_threshold = 0.01140293
        elif self.diagnosed_scenario == 3:
            verif_prediction = self.ab_20_04_lstmautoencoder.predict(data)
            verif_threshold = 0.00121183
        elif self.diagnosed_scenario == 4:
            verif_prediction = self.ab_15_07_lstmautoencoder.predict(data)
            verif_threshold = 0.00048918
        elif self.diagnosed_scenario == 5:
            verif_prediction = self.ab_15_08_lstmautoencoder.predict(data)
            verif_threshold = 0.00454363
        elif self.diagnosed_scenario == 6:
            verif_prediction = self.ab_63_04_lstmautoencoder.predict(data)
            verif_threshold = 0.00015933
        elif self.diagnosed_scenario == 7:
            verif_prediction = self.ab_63_02_lstmautoencoder.predict(data)
            verif_threshold =0.00376744
        elif self.diagnosed_scenario == 8:
            verif_prediction = self.ab_21_12_lstmautoencoder.predict(data)
            verif_threshold = 0.00236955
        elif self.diagnosed_scenario == 9:
            verif_prediction = self.ab_19_02_lstmautoencoder.predict(data)
            verif_threshold = 0.00610265
        elif self.diagnosed_scenario == 10:
            verif_prediction = self.ab_21_11_lstmautoencoder.predict(data)
            verif_threshold = 0.00265125
        elif self.diagnosed_scenario == 11:
            verif_prediction = self.ab_23_03_lstmautoencoder.predict(data)
            verif_threshold = 0.00017211
        elif self.diagnosed_scenario == 12:
            verif_prediction = self.ab_60_02_lstmautoencoder.predict(data)
            verif_threshold = 0.00370811
        elif self.diagnosed_scenario == 13:
            verif_prediction = self.ab_59_02_lstmautoencoder.predict(data)
            verif_threshold = 0.00424398
        elif self.diagnosed_scenario == 14:
            verif_prediction = self.ab_23_01_lstmautoencoder.predict(data)
            verif_threshold = 0.00314694
        elif self.diagnosed_scenario == 15:
            verif_prediction = self.ab_23_06_lstmautoencoder.predict(data)
            verif_threshold = 0.00584512
        abnormal_verif_error = np.mean(np.power(self.flatten(data) - self.flatten(verif_prediction), 2), axis=1)
        if abnormal_verif_error <= verif_threshold: # diagnosis success
            verif_result = 0
        else: # diagnosis failure
            verif_result = 1
        return verif_result



