import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
from collections import deque
import time
import shap


class Model_module:
    def __init__(self):
        self.train_untrain_reconstruction_error = []
        self.normal_abnormal_reconstruction_error = []
        self.abnormal_procedure_result = []
        self.abnormal_verif_reconstruction_error = []
        self.shap_result_postive = {'Normal':deque(maxlen=1), 'Ab21_01':deque(maxlen=1), 'Ab21_02':deque(maxlen=1), 'Ab20_04':deque(maxlen=1), 'Ab15_07':deque(maxlen=1), 'Ab15_08':deque(maxlen=1), 'Ab63_04':deque(maxlen=1), 'Ab63_02':deque(maxlen=1), 'Ab21_12':deque(maxlen=1), 'Ab19_02':deque(maxlen=1), 'Ab21_11':deque(maxlen=1), 'Ab23_03':deque(maxlen=1), 'Ab60_02':deque(maxlen=1), 'Ab59_02':deque(maxlen=1), 'Ab23_01':deque(maxlen=1), 'Ab23_06':deque(maxlen=1)}
        self.shap_result_negative = {'Normal':deque(maxlen=1), 'Ab21_01':deque(maxlen=1), 'Ab21_02':deque(maxlen=1), 'Ab20_04':deque(maxlen=1), 'Ab15_07':deque(maxlen=1), 'Ab15_08':deque(maxlen=1), 'Ab63_04':deque(maxlen=1), 'Ab63_02':deque(maxlen=1), 'Ab21_12':deque(maxlen=1), 'Ab19_02':deque(maxlen=1), 'Ab21_11':deque(maxlen=1), 'Ab23_03':deque(maxlen=1), 'Ab60_02':deque(maxlen=1), 'Ab59_02':deque(maxlen=1), 'Ab23_01':deque(maxlen=1), 'Ab23_06':deque(maxlen=1)}
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
        self.normal_abnormal_lstm_autoencoder = load_model('model/node32_Nor_Ab_epoch157_[0.00022733]_acc_[0.9758567350470185].h5', compile=False)
        self.multiple_xgbclassification = pickle.load(open('model/Lightgbm_max_depth_feature_137_200825.h5', 'rb')) # multiclassova
        # self.explainer = shap.TreeExplainer(self.multiple_xgbclassification, feature_perturbation='tree_path_dependent') # pickle로 저장하여 로드하면 더 빠름.
        self.explainer = pickle.load(open('model/explainer.pkl', 'rb')) # pickle로 저장하여 로드하면 더 빠름.
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
        # train_untrain_result = deque(maxlen=1)
        # train_untrain_result.append(train_untrain_prediction[0]) # 예측결과 저장
        train_untrain_error = np.mean(np.power(self.flatten(data) - self.flatten(train_untrain_prediction), 2), axis=1)
        # self.train_untrain_reconstruction_error.append(train_untrain_error) # 굳이 데이터를 쌓을 필요는 없어보이나, 나중에 그래프 그릴때 보고 수정합시다.
        return train_untrain_error

    def normal_abnormal_classifier(self, data):
        # LSTM_AutoEncoder를 활용한 Normal / Abnormal 분류
        normal_abnormal_prediction = self.normal_abnormal_lstm_autoencoder.predict(data) # 예측
        # normal_abnormal_result = deque(maxlen=1)
        # normal_abnormal_result.append(normal_abnormal_prediction[0]) # 예측결과 저장
        normal_abnormal_error = np.mean(np.power(self.flatten(data) - self.flatten(normal_abnormal_prediction), 2), axis=1)
        self.normal_abnormal_reconstruction_error.append(normal_abnormal_error) # 굳이 데이터를 쌓을 필요는 없어보이나, 나중에 그래프 그릴때 보고 수정합시다.
        return self.normal_abnormal_reconstruction_error

    def abnormal_procedure_classifier(self, data):
        # XGBClassifier를 활용한 Abnormal Procedure 분류
        abnormal_procedure_prediction = self.multiple_xgbclassification.predict(data)
        shap_value = self.explainer.shap_values(data)
        temp1 = pd.DataFrame(shap_value[int(np.argmax(abnormal_procedure_prediction, axis=1))], columns=self.selected_para['0'].tolist())
        temp2 = temp1.sort_values(by=0, ascending=False, axis=1)
        shap_values = temp2[temp2.iloc[:]>0].dropna(axis=1).T
        shap_add_des = shap_values.reset_index()
        col = shap_add_des['index']
        var = [self.selected_para['0'][self.selected_para['0'] == col_].index for col_ in col]
        val_col = [self.selected_para['1'][var_].iloc[0] for var_ in var]
        proba = [(shap_add_des[0][val_num]/sum(shap_add_des[0]))*100 for val_num in range(len(shap_add_des[0]))]
        val_system = [self.selected_para['2'][var_].iloc[0] for var_ in var]
        # val_component = [self.selected_para['3'][var_].iloc[0] for var_ in var]
        shap_add_des['describe'] = val_col
        shap_add_des['probability'] = proba
        shap_add_des['system'] = val_system
        # shap_add_des['component'] = val_component
        self.abnormal_procedure_prediction = abnormal_procedure_prediction # 분류 결과를 다음 LSTM-AutoEncoder로 삽입하기 위함.
        self.abnormal_procedure_result.append(abnormal_procedure_prediction)
        return self.abnormal_procedure_result, self.abnormal_procedure_prediction, shap_add_des, shap_value

    # def abnormal_procedure_classifier_1(self, data):
    #     # XGBClassifier를 활용한 Abnormal Procedure 분류
    #     diagnosis_start_time = time.time()
    #     self.abnormal_procedure_prediction = self.multiple_xgbclassification.predict(data) # Softmax 예측 값 출력
    #     diagnosis_end_time = time.time()
    #     shap_calculate_start_time = time.time()
    #     shap_value = self.explainer.shap_values(data) # Shap_value 출력
    #     shap_calculate_end_time = time.time()
    #
    #     posi_nega_start_time = time.time()
    #     sort_shap_values_positive = [pd.DataFrame(shap_value[senario_num], columns=self.selected_para['0'].tolist()).sort_values(by=0, ascending=False, axis=1) for senario_num in range(16)]
    #     sort_shap_values_negative = [pd.DataFrame(shap_value[senario_num], columns=self.selected_para['0'].tolist()).sort_values(by=0, ascending=True, axis=1) for senario_num in range(16)]
    #     posi_nega_end_time = time.time()
    #
    #     drop_start_time = time.time()
    #     drop_shap_values_positive = [sort_shap_values_positive[senario_num][sort_shap_values_positive[senario_num].iloc[:]>0].dropna(axis=1).T for senario_num in range(16)]
    #     drop_shap_values_negative = [sort_shap_values_negative[senario_num][sort_shap_values_negative[senario_num].iloc[:]<0].dropna(axis=1).T for senario_num in range(16)]
    #     drop_end_time = time.time()
    #
    #     reset_start_time = time.time()
    #     reset_shap_values_positive = [drop_shap_values_positive[senario_num].reset_index() for senario_num in range(16)]
    #     reset_shap_values_negative = [drop_shap_values_negative[senario_num].reset_index() for senario_num in range(16)]
    #     reset_end_time = time.time()
    #
    #     col_start_time = time.time()
    #     column_positive = [reset_shap_values_positive[senario_num]['index'] for senario_num in range(16)]
    #     column_negative = [reset_shap_values_negative[senario_num]['index'] for senario_num in range(16)]
    #     col_end_time = time.time()
    #
    #     var_start_time = time.time()
    #     var_positive = [[self.selected_para['0'][self.selected_para['0'] == col_].index for col_ in column_positive[senario_num]] for senario_num in range(16)]
    #     var_negative = [[self.selected_para['0'][self.selected_para['0'] == col_].index for col_ in column_negative[senario_num]] for senario_num in range(16)]
    #     var_end_time = time.time()
    #
    #     var_col_start_time = time.time()
    #     val_col_positive = [[self.selected_para['1'][var_].iloc[0] for var_ in var_positive[senario_num]] for senario_num in range(16)]
    #     val_col_negative = [[self.selected_para['1'][var_].iloc[0] for var_ in var_negative[senario_num]] for senario_num in range(16)]
    #     var_col_end_time = time.time()
    #
    #     proba_start_time = time.time()
    #     proba_positive = [[(reset_shap_values_positive[senario_num][0][val_num]/sum(reset_shap_values_positive[senario_num][0]))*100 for val_num in range(len(reset_shap_values_positive[senario_num]))] for senario_num in range(16)]
    #     proba_negative = [[(reset_shap_values_negative[senario_num][0][val_num]/sum(reset_shap_values_negative[senario_num][0]))*100 for val_num in range(len(reset_shap_values_negative[senario_num]))] for senario_num in range(16)]
    #     proba_end_time = time.time()
    #
    #     sys_start_time = time.time()
    #     val_system_positive = [[self.selected_para['2'][var_].iloc[0] for var_ in var_positive[senario_num]] for senario_num in range(16)]
    #     val_system_negative = [[self.selected_para['2'][var_].iloc[0] for var_ in var_negative[senario_num]] for senario_num in range(16)]
    #     sys_end_time = time.time()
    #     # val_component = [self.selected_para['3'][var_].iloc[0] for var_ in var]
    #     loop_start_time = time.time()
    #     for senario_num in range(16):
    #         # Positive
    #         reset_shap_values_positive[senario_num]['describe'] = val_col_positive[senario_num]
    #         reset_shap_values_positive[senario_num]['probability'] = proba_positive[senario_num]
    #         reset_shap_values_positive[senario_num]['system'] = val_system_positive[senario_num]
    #         # Negative
    #         reset_shap_values_negative[senario_num]['describe'] = val_col_negative[senario_num]
    #         reset_shap_values_negative[senario_num]['probability'] = proba_negative[senario_num]
    #         reset_shap_values_negative[senario_num]['system'] = val_system_negative[senario_num]
    #     loop_end_time = time.time()
    #
    #     save_start_time = time.time()
    #     [self.shap_result_postive[key].append(reset_shap_values_positive[senario_num]) for senario_num, key in enumerate(self.shap_result_postive.keys())]
    #     [self.shap_result_negative[key].append(reset_shap_values_negative[senario_num]) for senario_num, key in enumerate(self.shap_result_negative.keys())]
    #     save_end_time = time.time()
    #
    #     print('진단 시간 : ', diagnosis_end_time-diagnosis_start_time)
    #     print('shap 계산시간 : ', shap_calculate_end_time-shap_calculate_start_time)
    #     print('posi_nega 계산시간 : ', posi_nega_end_time-posi_nega_start_time)
    #     print('drop 계산시간 : ', drop_end_time-drop_start_time)
    #     print('reset 계산시간 : ', reset_end_time-reset_start_time)
    #     print('col 계산시간 : ', col_end_time-col_start_time)
    #     print('var 계산시간 : ', var_end_time-var_start_time)
    #     print('val_col 계산시간 : ', var_col_end_time-var_start_time)
    #     print('proba 계산시간 : ', proba_end_time-proba_start_time)
    #     print('sys 계산시간 : ', sys_end_time-sys_start_time)
    #     print('loop 계산시간 : ', loop_end_time-loop_start_time)
    #     print('save 계산시간 : ', save_end_time-save_start_time)
    #     # shap_add_des['component'] = val_component
    #     # self.abnormal_procedure_prediction = abnormal_procedure_prediction # 분류 결과를 다음 LSTM-AutoEncoder로 삽입하기 위함.
    #     # self.abnormal_procedure_result.append(abnormal_procedure_prediction)
    #     return self.abnormal_procedure_prediction, self.shap_result_postive, self.shap_result_negative

    def abnormal_procedure_classifier_0(self, data):
        self.abnormal_procedure_prediction = self.multiple_xgbclassification.predict(data) # Softmax 예측 값 출력
        self.shap_value = self.explainer.shap_values(data)  # Shap_value 출력
        return self.abnormal_procedure_prediction

    def abnormal_procedure_classifier_1(self):
        diagnosis_convert_text = {0: 'Normal', 1: 'Ab21_01', 2: 'Ab21_02', 3: 'Ab20_04', 4: 'Ab15_07', 5: 'Ab15_08',
                         6: 'Ab63_04', 7: 'Ab63_02', 8: 'Ab21_12',9: 'Ab19_02', 10: 'Ab21_11', 11: 'Ab23_03',
                         12: 'Ab60_02', 13: 'Ab59_02', 14: 'Ab23_01', 15: 'Ab23_06'}
        sort_shap_values_positive = pd.DataFrame(self.shap_value[np.argmax(self.abnormal_procedure_prediction, axis=1)[0]], columns=self.selected_para['0'].tolist()).sort_values(by=0, ascending=False, axis=1)
        drop_shap_values_positive = sort_shap_values_positive[sort_shap_values_positive.iloc[:]>0].dropna(axis=1).T
        reset_shap_values_positive = drop_shap_values_positive.reset_index()
        column_positive = reset_shap_values_positive['index']
        var_positive = [self.selected_para['0'][self.selected_para['0'] == col_].index for col_ in column_positive]
        val_col_positive = [self.selected_para['1'][var_].iloc[0] for var_ in var_positive]
        proba_positive = [(reset_shap_values_positive[0][val_num]/sum(reset_shap_values_positive[0]))*100 for val_num in range(len(reset_shap_values_positive))]
        val_system_positive = [self.selected_para['2'][var_].iloc[0] for var_ in var_positive]
        reset_shap_values_positive['describe'] = val_col_positive
        reset_shap_values_positive['probability'] = proba_positive
        reset_shap_values_positive['system'] = val_system_positive
        self.shap_result_postive[diagnosis_convert_text[np.argmax(self.abnormal_procedure_prediction, axis=1)[0]]].append(reset_shap_values_positive)
        return self.shap_result_postive

    def abnormal_procedure_classifier_2(self, scenario):
        int_convert_text = {0:'Normal', 1:'Ab21_01', 2:'Ab21_02', 3:'Ab20_04', 4:'Ab15_07', 5:'Ab15_08',
                                  6:'Ab63_04', 7:'Ab63_02', 8:'Ab21_12', 9:'Ab19_02', 10:'Ab21_11', 11:'Ab23_03',
                                  12:'Ab60_02', 13:'Ab59_02', 14:'Ab23_01', 15:'Ab23_06'}

        sort_shap_values_negative = pd.DataFrame(self.shap_value[scenario], columns=self.selected_para['0'].tolist()).sort_values(by=0, ascending=True, axis=1)
        drop_shap_values_negative = sort_shap_values_negative[sort_shap_values_negative.iloc[:]<0].dropna(axis=1).T
        reset_shap_values_negative = drop_shap_values_negative.reset_index()
        column_negative = reset_shap_values_negative['index']
        val_negative = [self.selected_para['0'][self.selected_para['0'] == col_].index for col_ in column_negative]
        val_col_negative = [self.selected_para['1'][var_].iloc[0] for var_ in val_negative]
        proba_negative = [(reset_shap_values_negative[0][val_num]/sum(reset_shap_values_negative[0]))*100 for val_num in range(len(reset_shap_values_negative))]
        val_system_negative = [self.selected_para['2'][var_].iloc[0] for var_ in val_negative]
        reset_shap_values_negative['describe'] = val_col_negative
        reset_shap_values_negative['probability'] = proba_negative
        reset_shap_values_negative['system'] = val_system_negative
        self.shap_result_negative[int_convert_text[scenario]].append(reset_shap_values_negative)
        return self.shap_result_negative

    def abnormal_procedure_verification(self, data): # 만약 abnormal 예측이 실패한다면??
        global verif_prediction, verif_threshold
        if np.argmax(self.abnormal_procedure_prediction, axis=1) == 0:
            verif_prediction = self.normal_abnormal_lstm_autoencoder.predict(data)
            verif_threshold = 0.00022733
        elif np.argmax(self.abnormal_procedure_prediction, axis=1) == 1:
            verif_prediction = self.ab_21_01_lstmautoencoder.predict(data)
            verif_threshold = 0.00712891
        elif np.argmax(self.abnormal_procedure_prediction, axis=1) == 2:
            verif_prediction = self.ab_21_02_lstmautoencoder.predict(data)
            verif_threshold = 0.01140293
        elif np.argmax(self.abnormal_procedure_prediction, axis=1) == 3:
            verif_prediction = self.ab_20_04_lstmautoencoder.predict(data)
            verif_threshold = 0.00121183
        elif np.argmax(self.abnormal_procedure_prediction, axis=1) == 4:
            verif_prediction = self.ab_15_07_lstmautoencoder.predict(data)
            verif_threshold = 0.00048918
        elif np.argmax(self.abnormal_procedure_prediction, axis=1) == 5:
            verif_prediction = self.ab_15_08_lstmautoencoder.predict(data)
            verif_threshold = 0.00454363
        elif np.argmax(self.abnormal_procedure_prediction, axis=1) == 6:
            verif_prediction = self.ab_63_04_lstmautoencoder.predict(data)
            verif_threshold = 0.00015933
        elif np.argmax(self.abnormal_procedure_prediction, axis=1) == 7:
            verif_prediction = self.ab_63_02_lstmautoencoder.predict(data)
            verif_threshold =0.00376744
        elif np.argmax(self.abnormal_procedure_prediction, axis=1) == 8:
            verif_prediction = self.ab_21_12_lstmautoencoder.predict(data)
            verif_threshold = 0.00236955
        elif np.argmax(self.abnormal_procedure_prediction, axis=1) == 9:
            verif_prediction = self.ab_19_02_lstmautoencoder.predict(data)
            verif_threshold = 0.00610265
        elif np.argmax(self.abnormal_procedure_prediction, axis=1) == 10:
            verif_prediction = self.ab_21_11_lstmautoencoder.predict(data)
            verif_threshold = 0.00265125
        elif np.argmax(self.abnormal_procedure_prediction, axis=1) == 11:
            verif_prediction = self.ab_23_03_lstmautoencoder.predict(data)
            verif_threshold = 0.00017211
        elif np.argmax(self.abnormal_procedure_prediction, axis=1) == 12:
            verif_prediction = self.ab_60_02_lstmautoencoder.predict(data)
            verif_threshold = 0.00370811
        elif np.argmax(self.abnormal_procedure_prediction, axis=1) == 13:
            verif_prediction = self.ab_59_02_lstmautoencoder.predict(data)
            verif_threshold = 0.00424398
        elif np.argmax(self.abnormal_procedure_prediction, axis=1) == 14:
            verif_prediction = self.ab_23_01_lstmautoencoder.predict(data)
            verif_threshold = 0.00314694
        elif np.argmax(self.abnormal_procedure_prediction, axis=1) == 15:
            verif_prediction = self.ab_23_06_lstmautoencoder.predict(data)
            verif_threshold = 0.00584512
        self.abnormal_verif_error = np.mean(np.power(self.flatten(data) - self.flatten(verif_prediction), 2), axis=1)
        # self.abnormal_verif_reconstruction_error.append(self.abnormal_verif_error)
        return verif_threshold, self.abnormal_verif_error



