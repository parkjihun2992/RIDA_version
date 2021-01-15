import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
from collections import deque
import matplotlib.pyplot as plt
import shap


class Model_module:
    def __init__(self):
        self.train_untrain_reconstruction_error = []
        self.normal_abnormal_reconstruction_error = []
        self.abnormal_procedure_result = []
        self.abnormal_verif_reconstruction_error = []
        self.selected_para = pd.read_csv('./DataBase/Final_parameter.csv')


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
        self.train_untrain_lstm_autoencoder = load_model('model/Train_Untrain_epoch27_[0.00225299]_acc_[0.9724685967462512].h5')
        self.normal_abnormal_lstm_autoencoder = load_model('model/node32_Nor_Ab_epoch157_[0.00022733]_acc_[0.9758567350470185].h5')
        self.multiple_xgbclassification = pickle.load(open('model/Lightgbm_max_depth_15.h5', 'rb'))
        self.explainer = shap.TreeExplainer(self.multiple_xgbclassification, feature_perturbation='tree_path_dependent')
        self.ab_21_01_lstmautoencoder = load_model('model/ab21-01_epoch74_[0.00712891]_acc_[0.9653829136428816].h5')
        self.ab_21_02_lstmautoencoder = load_model('model/ab21-02_epoch70_[0.0075423]_acc_[0.8994749588997183].h5')
        self.ab_20_04_lstmautoencoder = load_model('model/ab20-04_epoch62_[0.00121183]_acc_[0.9278062539244847].h5')
        self.ab_15_07_lstmautoencoder = load_model('model/ab15-07_epoch100_[0.00048918]_acc_[0.9711231231353337].h5')
        self.ab_15_08_lstmautoencoder = load_model('model/ab15-08_epoch97_[0.00454363]_acc_[0.9602876916386659].h5')
        self.ab_63_04_lstmautoencoder = load_model('model/ab63-04_epoch92_[0.00015933]_acc_[0.9953701129330438].h5')
        self.ab_63_02_lstmautoencoder = load_model('model/ab63-02_epoch100_[0.00376744]_acc_[0.8948400650463064].h5')
        self.ab_21_12_lstmautoencoder = load_model('model/ab21-12_epoch40_[0.00236955]_acc_[0.9473147939642043].h5')
        self.ab_19_02_lstmautoencoder = load_model('model/ab19-02_epoch99_[0.0125704]_acc_[0.969733098718478].h5')
        self.ab_21_11_lstmautoencoder = load_model('model/ab21-11_epoch98_[0.00265125]_acc_[0.9907740136695732].h5')
        self.ab_60_02_lstmautoencoder = load_model('model/ab60-02_epoch100_[0.00370811]_acc_[0.989989808320697].h5')
        self.ab_23_03_lstmautoencoder = load_model('model/ab23-03_epoch91_[0.00017211]_acc_[0.9931771647209489].h5')
        self.ab_59_02_lstmautoencoder = load_model('model/ab59-02_epoch98_[0.00424398]_acc_[0.9923899523150299].h5')
        self.ab_23_01_lstmautoencoder = load_model('model/ab23-01_epoch93_[0.00314694]_acc_[0.9407692298522362].h5')
        self.ab_23_06_lstmautoencoder = load_model('model/ab23-06_epoch96_[0.00584512]_acc_[0.8694280893287306].h5')
        return


    def train_untrain_classifier(self, data):
        # LSTM_AutoEncoder를 활용한 Train / Untrain 분류
        train_untrain_prediction = self.train_untrain_lstm_autoencoder.predict(data) # 예측
        # train_untrain_result = deque(maxlen=1)
        # train_untrain_result.append(train_untrain_prediction[0]) # 예측결과 저장
        train_untrain_error = np.mean(np.power(self.flatten(data) - self.flatten(train_untrain_prediction), 2), axis=1)
        self.train_untrain_reconstruction_error.append(train_untrain_error) # 굳이 데이터를 쌓을 필요는 없어보이나, 나중에 그래프 그릴때 보고 수정합시다.
        return self.train_untrain_reconstruction_error

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
        temp2 = temp1.sort_values(by=0, ascending=True, axis=1)
        shap_values = temp2[temp2.iloc[:]>0].dropna(axis=1)
        self.abnormal_procedure_prediction = abnormal_procedure_prediction # 분류 결과를 다음 LSTM-AutoEncoder로 삽입하기 위함.
        self.abnormal_procedure_result.append(abnormal_procedure_prediction)
        return self.abnormal_procedure_result, self.abnormal_procedure_prediction, shap_values

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
            verif_threshold = 0.0075423
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
            verif_threshold = 0.0125704
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
        self.abnormal_verif_reconstruction_error.append(self.abnormal_verif_error)
        return self.abnormal_verif_reconstruction_error, verif_threshold


    # def binary_prediction(self, data, dim2, binary_prediction_each, mse, XGB_prediction_each):
    #     # Binary LSTM을 활용한 분류
    #     binary_prediction = self.binary_LSTM.predict(data)
    #     binary_prediction = binary_prediction[0]
    #     binary_prediction_each.append(binary_prediction)
    #     self.binary_prediction_each = binary_prediction_each
    #     # AutoEncoder를 활용한 분류
    #     AE_error = deque(maxlen=1)
    #     AE_prediction = self.autoencoder.predict(dim2)
    #     AE_error.append(AE_prediction[0])
    #     error = np.mean(np.power(data[0][0] - AE_error, 2), axis=1)
    #     mse.append(error)
    #     self.mse = mse
    #     # XGBClassifier
    #     XGB_prediction = self.XGB.predict(dim2)
    #     XGB_prediction_each.append(XGB_prediction)
    #     self.XGB_prediction_each = XGB_prediction_each
    #     return binary_prediction, self.binary_prediction_each, self.mse, self.XGB_prediction_each
    #
    # def multiple_prediction(self, data, multiple_prediction_each):
    #     multiple_prediction = self.multiple_LSTM.predict(data)
    #     multiple_prediction = multiple_prediction[0]
    #     multiple_prediction_each.append(multiple_prediction)
    #     self.multiple_prediction_each = multiple_prediction_each
    #     return multiple_prediction, self.multiple_prediction_each




