import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
from collections import deque
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class Data_module:
    def __init__(self):
        self.temp = deque(maxlen=10)
        self.scaler = MinMaxScaler()
        self.scaler_lightgbm = MinMaxScaler()

    def load_data(self, file_name):
        with open(f'./DataBase/CNS_db/pkl/{file_name}', 'rb') as f:
            db = pickle.load(f)
        self.db = db
        self.check_db = self.db
        print(f'{file_name} 불러오기 완료')
        return self.db, self.check_db # test_db의 return 목적 : 실행파일에서 활용할 예정임.

    def data_processing(self):
        selected_para = pd.read_csv('./DataBase/Final_parameter.csv')
        selected_lightgbm = pd.read_csv('./DataBase/Final_parameter_200825.csv')
        min_value = pd.read_csv('min_value_final.csv')
        min_lightgbm = pd.read_csv('min_value_137_final.csv')
        max_value = pd.read_csv('max_value_final.csv')
        max_lightgbm = pd.read_csv('max_value_137_final.csv')
        self.scaler.fit([min_value['0'], max_value['0']])
        self.scaler_lightgbm.fit([min_lightgbm['0'], max_lightgbm['0']])
        self.db_para = self.db[selected_para['0'].tolist()]
        self.db_para_lightgbm = self.db[selected_lightgbm['0'].tolist()]
        self.scaled_data = self.scaler.transform(self.db_para)
        self.lgb_data = self.scaler_lightgbm.transform(self.db_para_lightgbm)
        return self.db_para, self.scaled_data, self.lgb_data

    def load_real_data(self, row): # 현재는 CNS와 연동이 되어있지 않아 임의로 실시간 데이터를 생성하는 목적으로 구성함.
        self.temp.append(self.scaled_data[row]) # LSTM용 데이터
        if len(self.temp) < 10:
            print('Not 10 Stack')
        else:
            data = self.temp
            return data

    def load_scaled_data(self, row):
        dim2 = deque(maxlen=1)
        dim2.append(self.lgb_data[row])
        return dim2

    def load_real_check_data(self, row): # Check data는 Min-Max Scaler 사용 X
        check_data = deque(maxlen=1)
        check_parameter = deque(maxlen=1)
        check_data.append(self.check_db.iloc[row])
        check_parameter.append(self.check_db.iloc[row-1:row+1])
        # check_data = pd.DataFrame(check_data)
        check_parameter = pd.DataFrame(check_parameter[0])
        return check_data, check_parameter