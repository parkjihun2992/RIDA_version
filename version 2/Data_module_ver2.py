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

    def load_data(self, file_name):
        with open(f'./DataBase/CNS_db/pkl/{file_name}', 'rb') as f:
            db = pickle.load(f)
        self.db = db
        self.check_db = self.db
        print(f'{file_name} 불러오기 완료')
        return self.db, self.check_db # test_db의 return 목적 : 실행파일에서 활용할 예정임.

    def data_processing(self):
        # with open('DataBase/scaler.bin', 'rb') as f:
        #     scaler = pickle.load(f)
        selected_para = pd.read_csv('./DataBase/Final_parameter.csv')
        min_value = pd.read_csv('min_value_final.csv')
        max_value = pd.read_csv('max_value_final.csv')
        self.scaler.fit([min_value['0'], max_value['0']])
        self.db_para = self.db[selected_para['0'].tolist()]
        # self.scaled_data = self.scaler.fit_transform(self.db_para)
        self.scaled_data = self.scaler.transform(self.db_para)
        # self.scaled_data = np.array(self.db_para)
        return self.db_para, self.scaled_data

    # def load_check_data(self, file_name):
    #     self.check_db = self.db[file_name]['train_x_db'][:-9] # test_db와 check_data의 shape을 일치하게 하기 위함. 추후에 데이터 shape 확인 필요함.
    #     print(f'{file_name}에 대한 Check_db 구축 완료')
    #     return self.check_db

    # def load_dim3(self, file_name, row):
    #     self.temp.append(self.scaled_data[row])
    #     if len(self.temp) < 10:
    #         print('Not 10 Stack')
    #     else:
    #         self.dim3 = self.temp
    #     return self.dim3

    # def compare_data_shape(self, file_name):
    #     if np.shape(self.dim3)[0] == np.shape(self.check_db)[0]:
    #         if np.shape(self.dim3)[0] < 10:
    #             print('Not 10 Stack')
    #         else:
    #             print('DB 및 Check_db의 Shape이 동일합니다.')
    #     else:
    #         print('Error : DB 및 Check_db의 Shape이 동일하지 않습니다. 다시 한번 확인해주세요!')

    def load_real_data(self, row): # 현재는 CNS와 연동이 되어있지 않아 임의로 실시간 데이터를 생성하는 목적으로 구성함.
        self.temp.append(self.scaled_data[row]) # LSTM용 데이터
        if len(self.temp) < 10:
            print('Not 10 Stack')
        else:
            data = self.temp
            # print(np.shape(data))
            return data

    def load_scaled_data(self, row): # AutoEncoder용 데이터
        dim2 = deque(maxlen=1)
        dim2.append(self.scaled_data[row])
        return dim2

    def load_real_check_data(self, row): # Check data는 Min-Max Scaler 사용 X
        check_data = deque(maxlen=1)
        check_parameter = deque(maxlen=1)
        check_data.append(self.check_db.iloc[row])
        check_parameter.append(self.check_db.iloc[row:row+2])
        check_data = pd.DataFrame(check_data)
        check_parameter = pd.DataFrame(check_parameter[0])
        return check_data, check_parameter


