from Data_module import Data_module
from Model_module import Model_module
from Diagnosis_module import Diagnosis_module
from Plot_module import Plot_module
import numpy as np


# Class의 Instance화
data_module = Data_module()
model_module = Model_module()
diagnosis_module = Diagnosis_module()

model_module.load_model()

# 실시간 데이터 처리할 Test_db 지정
test_db = ['normal11.pkl', 'ab20-04_1.pkl', 'ab20-04_3.pkl', 'ab20-04_5.pkl', 'ab20-04_7.pkl', 'ab20-04_9.pkl', 'ab23-03-15.pkl',  'ab23-03-55.pkl',
               'ab59-02-1011.pkl', 'ab59-02-1021.pkl', 'ab59-02-1031.pkl', 'ab59-02-1041.pkl', 'ab60-02_101.pkl', 'ab60-02_103.pkl', 'ab60-02_105.pkl']

for file_name in test_db:
    binary_prediction_each, mse, multiple_prediction_each, time, binary_result, multiple_result = [], [], [], [], [], [] # 알고리즘에서 필요한 데이터를 구축하는데 사용될 빈행렬
    data_time = 0
    db, check_db = data_module.load_data(file_name=file_name) # test_db 불러오기
    scaled_data = data_module.data_processing() # Min-Max o, 2 Dimension
    # check_db = data_module.load_check_data(file_name=file_name)
    # data_module.compare_data_shape(file_name=file_name)
    for line in range(np.shape(db)[0]):
        data_time +=1
        time.append(data_time)
        data = np.array([data_module.load_real_data(row=line)])
        if np.shape(data) == (1,10,94):
            print(multiple_result)
            dim2 = np.array(data_module.load_scaled_data(row=line - 9))
            check_data, check_parameter = data_module.load_real_check_data(row=line - 9)
            binary_prediction, binary_prediction_each, mse = model_module.binary_prediction(data=data, dim2=dim2, binary_prediction_each=binary_prediction_each, mse=mse)
            multiple_prediction, multiple_prediction_each = model_module.multiple_prediction(data=data, multiple_prediction_each=multiple_prediction_each)
            diagnosis_module.diagnosis(binary_prediction=binary_prediction, multiple_prediction=multiple_prediction, time=time, row=line, binary_prediction_each=binary_prediction_each, multiple_prediction_each=multiple_prediction_each, mse=mse, check_data=check_data, check_parameter=check_parameter, binary_result=binary_result, multiple_result=multiple_result)
        else:
            pass
