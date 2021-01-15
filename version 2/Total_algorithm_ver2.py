from Data_module import Data_module
from Model_module import Model_module
from Diagnosis_module import Diagnosis_module
from Plot_module import Plot_module
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Class의 Instance화
# data_module = Data_module()

diagnosis_module = Diagnosis_module()
model_module = Model_module()
model_module.load_model()


threshold_train_untrain = 0.00225299
threshold_normal_abnormal = 0.00022733

# 실시간 데이터 처리할 Test_db 지정
# test_db = ['ab21-12_16.pkl', 'ab15-07_1016.pkl', 'ab15-07_1035.pkl', 'ab15-08_1063.pkl', 'ab15-08_1089.pkl', 'ab19-02_01 (04;42트립).pkl', 'ab19-02_44(2;57trip).pkl', 'ab19-02-6(트립x).pkl', 'ab20-01_99(auto까지).pkl', 'ab20-04_8.pkl', 'ab20-04_13.pkl', 'ab21-01_158.pkl', 'ab21-01_168.pkl', 'ab21-02_132(Trip).pkl', 'ab21-02_150.pkl', 'ab21-11_24(27;32트립).pkl', 'ab21-11_88(4;08트립).pkl', 'ab21-12_16.pkl', 'ab21-12_38 (05;28트립).pkl', 'ab23-01_10024(01;02trip).pkl', 'ab23-01_30024(1;00_trip).pkl', 'ab23-03-16.pkl', 'ab23-03-95.pkl', 'ab23-06_10004(03;17_trip).pkl', 'ab23-06_30032(52s trip).pkl', 'ab59-02-1013.pkl', 'ab59-02-1073.pkl', 'ab60-02_57.pkl', 'ab60-02_210.pkl', 'ab63-02_3.pkl', 'ab63-03-1.pkl', 'ab63-04_124.pkl', 'ab63-04_335.pkl', 'ab64-03-02(트립).pkl', 'ab80-02_23.pkl', 'normal0.pkl']
test_db = ['ab21-01_165.pkl']

for file_name in test_db:
    Model_module() # model module 내의 빈행렬 초기화
    time = [] # 그래프에서 활용될 시간을 구축
    data_time = 0
    data_module = Data_module()
    db, check_db = data_module.load_data(file_name=file_name) # test_db 불러오기
    scaled_data = data_module.data_processing() # Min-Max o, 2 Dimension
    # check_db = data_module.load_check_data(file_name=file_name)
    # data_module.compare_data_shape(file_name=file_name)
    for line in range(np.shape(db)[0]):
        data_time +=1
        time.append(data_time)
        data = np.array([data_module.load_real_data(row=line)])
        if np.shape(data) == (1,10,46):
            # print(multiple_result)
            dim2 = np.array(data_module.load_scaled_data(row=line - 9)) # 2차원 scale
            check_data, check_parameter = data_module.load_real_check_data(row=line - 9)
            train_untrain_reconstruction_error = model_module.train_untrain_classifier(data=data)
            normal_abnormal_reconstruction_error = model_module.normal_abnormal_classifier(data=data)
            abnormal_procedure_result, abnormal_procedure_prediction, shap_values = model_module.abnormal_procedure_classifier(data= dim2)
            abnormal_verif_reconstruction_error, verif_threshold = model_module.abnormal_procedure_verification(data = data)
            diagnosis_module.diagnosis(time=time, row=line, check_data=check_data, check_parameter=check_parameter, threshold_train_untrain=threshold_train_untrain, threshold_normal_abnormal=threshold_normal_abnormal, train_untrain_reconstruction_error=train_untrain_reconstruction_error, normal_abnormal_reconstruction_error=normal_abnormal_reconstruction_error, abnormal_procedure_result=abnormal_procedure_result, shap_values=shap_values, abnormal_verif_reconstruction_error=abnormal_verif_reconstruction_error, verif_threshold=verif_threshold)
        else:
            plt.close()
