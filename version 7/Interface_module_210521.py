import numpy as np
import pandas as pd
import sys
import pickle
import pyqtgraph
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtTest import *
from model_module_thread_210210 import Model_module
from data_module_thread_210210 import Data_module
import time


# from Sub_widget import another_result_explain


class Worker(QObject):
    # Signal을 보낼 그릇을 생성 #############
    train_value = pyqtSignal(object)
    # nor_ab_value = pyqtSignal(object)
    procedure_value = pyqtSignal(object)
    procedure_list = pyqtSignal(object)
    verif_value = pyqtSignal(object)
    timer = pyqtSignal(object)
    symptom_db = pyqtSignal(object)
    shap = pyqtSignal(object)
    plot_db = pyqtSignal(object, object)
    display_ex = pyqtSignal(object, object, object)
    another_shap = pyqtSignal(object, object)
    another_shap_table = pyqtSignal(object)

    ## Log 수집용 그릇 생성 #############
    log_test = pyqtSignal(object)
    ##########################################
    def __init__(self, loaded_pkl):
        super().__init__()
        self.loaded_pkl = loaded_pkl  # TODO 업데이트된 내용
        # {Valname: pkl, ... } 호출  self.loaded_pkl['Valname']
        # 해당 변수는 Main -> Mainwindow -> Worker 까지 연결됨.

    def generate_db(self):
        self.diagnosis_convert_text = {0: 'Normal', 1: 'Ab21_01', 2: 'Ab21_02', 3: 'Ab20_04', 4: 'Ab15_07', 5: 'Ab15_08',
                                       6: 'Ab63_04', 7: 'Ab63_02', 8: 'Ab21_12', 9: 'Ab19_02', 10: 'Ab21_11', 11: 'Ab23_03',
                                       12: 'Ab60_02', 13: 'Ab59_02', 14: 'Ab23_01', 15: 'Ab23_06'}
        self.liner = []
        self.plot_data = []
        self.logging_db = {'time':[], 'train':[], 'scenario':{'name':[], 'probability':[]}, 'success':[]}
        self.compare_data = {dict_keys: [] for dict_keys in self.loaded_pkl.keys()}  # 초기에 선언함. # TODO ...

        test_db = input('구현할 시나리오를 입력해주세요 : ')
        print(f'입력된 시나리오 : {test_db}를 실행합니다.')
        self.data_module = Data_module()
        db, check_db = self.data_module.load_data(file_name=test_db)  # test_db 불러오기
        self.data_module.data_processing()  # Min-Max o, 2 Dimension
        for line in range(np.shape(db)[0]):
            started_time = time.time()
            QTest.qWait(0.0001)
            print(np.shape(db)[0], line)
            data = np.array([self.data_module.load_real_data(row=line)]) # 3차원 시계열 데이터 생성
            self.liner.append(line)
            check_data, check_parameter = self.data_module.load_real_check_data(row=line)
            self.plot_data.append(check_data[0])
            # 코드 개선
            try:
                [self.compare_data[compare_data_key].append(self.loaded_pkl[compare_data_key].iloc[line]) for compare_data_key in self.compare_data.keys()]
            except:
                pass
            if np.shape(data) == (1, 10, 46):
                dim2 = np.array(self.data_module.load_scaled_data(row=line))  # 2차원 scale
                train_untrain_result = model_module.train_untrain_classifier(data=data)
                abnormal_procedure_prediction, diagnosed_scenario = model_module.abnormal_procedure_classifier_0(data=dim2)
                diagnosed_shap_result_abs= model_module.abnormal_procedure_classifier_1()
                verif_result = model_module.abnormal_procedure_verification(data=data)
                calculate_time = time.time()
                self.logging_db['time'].append(line)
                self.logging_db['train'].append(train_untrain_result)
                self.logging_db['scenario']['name'].append(diagnosed_scenario)
                self.logging_db['scenario']['probability'].append(round(abnormal_procedure_prediction[0][diagnosed_scenario]*100, 2))
                self.logging_db['success'].append(verif_result)

                # Signal Emit
                self.train_value.emit(train_untrain_result)
                self.procedure_value.emit(diagnosed_scenario)
                self.procedure_list.emit([diagnosed_scenario, abnormal_procedure_prediction])
                self.verif_value.emit(verif_result)
                self.timer.emit([line, check_data[0]['QPROREL']]) # Time & Power variable 전달
                self.symptom_db.emit([diagnosed_scenario, check_parameter])
                self.shap.emit(diagnosed_shap_result_abs[self.diagnosis_convert_text[diagnosed_scenario]][0])
                self.plot_db.emit([self.liner, self.plot_data], diagnosed_scenario)
                self.display_ex.emit(diagnosed_shap_result_abs[self.diagnosis_convert_text[diagnosed_scenario]][0], [self.liner, self.plot_data], self.compare_data['Normal'])
                self.another_shap.emit([self.liner, self.plot_data], self.compare_data)
                self.another_shap_table.emit(line)
                self.log_test.emit(self.logging_db)

                loop_end_time = time.time()
                print('1 loop 계산시간 : ', calculate_time-started_time)
                print('1 loop Emit time : ', loop_end_time-calculate_time)
                print('1 loop 소요시간 : ', loop_end_time - started_time)



class AlignDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super(AlignDelegate, self).initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter


class Mainwindow(QWidget):
    @pyqtSlot(object)
    def time_display(self, display_variable):
        # display_variable[0] : time, display_variable[1] : QPROREL 변수
        self.time_label.setText("<html><img src='./icon/time_icon.png' height='20' width='20'></html>"+f'<b> Time :<b/> {display_variable[0]} sec')
        # self.time_label.setAlignment(Qt.AlignCenter)
        self.power_label.setText(f'Power : {round(display_variable[1] * 100, 2)}%')
        self.power_label.setIcon(QIcon('./icon/power_icon.png'))
        if round(display_variable[1] * 100, 2) < 95:
            self.power_label.setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: red;')
        else:
            self.power_label.setStyleSheet('color : black;' 'background-color: light gray;')

    @pyqtSlot(object)
    def Determine_train(self, train_untrain_result):
        # train_untrain_result = 0 : trained condition / train_untrain_result = 1 : untrained condition
        if train_untrain_result == 0: # trained condition
            self.trained_label.setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: green;')
            self.Untrained_label.setStyleSheet('color : black;' 'background-color: light gray;')
        elif train_untrain_result == 1: # untrained condition
            self.Untrained_label.setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: red;')
            self.trained_label.setStyleSheet('color : black;' 'background-color: light gray;')

    @pyqtSlot(object)
    def Determine_abnormal(self, abnormal_diagnosis):
        if abnormal_diagnosis == 0:  # 정상상태
            self.normal_label.setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: green;')
            self.abnormal_label.setStyleSheet('color : black;' 'background-color: light gray;')
        else:  # 비정상상태
            self.abnormal_label.setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: red;')
            self.normal_label.setStyleSheet('color : black;' 'background-color: light gray;')

    @pyqtSlot(object)
    def Determine_procedure(self, abnormal_procedure_result):
        # abnormal_procedure_result[0]: 진단된 시나리오 번호
        # abnormal_procedure_result[1]: 전체 시나리오 진단 확률
        text_set = {0: {'name': 'Normal', 'contents': 'Normal'}, 1: {'name': 'Ab21_01', 'contents': '가압기 압력 채널 고장 "고"'}, 2: {'name': 'Ab21_02', 'contents': '가압기 압력 채널 고장 "저"'},
                    3: {'name': 'Ab20_04', 'contents': '가압기 수위 채널 고장 "저"'}, 4: {'name': 'Ab15_07', 'contents': '증기발생기 수위 채널 고장 "저"'}, 5: {'name': 'Ab15_08', 'contents': '증기발생기 수위 채널 고장 "고"'},
                    6: {'name': 'Ab63_04', 'contents': '제어봉 낙하'}, 7: {'name': 'Ab63_02', 'contents': '제어봉의 계속적인 삽입'}, 8: {'name': 'Ab21_12', 'contents': 'Pressurizer PORV opening'},
                    9: {'name': 'Ab19_02', 'contents': '가압기 안전밸브 고장'}, 10: {'name': 'Ab21_11', 'contents': 'Opening of PRZ spray valve'}, 11: {'name': 'Ab23_03', 'contents': '1차기기 냉각수 계통으로 누설 "CVCS->CCW"'},
                    12: {'name': 'Ab60_02', 'contents': '재생열교환기 전단부위 파열'}, 13: {'name': 'Ab59_02', 'contents': '충전수 유량조절밸브 후단 누설'}, 14: {'name': 'Ab23_01', 'contents': '1차기기 냉각수 계통으로 누설 "RCS->CCW"'},
                    15: {'name': 'Ab23_06', 'contents': 'Steam generator tube rupture'}}
        self.num_procedure.setText(text_set[abnormal_procedure_result[0]]['name'])
        self.num_scnario.setText(f'{text_set[abnormal_procedure_result[0]]["contents"]} [{round(abnormal_procedure_result[1][0][abnormal_procedure_result[0]]*100, 2)}%]' )

    @pyqtSlot(object)
    def verifit_result(self, verif_result):
        if verif_result == 0:  # diagnosis success
            self.success_label.setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: green;')
            self.failure_label.setStyleSheet('color : black;' 'background-color: light gray;')
        elif verif_result == 1:  # diagnosis failure
            self.failure_label.setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: red;')
            self.success_label.setStyleSheet('color : black;' 'background-color: light gray;')

    def procedure_satisfaction_0(self, db):
        self.symptom_name.setText('Diagnosis Result : Normal → Symptoms : 0')
        self.symptom1.setText('')
        self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom2.setText('')
        self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom3.setText('')
        self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom4.setText('')
        self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom5.setText('')
        self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom6.setText('')
        self.symptom6.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")

    def procedure_satisfaction_1(self, db):
        self.symptom_name.setText('Diagnosis Result : Ab21-01 Pressurizer pressure channel failure "High" → Symptoms : 6')

        self.symptom1.setText("채널 고장으로 인한 가압기 '고' 압력 지시")
        if db.iloc[1]['PPRZN'] > db.iloc[1]['CPPRZH']:
            self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

        self.symptom2.setText("가압기 살수밸브 '열림' 지시")
        if db.iloc[1]['BPRZSP'] > 0:
            self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

        self.symptom3.setText("가압기 비례전열기 꺼짐")
        if db.iloc[1]['QPRZP'] == 0:
            self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

        self.symptom4.setText("가압기 보조전열기 꺼짐")
        if db.iloc[1]['QPRZB'] == 0:
            self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

        self.symptom5.setText("실제 가압기 '저' 압력 지시")
        if db.iloc[1]['PPRZ'] < db.iloc[1]['CPPRZL']:
            self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

        self.symptom6.setText("가압기 PORV 차단밸브 닫힘")
        if db.iloc[1]['BHV6'] == 0:
            self.symptom6.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

    def procedure_satisfaction_2(self, db):
        self.symptom_name.setText('진단 : Ab21-02 가압기 압력 채널 고장 "저" → 증상 : 5')

        self.symptom1.setText("채널 고장으로 인한 가압기 '저' 압력 지시")
        if db.iloc[1]['PPRZN'] < db.iloc[1]['CPPRZL']:
            self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

        self.symptom2.setText('가압기 저압력으로 인한 보조 전열기 켜짐 지시 및 경보 발생')
        if (db.iloc[1]['PPRZN'] < db.iloc[1]['CQPRZB']) and (db.iloc[1]['KBHON'] == 1):
            self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

        self.symptom3.setText("실제 가압기 '고' 압력 지시")
        if db.iloc[1]['PPRZ'] > db.iloc[1]['CPPRZH']:
            self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

        self.symptom4.setText('가압기 PORV 열림 지시 및 경보 발생')
        if db.iloc[1]['BPORV'] > 0:
            self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

        self.symptom5.setText('실제 가압기 압력 감소로 가압기 PORV 닫힘')  # 가압기 압력 감소에 대해 해결해야함.
        if db.iloc[1]['BPORV'] == 0 and (db.iloc[0]['PPRZ'] > db.iloc[1]['PPRZ']):
            self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

    def procedure_satisfaction_3(self, db):
        self.symptom_name.setText('진단 : Ab20-04 가압기 수위 채널 고장 "저" → 증상 : 5')

        self.symptom1.setText("채널 고장으로 인한 가압기 '저' 수위 지시")
        if db.iloc[1]['ZINST63'] < 17:  # 나중에 다시 확인해야함.
            self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")


        self.symptom2.setText('"LETDN HX OUTLET FLOW LOW" 경보 발생')
        if db.iloc[1]['UNRHXUT'] > db.iloc[1]['CULDHX']:
            self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")


        self.symptom3.setText('"CHARGING LINE FLOW HI/LO" 경보 발생')
        if (db.iloc[1]['WCHGNO'] < db.iloc[1]['CWCHGL']) or (db.iloc[1]['WCHGNO'] > db.iloc[1]['CWCHGH']):
            self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")


        self.symptom4.setText('충전 유량 증가')
        if db.iloc[0]['WCHGNO'] < db.iloc[1]['WCHGNO']:
            self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")


        self.symptom5.setText('건전한 수위지시계의 수위 지시치 증가')
        if db.iloc[0]['ZPRZNO'] < db.iloc[1]['ZPRZNO']:
            self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

    def procedure_satisfaction_4(self, db):
        self.symptom_name.setText('진단 : Ab15-07 증기발생기 수위 채널 고장 "저" → 증상 : ')
        self.symptom1.setText('증기발생기 수위 "저" 경보 발생')
        if db.iloc[1]['ZINST78'] * 0.01 < db.iloc[1]['CZSGW'] or db.iloc[1]['ZINST77'] * 0.01 < db.iloc[1]['CZSGW'] or db.iloc[1]['ZINST76'] * 0.01 < db.iloc[1]['CZSGW']:
            self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

        self.symptom2.setText('해당 SG MFCV 열림 방향으로 진행 및 해당 SG 실제 급수유량 증가')

    def procedure_satisfaction_5(self, db):
        self.symptom_name.setText('Diagnosis Result : Normal → Symptoms : 0')
        self.symptom1.setText('')
        self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom2.setText('')
        self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom3.setText('')
        self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom4.setText('')
        self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom5.setText('')
        self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom6.setText('')
        self.symptom6.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")

    def procedure_satisfaction_6(self, db):
        self.symptom_name.setText('Diagnosis Result : Normal → Symptoms : 0')
        self.symptom1.setText('')
        self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom2.setText('')
        self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom3.setText('')
        self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom4.setText('')
        self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom5.setText('')
        self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom6.setText('')
        self.symptom6.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")

    def procedure_satisfaction_7(self, db):
        self.symptom_name.setText('Diagnosis Result : Normal → Symptoms : 0')
        self.symptom1.setText('')
        self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom2.setText('')
        self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom3.setText('')
        self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom4.setText('')
        self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom5.setText('')
        self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom6.setText('')
        self.symptom6.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")

    def procedure_satisfaction_8(self, db):
        self.symptom_name.setText('Diagnosis result : Ab21-12 Pressurizer PORV opening → Symptoms : 5')

        self.symptom1.setText('Pressurizer PORV open indication and alarm')
        if db.iloc[1]['BPORV'] > 0:
            self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

        # self.symptom2.setText('가압기 저압력으로 인한 보조 전열기 켜짐 지시 및 경보 발생')
        self.symptom2.setText('Aux. heater turn on instruction and alarm due to pressurizer low pressure')
        if (db.iloc[1]['PPRZN'] < db.iloc[1]['CQPRZB']) and (db.iloc[1]['KBHON'] == 1):
            self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

        # self.symptom3.setText("가압기 '저' 압력 지시 및 경보 발생")
        self.symptom3.setText("pressurizer 'low' pressure indication and alarm")
        if db.iloc[1]['PPRZ'] < db.iloc[1]['CPPRZL']:
            self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

        self.symptom4.setText("PRT high temperature indication and alarm")
        if db.iloc[1]['UPRT'] > db.iloc[1]['CUPRT']:
            self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

        self.symptom5.setText("PRT high pressure indication and alarm")
        if (db.iloc[1]['PPRT'] - 0.98E5) > db.iloc[1]['CPPRT']:
            self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

        self.symptom6.setText("")
        self.symptom6.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")

    def procedure_satisfaction_9(self, db):
        self.symptom_name.setText('Diagnosis Result : Normal → Symptoms : 0')
        self.symptom1.setText('')
        self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom2.setText('')
        self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom3.setText('')
        self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom4.setText('')
        self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom5.setText('')
        self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom6.setText('')
        self.symptom6.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")

    def procedure_satisfaction_10(self, db):
        self.symptom_name.setText("Diagnosis Result : Ab21-11 Opening of PRZ spray valve  → Symptoms : 4")

        self.symptom1.setText("Pressurizer spray valve 'open' indication and status indicator ON")
        if db.iloc[1]['BPRZSP'] > 0:
            self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

        self.symptom2.setText("Pressurizer auxiliary heater ON indication and alarm")
        if (db.iloc[1]['PPRZN'] < db.iloc[1]['CQPRZB']) and (db.iloc[1]['KBHON'] == 1):
            self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

        self.symptom3.setText("Pressurizer 'low' pressure indication and alarm")
        if db.iloc[1]['PPRZ'] < db.iloc[1]['CPPRZL']:
            self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

        self.symptom4.setText("A sharp increase in the pressurizer water level")  # 급격한 증가에 대한 수정은 필요함 -> 추후 수정
        if db.iloc[0]['ZINST63'] < db.iloc[1]['ZINST63']:
            self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")

        self.symptom5.setText('')
        self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom6.setText('')
        self.symptom6.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")

    def procedure_satisfaction_11(self, db):
        self.symptom_name.setText('Diagnosis Result : CVCS leakage → Symptoms : 2')
        self.symptom1.setText('VCT 수위 감소')
        if db.iloc[0]['ZVCT'] >= db.iloc[1]['ZVCT']:
            self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
        self.symptom2.setText('충전 유량 증가')
        if db.iloc[0]['WCHGNO'] <= db.iloc[1]['WCHGNO']:
            self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
        self.symptom3.setText('')
        self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom4.setText('')
        self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom5.setText('')
        self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom6.setText('')
        self.symptom6.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")

    def procedure_satisfaction_12(self, db):
        self.symptom_name.setText('Diagnosis Result : Normal → Symptoms : 0')
        self.symptom1.setText('')
        self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom2.setText('')
        self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom3.setText('')
        self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom4.setText('')
        self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom5.setText('')
        self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom6.setText('')
        self.symptom6.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")

    def procedure_satisfaction_13(self, db):
        self.symptom_name.setText('Diagnosis Result : Normal → Symptoms : 0')
        self.symptom1.setText('')
        self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom2.setText('')
        self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom3.setText('')
        self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom4.setText('')
        self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom5.setText('')
        self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom6.setText('')
        self.symptom6.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")

    def procedure_satisfaction_14(self, db):
        self.symptom_name.setText('Diagnosis Result : Normal → Symptoms : 0')
        self.symptom1.setText('')
        self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom2.setText('')
        self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom3.setText('')
        self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom4.setText('')
        self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom5.setText('')
        self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom6.setText('')
        self.symptom6.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")

    def procedure_satisfaction_15(self, db):
        self.symptom_name.setText('Diagnosis Result : Ab23-06 Steam generator tube rupture → Symptoms : 0')
        self.symptom1.setText('')
        self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom2.setText('')
        self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom3.setText('')
        self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom4.setText('')
        self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom5.setText('')
        self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")
        self.symptom6.setText('')
        self.symptom6.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : light gray;""}")


    @pyqtSlot(object)
    def procedure_satisfaction(self, symptom_db):
        # symptom_db[0] : classification result [0~15]
        # symptom_db[1] : check_db [2,2222] -> 현시점과 이전시점 비교를 위함.
        # symptom_db[1].iloc[0] : 이전 시점  # symptom_db[1].iloc[1] : 현재 시점
        self.procedure_satisfaction_func = {0:self.procedure_satisfaction_0, 1:self.procedure_satisfaction_1, 2:self.procedure_satisfaction_2, 3:self.procedure_satisfaction_3,
                                       4:self.procedure_satisfaction_4, 5:self.procedure_satisfaction_5, 6:self.procedure_satisfaction_6,
                                       7:self.procedure_satisfaction_7, 8:self.procedure_satisfaction_8, 9:self.procedure_satisfaction_9,
                                       10:self.procedure_satisfaction_10, 11:self.procedure_satisfaction_11, 12:self.procedure_satisfaction_12,
                                       13:self.procedure_satisfaction_13, 14:self.procedure_satisfaction_14, 15:self.procedure_satisfaction_15}

        self.procedure_satisfaction_func[symptom_db[0]](symptom_db[1])


    @pyqtSlot(object)
    def explain_result(self, shap_result):
        '''
        # shap_add_des['variable'] : 변수 이름 / shap_add_des[0] : shap value
        # shap_add_des['describe'] : 변수에 대한 설명 / shap_add_des['probability'] : shap value를 확률로 환산한 값
        '''
        self.tableWidget.clear()
        self.tableWidget.setRowCount(len(shap_result))
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setHorizontalHeaderLabels(["value_name", 'probability', 'describe'])

        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        # header.setSectionResizeMode(3, QHeaderView.Stretch)

        [self.tableWidget.setItem(i, 0, QTableWidgetItem(f"{shap_result['variable'][i]}")) for i in range(len(shap_result['variable']))]
        [self.tableWidget.setItem(i, 1, QTableWidgetItem(f"{round(shap_result['probability'][i], 2)}%")) for i in range(len(shap_result['probability']))]
        [self.tableWidget.setItem(i, 2, QTableWidgetItem(f"{shap_result['describe'][i]}")) for i in range(len(shap_result['describe']))]
        # [self.tableWidget.setItem(i, 3, QTableWidgetItem(f"{shap_result['system'][i]}")) for i in range(len(shap_result['system']))]

        delegate = AlignDelegate(self.tableWidget)
        self.tableWidget.setItemDelegate(delegate)

    def table_control(self):
        if self.detailed_table.isChecked():
            print('Mainwindow table Connect')
            self.worker.shap.connect(self.explain_result)
            self.tableWidget.show()
        # 클릭시 Thread를 통해 신호를 전달하기 때문에 버퍼링이 발생함. 2초 정도? 이 부분은 나중에 생각해서 초기에 불러올지 고민해봐야할듯.
        else:
            print('Mainwindow table DisConnect')
            self.worker.shap.disconnect(self.explain_result)
            self.tableWidget.close()



    @pyqtSlot(object, object)
    def plotting(self, symptom_db, diagnosis_result):
        #  symptom_db[0] : liner : appended time (axis-x) / symptom_db[1].iloc[1] : check_db (:line,2222)[1]
        # diagnosis_result : diagnosis number
        self.plot_1.clear()
        self.plot_2.clear()
        self.plot_3.clear()
        self.plot_4.clear()

        self.plot_1.showGrid(x=True, y=True, alpha=0.3)
        self.plot_2.showGrid(x=True, y=True, alpha=0.3)
        self.plot_3.showGrid(x=True, y=True, alpha=0.3)
        self.plot_4.showGrid(x=True, y=True, alpha=0.3)

        if diagnosis_result == 0:
            self.plot_1.setTitle('')
            self.plot_2.setTitle('')
            self.plot_3.setTitle('')
            self.plot_4.setTitle('')

        elif diagnosis_result == 10:
            self.plot_1.setTitle('PRZ SPRAY VALVE POSITION. (0.0-1.0)')
            self.plot_2.setTitle('PRZ BACK-UP HEATER STATUS(1:ON)')
            self.plot_3.setTitle('PRESSURIZER PRESSURE')
            self.plot_4.setTitle('PRZ LEVEL')

            self.plot_1.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])['BPRZSP'], pen=pyqtgraph.mkPen('k', width=3))
            self.plot_2.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])['KBHON'], pen=pyqtgraph.mkPen('k', width=3))
            self.plot_3.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])['PPRZ'], pen=pyqtgraph.mkPen('k', width=3))
            self.plot_4.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])['ZINST63'], pen=pyqtgraph.mkPen('k', width=3))
        elif diagnosis_result == 11:
            self.plot_1.setTitle('VCT water level')
            self.plot_2.setTitle('Charging flow')
            self.plot_1.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])['ZVCT'], pen=pyqtgraph.mkPen('k', width=3))
            self.plot_2.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])['WCHGNO'], pen=pyqtgraph.mkPen('k', width=3))
        elif diagnosis_result == 15:
            self.plot_1.setTitle('secondary radiation')
            self.plot_1.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])['ZINST102'], pen=pyqtgraph.mkPen('k', width=3))
        else:
            self.plot_1.setTitle('PORV open state')
            self.plot_2.setTitle('Pressurizer pressure')
            self.plot_3.setTitle('PRT temperature')
            self.plot_4.setTitle('PRT pressure')

            self.plot_1.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])['BPORV'], pen=pyqtgraph.mkPen('k', width=3))
            self.plot_2.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])['PPRZN'], pen=pyqtgraph.mkPen('k', width=3))
            self.plot_3.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])['UPRT'], pen=pyqtgraph.mkPen('k', width=3))
            self.plot_4.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])['PPRT'], pen=pyqtgraph.mkPen('k', width=3))


    @pyqtSlot(object, object, object)
    def display_explain(self, shap_result, symptom_db, normal_db):
        '''
        # display_db['variable'] : 변수 이름 / display_db[0] : shap value
        # display_db['describe'] : 변수에 대한 설명 / display_db['probability'] : shap value를 확률로 환산한 값
        # display_db['sign'] : 산출된 변수 별 Shapley value의 부호 값
        #  symptom_db[0] : liner : appended time (axis-x) / symptom_db[1].iloc[1] : check_db (:line,2222)[1]
        '''
        red_range = shap_result[shap_result['probability'] >= 10]
        orange_range = shap_result[[shap_result['probability'][i] < 10 and shap_result['probability'][i] > 1 for i in range(len(shap_result['probability']))]]
        convert_red = {0: self.red1, 1: self.red2, 2: self.red3, 3: self.red4}
        convert_orange = {0: self.orange1, 1: self.orange2, 2: self.orange3, 3: self.orange4, 4: self.orange5, 5: self.orange6, 6: self.orange7, 7: self.orange8, 8: self.orange9, 9: self.orange10, 10: self.orange11, 11: self.orange12}
        '''
        [i for i in range(0, 4)] -> [0,1,2,3]
        [i for i in range(1, 4)] -> [1,2,3]
        [i for i in range(2, 4)] -> [2,3]
        [i for i in range(3, 4)] -> [3]
        [i for i in range(4, 4)] -> []
        ※ red_range에 포함되지 않는 red_button 지역 탐색
        '''
        red_del = [i for i in range(len(red_range), 4)]
        orange_del = [i for i in range(len(orange_range), 12)]

        [convert_red[i].setText(f'{red_range["describe"].iloc[i]} \n[{round(red_range["probability"].iloc[i], 2)}%]') for i in range(len(red_range))]
        [convert_red[i].setText('None\nParameter') for i in red_del]
        for i in range(len(red_range)):
            if red_range['sign'].iloc[i] == '+':
                convert_red[i].setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: red;')
            elif red_range['sign'].iloc[i] == '-':
                convert_red[i].setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: blue;')
        # [convert_red[i].setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: blue;') for i in range(len(red_range))]
        [convert_red[i].setStyleSheet('color : black;' 'background-color: light gray;') for i in red_del]

        [convert_orange[i].setText(f'{orange_range["describe"].iloc[i]} \n[{round(orange_range["probability"].iloc[i], 2)}%]') for i in range(len(orange_range))]
        [convert_orange[i].setText('None\nParameter') for i in orange_del]
        for i in range(len(orange_range)):
            if orange_range['sign'].iloc[i] == '+':
                convert_orange[i].setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: red;')
            elif orange_range['sign'].iloc[i] == '-':
                convert_orange[i].setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: blue;')
        [convert_orange[i].setStyleSheet('color : black;' 'background-color: light gray;') for i in orange_del]

        # 각 Button에 호환되는 Plotting 데이터 구축
        # Red1 Button
        if self.red1.text().split()[0] != 'None':
            if self.red1.isChecked():
                self.red_plot_1.clear()
                self.red_plot_1.setTitle(red_range['describe'].iloc[0])
                self.red_plot_1.addLegend(offset=(-30, 20))
                self.red_plot_1.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[red_range['variable'].iloc[0]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.red_plot_1.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[red_range['variable'].iloc[0]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Red2 Button
        if self.red2.text().split()[0] != 'None':
            if self.red2.isChecked():
                self.red_plot_2.clear()
                self.red_plot_2.setTitle(red_range['describe'].iloc[1])
                self.red_plot_2.addLegend(offset=(-30, 20))
                self.red_plot_2.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[red_range['variable'].iloc[1]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.red_plot_2.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[red_range['variable'].iloc[1]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Red3 Button
        if self.red3.text().split()[0] != 'None':
            if self.red3.isChecked():
                self.red_plot_3.clear()
                self.red_plot_3.setTitle(red_range['describe'].iloc[2])
                self.red_plot_3.addLegend(offset=(-30, 20))
                self.red_plot_3.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[red_range['variable'].iloc[2]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.red_plot_3.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[red_range['variable'].iloc[2]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Red4 Button
        if self.red4.text().split()[0] != 'None':
            if self.red4.isChecked():
                self.red_plot_4.clear()
                self.red_plot_4.setTitle(red_range['describe'].iloc[3])
                self.red_plot_4.addLegend(offset=(-30, 20))
                self.red_plot_4.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[red_range['variable'].iloc[3]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.red_plot_4.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[red_range['variable'].iloc[3]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange1 Button
        if self.orange1.text().split()[0] != 'None':
            if self.orange1.isChecked():
                self.orange_plot_1.clear()
                self.orange_plot_1.setTitle(orange_range['describe'].iloc[0])
                self.orange_plot_1.addLegend(offset=(-30, 20))
                self.orange_plot_1.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[0]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.orange_plot_1.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['variable'].iloc[0]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange2 Button
        if self.orange2.text().split()[0] != 'None':
            if self.orange2.isChecked():
                self.orange_plot_2.clear()
                self.orange_plot_2.setTitle(orange_range['describe'].iloc[1])
                self.orange_plot_2.addLegend(offset=(-30, 20))
                self.orange_plot_2.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[1]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.orange_plot_2.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['variable'].iloc[1]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange3 Button
        if self.orange3.text().split()[0] != 'None':
            if self.orange3.isChecked():
                self.orange_plot_3.clear()
                self.orange_plot_3.setTitle(orange_range['describe'].iloc[2])
                self.orange_plot_3.addLegend(offset=(-30, 20))
                self.orange_plot_3.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[2]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.orange_plot_3.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['variable'].iloc[2]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange4 Button
        if self.orange4.text().split()[0] != 'None':
            if self.orange4.isChecked():
                self.orange_plot_4.clear()
                self.orange_plot_4.setTitle(orange_range['describe'].iloc[3])
                self.orange_plot_4.addLegend(offset=(-30, 20))
                self.orange_plot_4.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[3]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.orange_plot_4.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['variable'].iloc[3]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange5 Button
        if self.orange5.text().split()[0] != 'None':
            if self.orange5.isChecked():
                self.orange_plot_5.clear()
                self.orange_plot_5.setTitle(orange_range['describe'].iloc[4])
                self.orange_plot_5.addLegend(offset=(-30, 20))
                self.orange_plot_5.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[4]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.orange_plot_5.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['variable'].iloc[4]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange6 Button
        if self.orange6.text().split()[0] != 'None':
            if self.orange6.isChecked():
                self.orange_plot_6.clear()
                self.orange_plot_6.setTitle(orange_range['describe'].iloc[5])
                self.orange_plot_6.addLegend(offset=(-30, 20))
                self.orange_plot_6.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[5]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.orange_plot_6.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['variable'].iloc[5]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange7 Button
        if self.orange7.text().split()[0] != 'None':
            if self.orange7.isChecked():
                self.orange_plot_7.clear()
                self.orange_plot_7.setTitle(orange_range['describe'].iloc[6])
                self.orange_plot_7.addLegend(offset=(-30, 20))
                self.orange_plot_7.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[6]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.orange_plot_7.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['variable'].iloc[6]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange8 Button
        if self.orange8.text().split()[0] != 'None':
            if self.orange8.isChecked():
                self.orange_plot_8.clear()
                self.orange_plot_8.setTitle(orange_range['describe'].iloc[7])
                self.orange_plot_8.addLegend(offset=(-30, 20))
                self.orange_plot_8.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[7]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.orange_plot_8.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['variable'].iloc[7]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange9 Button
        if self.orange9.text().split()[0] != 'None':
            if self.orange9.isChecked():
                self.orange_plot_9.clear()
                self.orange_plot_9.setTitle(orange_range['describe'].iloc[8])
                self.orange_plot_9.addLegend(offset=(-30, 20))
                self.orange_plot_9.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[8]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.orange_plot_9.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['variable'].iloc[8]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange10 Button
        if self.orange10.text().split()[0] != 'None':
            if self.orange10.isChecked():
                self.orange_plot_10.clear()
                self.orange_plot_10.setTitle(orange_range['describe'].iloc[9])
                self.orange_plot_10.addLegend(offset=(-30, 20))
                self.orange_plot_10.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[9]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.orange_plot_10.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['variable'].iloc[9]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange11 Button
        if self.orange11.text().split()[0] != 'None':
            if self.orange11.isChecked():
                self.orange_plot_11.clear()
                self.orange_plot_11.setTitle(orange_range['describe'].iloc[10])
                self.orange_plot_11.addLegend(offset=(-30, 20))
                self.orange_plot_11.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[10]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.orange_plot_11.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['variable'].iloc[10]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange12 Button
        if self.orange12.text().split()[0] != 'None':
            if self.orange12.isChecked():
                self.orange_plot_12.clear()
                self.orange_plot_12.setTitle(orange_range['describe'].iloc[11])
                self.orange_plot_12.addLegend(offset=(-30, 20))
                self.orange_plot_12.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[11]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.orange_plot_12.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['variable'].iloc[11]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        [convert_red[i].setCheckable(True) for i in range(4)]
        [convert_orange[i].setCheckable(True) for i in range(12)]

    def red1_plot(self):
        if self.red1.isChecked():
            if self.red1.text().split()[0] != 'None':
                self.red_plot_1.show()

    def red2_plot(self):
        if self.red2.isChecked():
            if self.red2.text().split()[0] != 'None':
                self.red_plot_2.show()

    def red3_plot(self):
        if self.red3.isChecked():
            if self.red3.text().split()[0] != 'None':
                self.red_plot_3.show()

    def red4_plot(self):
        if self.red4.isChecked():
            if self.red4.text().split()[0] != 'None':
                self.red_plot_4.show()

    def orange1_plot(self):
        if self.orange1.isChecked():
            if self.orange1.text().split()[0] != 'None':
                self.orange_plot_1.show()

    def orange2_plot(self):
        if self.orange2.isChecked():
            if self.orange2.text().split()[0] != 'None':
                self.orange_plot_2.show()

    def orange3_plot(self):
        if self.orange3.isChecked():
            if self.orange3.text().split()[0] != 'None':
                self.orange_plot_3.show()

    def orange4_plot(self):
        if self.orange4.isChecked():
            if self.orange4.text().split()[0] != 'None':
                self.orange_plot_4.show()

    def orange5_plot(self):
        if self.orange5.isChecked():
            if self.orange5.text().split()[0] != 'None':
                self.orange_plot_5.show()

    def orange6_plot(self):
        if self.orange6.isChecked():
            if self.orange6.text().split()[0] != 'None':
                self.orange_plot_6.show()

    def orange7_plot(self):
        if self.orange7.isChecked():
            if self.orange7.text().split()[0] != 'None':
                self.orange_plot_7.show()

    def orange8_plot(self):
        if self.orange8.isChecked():
            if self.orange8.text().split()[0] != 'None':
                self.orange_plot_8.show()

    def orange9_plot(self):
        if self.orange9.isChecked():
            if self.orange9.text().split()[0] != 'None':
                self.orange_plot_9.show()

    def orange10_plot(self):
        if self.orange10.isChecked():
            if self.orange10.text().split()[0] != 'None':
                self.orange_plot_10.show()

    def orange11_plot(self):
        if self.orange11.isChecked():
            if self.orange11.text().split()[0] != 'None':
                self.orange_plot_11.show()

    def orange12_plot(self):
        if self.orange12.isChecked():
            if self.orange12.text().split()[0] != 'None':
                self.orange_plot_12.show()

    def show_another_result(self):
        if self.another_classification.isChecked():
            print('Subwindow Connect')
            self.worker.another_shap_table.connect(self.other.show_another_result_table)
            self.worker.another_shap.connect(self.other.show_shap)
            self.other.show()
        else:
            print('Subwindow DisConnect')
            self.worker.another_shap_table.disconnect(self.other.show_another_result_table)
            self.worker.another_shap.disconnect(self.other.show_shap)
            self.other.close()

    def log_system_control(self):
        if self.log_btn.isChecked():
            print('Log system Connect')
            self.worker.log_test.connect(self.log_sys.total_log)
            self.log_sys.show()
        else:
            print('Log system DisConnect')
            self.worker.log_test.disconnect(self.log_sys.total_log)
            self.log_sys.close()

    @pyqtSlot(bool, str)
    def __init__(self, loaded_pkl):
        super().__init__()
        # PKL 파일
        self.loaded_pkl = loaded_pkl  # TODO 업데이트된 내용
        # {Valname: pkl, ... } 호출  self.loaded_pkl['Valname']
        # 해당 변수는 Main -> Mainwindow -> Worker 까지 연결됨.
        self.other = another_result_explain()
        self.log_sys = log_system()
        self.setObjectName('MainWidget')
        self.setStyleSheet("""#MainWidget {background-color: white;}""")
        self.setWindowTitle("Reliable Intelligent Diagnostic Assistant [RIDA]")
        self.setWindowIcon(QIcon('./icon/system_icon.png'))
        self.setGeometry(150, 50, 1700, 800)
        # 그래프 초기조건
        pyqtgraph.setConfigOption("background", "w")
        pyqtgraph.setConfigOption("foreground", "k")
        #############################################
        # GUI part 1 Layout (진단 부분 통합)
        layout_left = QVBoxLayout()

        # 영 번째 그룹 설정 (Time and Power)
        gb_0 = QGroupBox("Time and NPP's Power")  # 영 번째 그룹 이름 설정
        gb_0.setStyleSheet("QGroupBox { font: bold;}")
        gb_0.setFont(QFont('Times new roman'))
        layout_left.addWidget(gb_0)  # 전체 틀에 영 번째 그룹 넣기
        gb_0_layout = QBoxLayout(QBoxLayout.LeftToRight)  # 영 번째 그룹 내용을 넣을 레이아웃 설정

        # 첫 번째 그룹 설정
        gb_1 = QGroupBox("Training Status Diagnosis Function Result")  # 첫 번째 그룹 이름 설정
        gb_1.setStyleSheet("QGroupBox { font: bold;}")
        gb_1.setFont(QFont('Times new roman'))
        layout_left.addWidget(gb_1)  # 전체 틀에 첫 번째 그룹 넣기
        gb_1_layout = QBoxLayout(QBoxLayout.LeftToRight)  # 첫 번째 그룹 내용을 넣을 레이아웃 설정

        # 두 번째 그룹 설정
        gb_2 = QGroupBox('Scenario Diagnosis Function Result')
        gb_2.setStyleSheet("QGroupBox { font: bold;}")
        gb_2.setFont(QFont('Times new roman'))
        layout_left.addWidget(gb_2)
        gb_2_layout = QBoxLayout(QBoxLayout.LeftToRight)

        # 세 번째 그룹 설정
        gb_3_layout = QBoxLayout(QBoxLayout.LeftToRight)
        gb_2_3_layout = QBoxLayout(QBoxLayout.TopToBottom)


        # 네 번째 그룹 설정
        gb_4 = QGroupBox('Diagnosed Scenario Verification Function Result')
        gb_4.setStyleSheet("QGroupBox { font: bold;}")
        gb_4.setFont(QFont('Times new roman'))
        layout_left.addWidget(gb_4)
        gb_4_layout = QBoxLayout(QBoxLayout.LeftToRight)

        # 다섯 번째 그룹 설정
        gb_5 = QGroupBox('Symptom Satisfaction evaluation Function Result')
        gb_5.setStyleSheet("QGroupBox { font: bold;}")
        gb_5.setFont(QFont('Times new roman'))
        layout_left.addWidget(gb_5)
        gb_5_layout = QBoxLayout(QBoxLayout.TopToBottom)

        # 영 번째 그룹 내용
        self.time_label = QLabel("Time Display")
        self.time_label.setFont(QFont('Times new roman',12))
        self.time_label.setAlignment(Qt.AlignCenter)
        self.power_label = QPushButton('Power display')
        self.power_label.setFont(QFont('Times new roman'))


        # 첫 번째 그룹 내용
        # Trained / Untrained condition label
        self.trained_label = QPushButton('Trained condition')
        self.trained_label.setFont(QFont('Times new roman'))
        self.Untrained_label = QPushButton('Untrained condition')
        self.Untrained_label.setFont(QFont('Times new roman'))

        # 두 번째 그룹 내용
        self.normal_label = QPushButton('Normal operating condition')
        self.normal_label.setFont(QFont('Times new roman'))
        self.abnormal_label = QPushButton('Abnormal operating condition')
        self.abnormal_label.setFont(QFont('Times new roman'))

        # 세 번째 그룹 내용
        self.name_procedure = QLabel("Scenario's Label: ")
        self.name_procedure.setFont(QFont('Times new roman'))
        self.num_procedure = QLineEdit(self)
        self.num_procedure.setFont(QFont('Times new roman'))
        self.num_procedure.setAlignment(Qt.AlignCenter)
        self.name_scnario = QLabel("Scenario's Name: ")
        self.name_scnario.setFont(QFont('Times new roman'))
        self.num_scnario = QLineEdit(self)
        self.num_scnario.setFont(QFont('Times new roman'))
        self.num_scnario.setAlignment(Qt.AlignCenter)

        # 네 번째 그룹 내용
        self.success_label = QPushButton('Diagnosis success')
        self.success_label.setFont(QFont('Times new roman'))
        self.failure_label = QPushButton('Diagnosis failure')
        self.failure_label.setFont(QFont('Times new roman'))

        # 다섯 번째 그룹 내용
        self.symptom_name = QLabel(self)
        self.symptom_name.setFont(QFont('Times new roman'))
        self.symptom1 = QCheckBox(self)
        self.symptom1.setFont(QFont('Times new roman'))
        self.symptom2 = QCheckBox(self)
        self.symptom2.setFont(QFont('Times new roman'))
        self.symptom3 = QCheckBox(self)
        self.symptom3.setFont(QFont('Times new roman'))
        self.symptom4 = QCheckBox(self)
        self.symptom4.setFont(QFont('Times new roman'))
        self.symptom5 = QCheckBox(self)
        self.symptom5.setFont(QFont('Times new roman'))
        self.symptom6 = QCheckBox(self)
        self.symptom6.setFont(QFont('Times new roman'))

        # 영 번째 그룹 내용 입력
        gb_0_layout.addWidget(self.time_label)
        gb_0_layout.addWidget(self.power_label)
        gb_0.setLayout(gb_0_layout)

        # 첫 번째 그룹 내용 입력
        gb_1_layout.addWidget(self.trained_label)
        gb_1_layout.addWidget(self.Untrained_label)
        gb_1.setLayout(gb_1_layout)  # 첫 번째 레이아웃 내용을 첫 번째 그룹 틀로 넣기

        # 두 번째 그룹 내용 입력
        gb_2_layout.addWidget(self.normal_label)
        gb_2_layout.addWidget(self.abnormal_label)
        # gb_2.setLayout(gb_2_layout)

        # 세 번째 그룹 내용 입력
        gb_3_layout.addWidget(self.name_procedure)
        gb_3_layout.addWidget(self.num_procedure)
        gb_3_layout.addWidget(self.name_scnario)
        gb_3_layout.addWidget(self.num_scnario)
        gb_2_3_layout.addLayout(gb_2_layout)
        gb_2_3_layout.addLayout(gb_3_layout)
        gb_2.setLayout(gb_2_3_layout)

        # 네 번째 그룹 내용 입력
        gb_4_layout.addWidget(self.success_label)
        gb_4_layout.addWidget(self.failure_label)
        gb_4.setLayout(gb_4_layout)

        # 다섯 번째 그룹 내용 입력
        gb_5_layout.addWidget(self.symptom_name)
        gb_5_layout.addWidget(self.symptom1)
        gb_5_layout.addWidget(self.symptom2)
        gb_5_layout.addWidget(self.symptom3)
        gb_5_layout.addWidget(self.symptom4)
        gb_5_layout.addWidget(self.symptom5)
        gb_5_layout.addWidget(self.symptom6)
        gb_5.setLayout(gb_5_layout)

        # Start 버튼 맨 아래에 위치
        self.start_btn = QPushButton('Start')
        self.start_btn.setFont(QFont('Times new roman'))
        self.start_btn.setIcon(QIcon('./icon/play_icon.png'))
        self.log_btn = QPushButton('Log')
        self.log_btn.setFont(QFont('Times new roman'))
        self.log_btn.setIcon(QIcon('./icon/log_icon.png'))
        self.log_btn.setCheckable(True)

        self.tableWidget = QTableWidget(0, 0)
        self.tableWidget.setFixedHeight(500)
        self.tableWidget.setFixedWidth(800)
        self.tableWidget.setWindowTitle('Details of diagnosis evidence [Table]')
        self.tableWidget.setFont(QFont('Times new roman'))

        # Plot 구현
        self.plot_1 = pyqtgraph.PlotWidget(title='Activation Plot1')
        self.plot_2 = pyqtgraph.PlotWidget(title='Activation Plot2')
        self.plot_3 = pyqtgraph.PlotWidget(title='Activation Plot3')
        self.plot_4 = pyqtgraph.PlotWidget(title='Activation Plot4')
        # Plot style을 위한 getPlotItem 선언
        item1 = self.plot_1.getPlotItem()
        item2 = self.plot_2.getPlotItem()
        item3 = self.plot_3.getPlotItem()
        item4 = self.plot_4.getPlotItem()
        font = QFont('Times New Roman')
        # plot1_item 선언
        item1.titleLabel.item.setFont(font)
        item1.getAxis("bottom").setStyle(tickFont = font)
        item1.getAxis("left").setStyle(tickFont = font)
        # plot2_item 선언
        item2.titleLabel.item.setFont(font)
        item2.getAxis("bottom").setStyle(tickFont=font)
        item2.getAxis("left").setStyle(tickFont=font)
        # plot3_item 선언
        item3.titleLabel.item.setFont(font)
        item3.getAxis("bottom").setStyle(tickFont=font)
        item3.getAxis("left").setStyle(tickFont=font)
        # plot4_item 선언
        item4.titleLabel.item.setFont(font)
        item4.getAxis("bottom").setStyle(tickFont=font)
        item4.getAxis("left").setStyle(tickFont=font)

        self.plot_1.clear()
        self.plot_2.clear()
        self.plot_3.clear()
        self.plot_4.clear()

        # Explanation Alarm 구현
        ex_label = QLabel("<b>Diagnosis Evidence Derivation Function Result<b/>")
        ex_label.setFont(QFont('Times new roman'))
        ex_label.setAlignment(Qt.AlignCenter)
        red_alarm = QGroupBox('Main evidence for diagnosis (10%↑)')
        red_alarm.setStyleSheet("QGroupBox { font: bold;}")
        red_alarm.setFont(QFont('Times new roman'))
        red_alarm_layout = QGridLayout()
        orange_alarm = QGroupBox('Sub evidence for diagnosis (1%~10%)')
        orange_alarm.setStyleSheet("QGroupBox { font: bold;}")
        orange_alarm.setFont(QFont('Times new roman'))
        orange_alarm_layout = QGridLayout()
        # Display Button 생성
        self.red1 = QPushButton(self)
        self.red1.setFont(QFont('Times new roman'))
        self.red2 = QPushButton(self)
        self.red2.setFont(QFont('Times new roman'))
        self.red3 = QPushButton(self)
        self.red3.setFont(QFont('Times new roman'))
        self.red4 = QPushButton(self)
        self.red4.setFont(QFont('Times new roman'))
        self.orange1 = QPushButton(self)
        self.orange1.setFont(QFont('Times new roman'))
        self.orange2 = QPushButton(self)
        self.orange2.setFont(QFont('Times new roman'))
        self.orange3 = QPushButton(self)
        self.orange3.setFont(QFont('Times new roman'))
        self.orange4 = QPushButton(self)
        self.orange4.setFont(QFont('Times new roman'))
        self.orange5 = QPushButton(self)
        self.orange5.setFont(QFont('Times new roman'))
        self.orange6 = QPushButton(self)
        self.orange6.setFont(QFont('Times new roman'))
        self.orange7 = QPushButton(self)
        self.orange7.setFont(QFont('Times new roman'))
        self.orange8 = QPushButton(self)
        self.orange8.setFont(QFont('Times new roman'))
        self.orange9 = QPushButton(self)
        self.orange9.setFont(QFont('Times new roman'))
        self.orange10 = QPushButton(self)
        self.orange10.setFont(QFont('Times new roman'))
        self.orange11 = QPushButton(self)
        self.orange11.setFont(QFont('Times new roman'))
        self.orange12 = QPushButton(self)
        self.orange12.setFont(QFont('Times new roman'))
        # Layout에 widget 삽입
        red_alarm_layout.addWidget(self.red1, 0, 0)
        red_alarm_layout.addWidget(self.red2, 0, 1)
        red_alarm_layout.addWidget(self.red3, 1, 0)
        red_alarm_layout.addWidget(self.red4, 1, 1)
        orange_alarm_layout.addWidget(self.orange1, 0, 0)
        orange_alarm_layout.addWidget(self.orange2, 0, 1)
        orange_alarm_layout.addWidget(self.orange3, 1, 0)
        orange_alarm_layout.addWidget(self.orange4, 1, 1)
        orange_alarm_layout.addWidget(self.orange5, 2, 0)
        orange_alarm_layout.addWidget(self.orange6, 2, 1)
        orange_alarm_layout.addWidget(self.orange7, 3, 0)
        orange_alarm_layout.addWidget(self.orange8, 3, 1)
        orange_alarm_layout.addWidget(self.orange9, 4, 0)
        orange_alarm_layout.addWidget(self.orange10, 4, 1)
        orange_alarm_layout.addWidget(self.orange11, 5, 0)
        orange_alarm_layout.addWidget(self.orange12, 5, 1)
        # Group Box에 Layout 삽입
        red_alarm.setLayout(red_alarm_layout)
        orange_alarm.setLayout(orange_alarm_layout)
        # 각 Group Box를 상위 Layout에 삽입
        hline = QFrame() # 각 window 구분을 위한 수평선 생성
        hline.setFrameShape(QFrame.HLine)
        hline.setFrameShadow(QFrame.Raised)
        hline.setStyleSheet('border: 1.5px solid gray')


        layout_part1 = QVBoxLayout()
        detail_part = QHBoxLayout()
        self.detailed_table = QPushButton('Overall Evidence Table')
        self.detailed_table.setFont(QFont('Times new roman'))
        self.detailed_table.setIcon(QIcon('./icon/table_icon.png'))
        self.detailed_table.setCheckable(True)
        self.another_classification = QPushButton('Evidence for unselected diagnostic results')
        self.another_classification.setFont(QFont('Times new roman'))
        self.another_classification.setCheckable(True)
        detail_part.addWidget(self.detailed_table)
        detail_part.addWidget(self.another_classification)
        alarm_main = QVBoxLayout()
        alarm_main.addWidget(ex_label)
        alarm_main.addWidget(red_alarm)
        alarm_main.addWidget(orange_alarm)
        layout_part1.addLayout(layout_left)
        layout_part1.addWidget(hline)
        layout_part1.addLayout(alarm_main)
        layout_part1.addLayout(detail_part)
        layout_part1.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # GUI part2 Layout (XAI 구현)
        layout_part2 = QVBoxLayout()
        layout_part2.addWidget(self.plot_1)
        layout_part2.addWidget(self.plot_2)
        layout_part2.addWidget(self.plot_3)
        layout_part2.addWidget(self.plot_4)
        # layout_part2.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        # layout_part2.addWidget(self.tableWidget)

        # GUI part1 and part2 통합
        layout_base = QHBoxLayout()
        layout_base.addLayout(layout_part1)
        layout_base.addLayout(layout_part2)

        # GUI 최종 통합 (start button을 하단에 배치시키기 위함)
        total_layout = QVBoxLayout()
        start_log_layout = QHBoxLayout()
        start_log_layout.addWidget(self.start_btn)
        start_log_layout.addWidget(self.log_btn)
        total_layout.addLayout(layout_base)
        total_layout.addLayout(start_log_layout)

        self.setLayout(total_layout)  # setLayout : 최종 출력될 GUI 화면을 결정

        # Threading Part Start ##############################################################################################################
        self.worker = Worker(Loaded_DB)

        # 데이터 연산 부분 Thread화
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)

        self.start_btn.clicked.connect(lambda: self.worker.generate_db())  # 누르면 For문 실행
        self.worker_thread.start()
        print('Thread Start')

        # Signal을 Main Thread 내의 함수와 연결
        self.worker.train_value.connect(self.Determine_train)
        self.worker.procedure_value.connect(self.Determine_abnormal)
        self.worker.procedure_list.connect(self.Determine_procedure)
        self.worker.verif_value.connect(self.verifit_result)
        self.worker.timer.connect(self.time_display)
        self.worker.symptom_db.connect(self.procedure_satisfaction)
        self.worker.plot_db.connect(self.plotting)
        self.worker.display_ex.connect(self.display_explain)





        # Threading Part END ##############################################################################################################

        # 이벤트 처리 ----------------------------------------------------------------------------------------------------
        self.detailed_table.clicked.connect(self.table_control)
        self.another_classification.clicked.connect(self.show_another_result)
        self.log_btn.clicked.connect(self.log_system_control)



        # Button 클릭 연동 이벤트 처리
        convert_red_btn = {0: self.red1, 1: self.red2, 2: self.red3, 3: self.red4}  # Red Button
        convert_red_plot = {0: self.red1_plot, 1: self.red2_plot, 2: self.red3_plot, 3: self.red4_plot}  #

        convert_orange_btn = {0: self.orange1, 1: self.orange2, 2: self.orange3, 3: self.orange4, 4: self.orange5,
                              5: self.orange6, 6: self.orange7, 7: self.orange8, 8: self.orange9, 9: self.orange10,
                              10: self.orange11, 11: self.orange12}  # Orange Button
        convert_orange_plot = {0: self.orange1_plot, 1: self.orange2_plot, 2: self.orange3_plot, 3: self.orange4_plot, 4: self.orange5_plot,
                               5: self.orange6_plot, 6: self.orange7_plot, 7: self.orange8_plot, 8: self.orange9_plot, 9: self.orange10_plot,
                               10: self.orange11_plot, 11: self.orange12_plot}

        # 초기 Button 위젯 선언 -> 초기에 선언해야 끊기지않고 유지됨.
        # Red Button
        [convert_red_btn[i].clicked.connect(convert_red_plot[i]) for i in range(4)]
        self.red_plot_1 = pyqtgraph.PlotWidget(title=self)
        self.red_plot_2 = pyqtgraph.PlotWidget(title=self)
        self.red_plot_3 = pyqtgraph.PlotWidget(title=self)
        self.red_plot_4 = pyqtgraph.PlotWidget(title=self)
        self.red_plot_1.setWindowTitle('Main Evidence 1 Graph')
        self.red_plot_2.setWindowTitle('Main Evidence 2 Graph')
        self.red_plot_3.setWindowTitle('Main Evidence 3 Graph')
        self.red_plot_4.setWindowTitle('Main Evidence 4 Graph')
        # Grid setting
        self.red_plot_1.showGrid(x=True, y=True, alpha=0.3)
        self.red_plot_2.showGrid(x=True, y=True, alpha=0.3)
        self.red_plot_3.showGrid(x=True, y=True, alpha=0.3)
        self.red_plot_4.showGrid(x=True, y=True, alpha=0.3)
        # Plot style을 위한 getPlotItem 선언
        red_item1 = self.red_plot_1.getPlotItem()
        red_item2 = self.red_plot_2.getPlotItem()
        red_item3 = self.red_plot_3.getPlotItem()
        red_item4 = self.red_plot_4.getPlotItem()
        font = QFont('Times New Roman')
        # red_plot1_item 선언
        red_item1.titleLabel.item.setFont(font)
        red_item1.getAxis("bottom").setStyle(tickFont=font)
        red_item1.getAxis("left").setStyle(tickFont=font)
        # red_plot2_item 선언
        red_item2.titleLabel.item.setFont(font)
        red_item2.getAxis("bottom").setStyle(tickFont=font)
        red_item2.getAxis("left").setStyle(tickFont=font)
        # red_plot3_item 선언
        red_item3.titleLabel.item.setFont(font)
        red_item3.getAxis("bottom").setStyle(tickFont=font)
        red_item3.getAxis("left").setStyle(tickFont=font)
        # red_plot4_item 선언
        red_item4.titleLabel.item.setFont(font)
        red_item4.getAxis("bottom").setStyle(tickFont=font)
        red_item4.getAxis("left").setStyle(tickFont=font)

        # Orange Button
        [convert_orange_btn[i].clicked.connect(convert_orange_plot[i]) for i in range(12)]
        self.orange_plot_1 = pyqtgraph.PlotWidget(title=self)
        self.orange_plot_2 = pyqtgraph.PlotWidget(title=self)
        self.orange_plot_3 = pyqtgraph.PlotWidget(title=self)
        self.orange_plot_4 = pyqtgraph.PlotWidget(title=self)
        self.orange_plot_5 = pyqtgraph.PlotWidget(title=self)
        self.orange_plot_6 = pyqtgraph.PlotWidget(title=self)
        self.orange_plot_7 = pyqtgraph.PlotWidget(title=self)
        self.orange_plot_8 = pyqtgraph.PlotWidget(title=self)
        self.orange_plot_9 = pyqtgraph.PlotWidget(title=self)
        self.orange_plot_10 = pyqtgraph.PlotWidget(title=self)
        self.orange_plot_11 = pyqtgraph.PlotWidget(title=self)
        self.orange_plot_12 = pyqtgraph.PlotWidget(title=self)
        self.orange_plot_1.setWindowTitle('Sub Evidence 1 Graph')
        self.orange_plot_2.setWindowTitle('Sub Evidence 2 Graph')
        self.orange_plot_3.setWindowTitle('Sub Evidence 3 Graph')
        self.orange_plot_4.setWindowTitle('Sub Evidence 4 Graph')
        self.orange_plot_5.setWindowTitle('Sub Evidence 5 Graph')
        self.orange_plot_6.setWindowTitle('Sub Evidence 6 Graph')
        self.orange_plot_7.setWindowTitle('Sub Evidence 7 Graph')
        self.orange_plot_8.setWindowTitle('Sub Evidence 8 Graph')
        self.orange_plot_9.setWindowTitle('Sub Evidence 9 Graph')
        self.orange_plot_10.setWindowTitle('Sub Evidence 10 Graph')
        self.orange_plot_11.setWindowTitle('Sub Evidence 11 Graph')
        self.orange_plot_12.setWindowTitle('Sub Evidence 12 Graph')
        # Grid setting
        self.orange_plot_1.showGrid(x=True, y=True, alpha=0.3)
        self.orange_plot_2.showGrid(x=True, y=True, alpha=0.3)
        self.orange_plot_3.showGrid(x=True, y=True, alpha=0.3)
        self.orange_plot_4.showGrid(x=True, y=True, alpha=0.3)
        self.orange_plot_5.showGrid(x=True, y=True, alpha=0.3)
        self.orange_plot_6.showGrid(x=True, y=True, alpha=0.3)
        self.orange_plot_7.showGrid(x=True, y=True, alpha=0.3)
        self.orange_plot_8.showGrid(x=True, y=True, alpha=0.3)
        self.orange_plot_9.showGrid(x=True, y=True, alpha=0.3)
        self.orange_plot_10.showGrid(x=True, y=True, alpha=0.3)
        self.orange_plot_11.showGrid(x=True, y=True, alpha=0.3)
        self.orange_plot_12.showGrid(x=True, y=True, alpha=0.3)
        # Plot style을 위한 getPlotItem 선언
        orange_item1 = self.orange_plot_1.getPlotItem()
        orange_item2 = self.orange_plot_2.getPlotItem()
        orange_item3 = self.orange_plot_3.getPlotItem()
        orange_item4 = self.orange_plot_4.getPlotItem()
        orange_item5 = self.orange_plot_5.getPlotItem()
        orange_item6 = self.orange_plot_6.getPlotItem()
        orange_item7 = self.orange_plot_7.getPlotItem()
        orange_item8 = self.orange_plot_8.getPlotItem()
        orange_item9 = self.orange_plot_9.getPlotItem()
        orange_item10 = self.orange_plot_10.getPlotItem()
        orange_item11 = self.orange_plot_11.getPlotItem()
        orange_item12 = self.orange_plot_12.getPlotItem()
        font = QFont('Times New Roman')
        # orange_item1 선언
        orange_item1.titleLabel.item.setFont(font)
        orange_item1.getAxis("bottom").setStyle(tickFont=font)
        orange_item1.getAxis("left").setStyle(tickFont=font)
        # orange_item2 선언
        orange_item2.titleLabel.item.setFont(font)
        orange_item2.getAxis("bottom").setStyle(tickFont=font)
        orange_item2.getAxis("left").setStyle(tickFont=font)
        # orange_item3 선언
        orange_item3.titleLabel.item.setFont(font)
        orange_item3.getAxis("bottom").setStyle(tickFont=font)
        orange_item3.getAxis("left").setStyle(tickFont=font)
        # orange_item4 선언
        orange_item4.titleLabel.item.setFont(font)
        orange_item4.getAxis("bottom").setStyle(tickFont=font)
        orange_item4.getAxis("left").setStyle(tickFont=font)
        # orange_item5 선언
        orange_item5.titleLabel.item.setFont(font)
        orange_item5.getAxis("bottom").setStyle(tickFont=font)
        orange_item5.getAxis("left").setStyle(tickFont=font)
        # orange_item6 선언
        orange_item6.titleLabel.item.setFont(font)
        orange_item6.getAxis("bottom").setStyle(tickFont=font)
        orange_item6.getAxis("left").setStyle(tickFont=font)
        # orange_item7 선언
        orange_item7.titleLabel.item.setFont(font)
        orange_item7.getAxis("bottom").setStyle(tickFont=font)
        orange_item7.getAxis("left").setStyle(tickFont=font)
        # orange_item8 선언
        orange_item8.titleLabel.item.setFont(font)
        orange_item8.getAxis("bottom").setStyle(tickFont=font)
        orange_item8.getAxis("left").setStyle(tickFont=font)
        # orange_item9 선언
        orange_item9.titleLabel.item.setFont(font)
        orange_item9.getAxis("bottom").setStyle(tickFont=font)
        orange_item9.getAxis("left").setStyle(tickFont=font)
        # orange_item10 선언
        orange_item10.titleLabel.item.setFont(font)
        orange_item10.getAxis("bottom").setStyle(tickFont=font)
        orange_item10.getAxis("left").setStyle(tickFont=font)
        # orange_item11 선언
        orange_item11.titleLabel.item.setFont(font)
        orange_item11.getAxis("bottom").setStyle(tickFont=font)
        orange_item11.getAxis("left").setStyle(tickFont=font)
        # orange_item12 선언
        orange_item12.titleLabel.item.setFont(font)
        orange_item12.getAxis("bottom").setStyle(tickFont=font)
        orange_item12.getAxis("left").setStyle(tickFont=font)

        self.show()  # UI show command

class another_result_explain(QWidget):
    cb_text = pyqtSignal(object)
    @pyqtSlot('QString', 'QString')
    def __init__(self):
        super().__init__()
        # 서브 인터페이스 초기 설정
        self.setObjectName('SubWidget')
        self.setStyleSheet("""#SubWidget {background-color: white;}""")
        self.setWindowTitle('Evidence for unselected diagnostic results')
        self.setWindowIcon(QIcon('./icon/system_icon.png'))
        self.setGeometry(300, 300, 800, 500)

        pyqtgraph.setConfigOption("background", "w")
        pyqtgraph.setConfigOption("foreground", "k")


        # 레이아웃 구성
        combo_layout = QVBoxLayout()
        self.title_label = QLabel("<b>Diagnosis Evidence Derivation Function Result -> Evidence for unselected diagnostic results<b/>")
        self.title_label.setFont(QFont('Times new roman'))
        self.title_label.setAlignment(Qt.AlignCenter)

        self.blank = QLabel(self)  # Enter를 위한 라벨

        self.show_table = QPushButton("Overall Evidence Table")
        self.show_table.setFont(QFont('Times new roman'))
        self.show_table.setIcon(QIcon('./icon/table_icon.png'))

        self.cb = QComboBox(self)
        self.cb.setFont(QFont('Times new roman'))
        self.cb.addItem('Normal')
        self.cb.addItem('Ab21_01: Pressurizer pressure channel failure (High)')
        self.cb.addItem('Ab21_02: Pressurizer pressure channel failure (Low)')
        self.cb.addItem('Ab20_04: Pressurizer level channel failure (Low)')
        self.cb.addItem('Ab15_07: Steam generator level channel failure (High)')
        self.cb.addItem('Ab15_08: Steam generator level channel failure (Low)')
        self.cb.addItem('Ab63_04: Control rod fall')
        self.cb.addItem('Ab63_02: Continuous insertion of control rod')
        self.cb.addItem('Ab21_12: Pressurizer PORV opening')
        self.cb.addItem('Ab19_02: Pressurizer safety valve failure')
        self.cb.addItem('Ab21_11: Opening of PRZ spray valve')
        self.cb.addItem('Ab23_03: Leakage from CVCS to RCS')
        self.cb.addItem('Ab60_02: Rupture of the front end of the regenerative heat exchanger')
        self.cb.addItem('Ab59_02: Leakage at the rear end of the charging flow control valve')
        self.cb.addItem('Ab23_01: Leakage from CVCS to CCW')
        self.cb.addItem('Ab23_06: Steam generator u-tube leakage')

        # Explanation Alarm 구현
        cb_red_alarm = QGroupBox('Main evidence for diagnosis (10%↑)')
        cb_red_alarm.setStyleSheet("QGroupBox { font: bold;}")
        cb_red_alarm.setFont(QFont('Times new roman'))
        cb_red_alarm_layout = QGridLayout()
        cb_orange_alarm = QGroupBox('Sub evidence for diagnosis (1%~10%)')
        cb_orange_alarm.setStyleSheet("QGroupBox { font: bold;}")
        cb_orange_alarm.setFont(QFont('Times new roman'))
        cb_orange_alarm_layout = QGridLayout()

        # Display Button 생성
        self.cb_red1 = QPushButton(self)
        self.cb_red1.setFont(QFont('Times new roman'))
        self.cb_red2 = QPushButton(self)
        self.cb_red2.setFont(QFont('Times new roman'))
        self.cb_red3 = QPushButton(self)
        self.cb_red3.setFont(QFont('Times new roman'))
        self.cb_red4 = QPushButton(self)
        self.cb_red4.setFont(QFont('Times new roman'))
        self.cb_orange1 = QPushButton(self)
        self.cb_orange1.setFont(QFont('Times new roman'))
        self.cb_orange2 = QPushButton(self)
        self.cb_orange2.setFont(QFont('Times new roman'))
        self.cb_orange3 = QPushButton(self)
        self.cb_orange3.setFont(QFont('Times new roman'))
        self.cb_orange4 = QPushButton(self)
        self.cb_orange4.setFont(QFont('Times new roman'))
        self.cb_orange5 = QPushButton(self)
        self.cb_orange5.setFont(QFont('Times new roman'))
        self.cb_orange6 = QPushButton(self)
        self.cb_orange6.setFont(QFont('Times new roman'))
        self.cb_orange7 = QPushButton(self)
        self.cb_orange7.setFont(QFont('Times new roman'))
        self.cb_orange8 = QPushButton(self)
        self.cb_orange8.setFont(QFont('Times new roman'))
        self.cb_orange9 = QPushButton(self)
        self.cb_orange9.setFont(QFont('Times new roman'))
        self.cb_orange10 = QPushButton(self)
        self.cb_orange10.setFont(QFont('Times new roman'))
        self.cb_orange11 = QPushButton(self)
        self.cb_orange11.setFont(QFont('Times new roman'))
        self.cb_orange12 = QPushButton(self)
        self.cb_orange12.setFont(QFont('Times new roman'))
        # Layout에 widget 삽입
        cb_red_alarm_layout.addWidget(self.cb_red1, 0, 0)
        cb_red_alarm_layout.addWidget(self.cb_red2, 0, 1)
        cb_red_alarm_layout.addWidget(self.cb_red3, 1, 0)
        cb_red_alarm_layout.addWidget(self.cb_red4, 1, 1)
        cb_orange_alarm_layout.addWidget(self.cb_orange1, 0, 0)
        cb_orange_alarm_layout.addWidget(self.cb_orange2, 0, 1)
        cb_orange_alarm_layout.addWidget(self.cb_orange3, 1, 0)
        cb_orange_alarm_layout.addWidget(self.cb_orange4, 1, 1)
        cb_orange_alarm_layout.addWidget(self.cb_orange5, 2, 0)
        cb_orange_alarm_layout.addWidget(self.cb_orange6, 2, 1)
        cb_orange_alarm_layout.addWidget(self.cb_orange7, 3, 0)
        cb_orange_alarm_layout.addWidget(self.cb_orange8, 3, 1)
        cb_orange_alarm_layout.addWidget(self.cb_orange9, 4, 0)
        cb_orange_alarm_layout.addWidget(self.cb_orange10, 4, 1)
        cb_orange_alarm_layout.addWidget(self.cb_orange11, 5, 0)
        cb_orange_alarm_layout.addWidget(self.cb_orange12, 5, 1)

        cb_red_alarm.setLayout(cb_red_alarm_layout)
        cb_orange_alarm.setLayout(cb_orange_alarm_layout)

        combo_layout.addWidget(self.title_label)
        combo_layout.addWidget(self.blank)
        combo_layout.addWidget(self.cb)
        combo_layout.addWidget(self.blank)
        combo_layout.addWidget(cb_red_alarm)
        combo_layout.addWidget(cb_orange_alarm)
        combo_layout.addWidget(self.blank)
        combo_layout.addWidget(self.show_table)
        combo_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.setLayout(combo_layout)

        self.combo_tableWidget = QTableWidget(0, 0)
        self.combo_tableWidget.setFixedHeight(500)
        self.combo_tableWidget.setFixedWidth(800)
        self.combo_tableWidget.setWindowTitle('Details of the others diagnosis evidence [Table]')
        self.combo_tableWidget.setFont(QFont('Times new roman'))

        # 이벤트 처리 부분 ########################################################


        self.show_table.clicked.connect(self.show_anoter_table)
        self.cb.activated[str].connect(self.show_another_result_table)
        self.cb.activated[str].connect(self.show_shap)


        ##########################################################################

        # Button 클릭 연동 이벤트 처리
        convert_cb_red_btn = {0: self.cb_red1, 1: self.cb_red2, 2: self.cb_red3, 3: self.cb_red4}  # Red Button
        convert_cb_red_plot = {0: self.cb_red1_plot, 1: self.cb_red2_plot, 2: self.cb_red3_plot, 3: self.cb_red4_plot}

        convert_cb_orange_btn = {0: self.cb_orange1, 1: self.cb_orange2, 2: self.cb_orange3, 3: self.cb_orange4, 4: self.cb_orange5,
                                 5: self.cb_orange6, 6: self.cb_orange7, 7: self.cb_orange8, 8: self.cb_orange9, 9: self.cb_orange10,
                                 10: self.cb_orange11, 11: self.cb_orange12}  # Orange Button
        convert_cb_orange_plot = {0: self.cb_orange1_plot, 1: self.cb_orange2_plot, 2: self.cb_orange3_plot, 3: self.cb_orange4_plot,
                                  4: self.cb_orange5_plot, 5: self.cb_orange6_plot, 6: self.cb_orange7_plot, 7: self.cb_orange8_plot,
                                  8: self.cb_orange9_plot, 9: self.cb_orange10_plot, 10: self.cb_orange11_plot, 11: self.cb_orange12_plot}

        ################################################################################################################
        # 초기 Button 위젯 선언 -> 초기에 선언해야 끊기지않고 유지됨.
        # Red Button
        [convert_cb_red_btn[i].clicked.connect(convert_cb_red_plot[i]) for i in range(4)]
        self.cb_red_plot_1 = pyqtgraph.PlotWidget(title=self)
        self.cb_red_plot_2 = pyqtgraph.PlotWidget(title=self)
        self.cb_red_plot_3 = pyqtgraph.PlotWidget(title=self)
        self.cb_red_plot_4 = pyqtgraph.PlotWidget(title=self)
        self.cb_red_plot_1.setWindowTitle('Evidence for unselected diagnostic results -> Main Evidence 1 Graph')
        self.cb_red_plot_2.setWindowTitle('Evidence for unselected diagnostic results -> Main Evidence 2 Graph')
        self.cb_red_plot_3.setWindowTitle('Evidence for unselected diagnostic results -> Main Evidence 3 Graph')
        self.cb_red_plot_4.setWindowTitle('Evidence for unselected diagnostic results -> Main Evidence 4 Graph')
        # Grid setting
        self.cb_red_plot_1.showGrid(x=True, y=True, alpha=0.3)
        self.cb_red_plot_2.showGrid(x=True, y=True, alpha=0.3)
        self.cb_red_plot_3.showGrid(x=True, y=True, alpha=0.3)
        self.cb_red_plot_4.showGrid(x=True, y=True, alpha=0.3)
        # Plot style을 위한 getPlotItem 선언
        cb_red_item1 = self.cb_red_plot_1.getPlotItem()
        cb_red_item2 = self.cb_red_plot_2.getPlotItem()
        cb_red_item3 = self.cb_red_plot_3.getPlotItem()
        cb_red_item4 = self.cb_red_plot_4.getPlotItem()
        font = QFont('Times New Roman')
        # cb_red_plot1_item 선언
        cb_red_item1.titleLabel.item.setFont(font)
        cb_red_item1.getAxis("bottom").setStyle(tickFont=font)
        cb_red_item1.getAxis("left").setStyle(tickFont=font)
        # cb_red_plot2_item 선언
        cb_red_item2.titleLabel.item.setFont(font)
        cb_red_item2.getAxis("bottom").setStyle(tickFont=font)
        cb_red_item2.getAxis("left").setStyle(tickFont=font)
        # cb_red_plot3_item 선언
        cb_red_item3.titleLabel.item.setFont(font)
        cb_red_item3.getAxis("bottom").setStyle(tickFont=font)
        cb_red_item3.getAxis("left").setStyle(tickFont=font)
        # cb_red_plot4_item 선언
        cb_red_item4.titleLabel.item.setFont(font)
        cb_red_item4.getAxis("bottom").setStyle(tickFont=font)
        cb_red_item4.getAxis("left").setStyle(tickFont=font)

        # Orange Button
        [convert_cb_orange_btn[i].clicked.connect(convert_cb_orange_plot[i]) for i in range(12)]
        self.cb_orange_plot_1 = pyqtgraph.PlotWidget(title=self)
        self.cb_orange_plot_2 = pyqtgraph.PlotWidget(title=self)
        self.cb_orange_plot_3 = pyqtgraph.PlotWidget(title=self)
        self.cb_orange_plot_4 = pyqtgraph.PlotWidget(title=self)
        self.cb_orange_plot_5 = pyqtgraph.PlotWidget(title=self)
        self.cb_orange_plot_6 = pyqtgraph.PlotWidget(title=self)
        self.cb_orange_plot_7 = pyqtgraph.PlotWidget(title=self)
        self.cb_orange_plot_8 = pyqtgraph.PlotWidget(title=self)
        self.cb_orange_plot_9 = pyqtgraph.PlotWidget(title=self)
        self.cb_orange_plot_10 = pyqtgraph.PlotWidget(title=self)
        self.cb_orange_plot_11 = pyqtgraph.PlotWidget(title=self)
        self.cb_orange_plot_12 = pyqtgraph.PlotWidget(title=self)
        self.cb_orange_plot_1.setWindowTitle('Evidence for unselected diagnostic results -> Sub Evidence 1 Graph')
        self.cb_orange_plot_2.setWindowTitle('Evidence for unselected diagnostic results -> Sub Evidence 2 Graph')
        self.cb_orange_plot_3.setWindowTitle('Evidence for unselected diagnostic results -> Sub Evidence 3 Graph')
        self.cb_orange_plot_4.setWindowTitle('Evidence for unselected diagnostic results -> Sub Evidence 4 Graph')
        self.cb_orange_plot_5.setWindowTitle('Evidence for unselected diagnostic results -> Sub Evidence 5 Graph')
        self.cb_orange_plot_6.setWindowTitle('Evidence for unselected diagnostic results -> Sub Evidence 6 Graph')
        self.cb_orange_plot_7.setWindowTitle('Evidence for unselected diagnostic results -> Sub Evidence 7 Graph')
        self.cb_orange_plot_8.setWindowTitle('Evidence for unselected diagnostic results -> Sub Evidence 8 Graph')
        self.cb_orange_plot_9.setWindowTitle('Evidence for unselected diagnostic results -> Sub Evidence 9 Graph')
        self.cb_orange_plot_10.setWindowTitle('Evidence for unselected diagnostic results -> Sub Evidence 10 Graph')
        self.cb_orange_plot_11.setWindowTitle('Evidence for unselected diagnostic results -> Sub Evidence 11 Graph')
        self.cb_orange_plot_12.setWindowTitle('Evidence for unselected diagnostic results -> Sub Evidence 12 Graph')
        # Grid setting
        self.cb_orange_plot_1.showGrid(x=True, y=True, alpha=0.3)
        self.cb_orange_plot_2.showGrid(x=True, y=True, alpha=0.3)
        self.cb_orange_plot_3.showGrid(x=True, y=True, alpha=0.3)
        self.cb_orange_plot_4.showGrid(x=True, y=True, alpha=0.3)
        self.cb_orange_plot_5.showGrid(x=True, y=True, alpha=0.3)
        self.cb_orange_plot_6.showGrid(x=True, y=True, alpha=0.3)
        self.cb_orange_plot_7.showGrid(x=True, y=True, alpha=0.3)
        self.cb_orange_plot_8.showGrid(x=True, y=True, alpha=0.3)
        self.cb_orange_plot_9.showGrid(x=True, y=True, alpha=0.3)
        self.cb_orange_plot_10.showGrid(x=True, y=True, alpha=0.3)
        self.cb_orange_plot_11.showGrid(x=True, y=True, alpha=0.3)
        self.cb_orange_plot_12.showGrid(x=True, y=True, alpha=0.3)
        # Plot style을 위한 getPlotItem 선언
        cb_orange_item1 = self.cb_orange_plot_1.getPlotItem()
        cb_orange_item2 = self.cb_orange_plot_2.getPlotItem()
        cb_orange_item3 = self.cb_orange_plot_3.getPlotItem()
        cb_orange_item4 = self.cb_orange_plot_4.getPlotItem()
        cb_orange_item5 = self.cb_orange_plot_5.getPlotItem()
        cb_orange_item6 = self.cb_orange_plot_6.getPlotItem()
        cb_orange_item7 = self.cb_orange_plot_7.getPlotItem()
        cb_orange_item8 = self.cb_orange_plot_8.getPlotItem()
        cb_orange_item9 = self.cb_orange_plot_9.getPlotItem()
        cb_orange_item10 = self.cb_orange_plot_10.getPlotItem()
        cb_orange_item11 = self.cb_orange_plot_11.getPlotItem()
        cb_orange_item12 = self.cb_orange_plot_12.getPlotItem()
        font = QFont('Times New Roman')
        # cb_orange_item1 선언
        cb_orange_item1.titleLabel.item.setFont(font)
        cb_orange_item1.getAxis("bottom").setStyle(tickFont=font)
        cb_orange_item1.getAxis("left").setStyle(tickFont=font)
        # cb_orange_item2 선언
        cb_orange_item2.titleLabel.item.setFont(font)
        cb_orange_item2.getAxis("bottom").setStyle(tickFont=font)
        cb_orange_item2.getAxis("left").setStyle(tickFont=font)
        # cb_orange_item3 선언
        cb_orange_item3.titleLabel.item.setFont(font)
        cb_orange_item3.getAxis("bottom").setStyle(tickFont=font)
        cb_orange_item3.getAxis("left").setStyle(tickFont=font)
        # cb_orange_item4 선언
        cb_orange_item4.titleLabel.item.setFont(font)
        cb_orange_item4.getAxis("bottom").setStyle(tickFont=font)
        cb_orange_item4.getAxis("left").setStyle(tickFont=font)
        # cb_orange_item5 선언
        cb_orange_item5.titleLabel.item.setFont(font)
        cb_orange_item5.getAxis("bottom").setStyle(tickFont=font)
        cb_orange_item5.getAxis("left").setStyle(tickFont=font)
        # cb_orange_item6 선언
        cb_orange_item6.titleLabel.item.setFont(font)
        cb_orange_item6.getAxis("bottom").setStyle(tickFont=font)
        cb_orange_item6.getAxis("left").setStyle(tickFont=font)
        # cb_orange_item7 선언
        cb_orange_item7.titleLabel.item.setFont(font)
        cb_orange_item7.getAxis("bottom").setStyle(tickFont=font)
        cb_orange_item7.getAxis("left").setStyle(tickFont=font)
        # cb_orange_item8 선언
        cb_orange_item8.titleLabel.item.setFont(font)
        cb_orange_item8.getAxis("bottom").setStyle(tickFont=font)
        cb_orange_item8.getAxis("left").setStyle(tickFont=font)
        # cb_orange_item9 선언
        cb_orange_item9.titleLabel.item.setFont(font)
        cb_orange_item9.getAxis("bottom").setStyle(tickFont=font)
        cb_orange_item9.getAxis("left").setStyle(tickFont=font)
        # cb_orange_item10 선언
        cb_orange_item10.titleLabel.item.setFont(font)
        cb_orange_item10.getAxis("bottom").setStyle(tickFont=font)
        cb_orange_item10.getAxis("left").setStyle(tickFont=font)
        # cb_orange_item11 선언
        cb_orange_item11.titleLabel.item.setFont(font)
        cb_orange_item11.getAxis("bottom").setStyle(tickFont=font)
        cb_orange_item11.getAxis("left").setStyle(tickFont=font)
        # cb_orange_item12 선언
        cb_orange_item12.titleLabel.item.setFont(font)
        cb_orange_item12.getAxis("bottom").setStyle(tickFont=font)
        cb_orange_item12.getAxis("left").setStyle(tickFont=font)
        ################################################################################################################

    @pyqtSlot(str, name='show_shap')
    @pyqtSlot(object, object)
    def show_shap(self, symptom_db, compare_data):
        # all_shap : 전체 시나리오에 해당하는 shap_value를 가지고 있음.
        # symptom_db[0] : liner : appended time (axis-x) / symptom_db[1].iloc[1] : check_db (:line,2222)[1]
        text_convert_int = {'Normal': 0, 'Ab21_01': 1, 'Ab21_02': 2, 'Ab20_04': 3, 'Ab15_07': 4, 'Ab15_08': 5,
                            'Ab63_04': 6, 'Ab63_02': 7, 'Ab21_12': 8, 'Ab19_02': 9, 'Ab21_11': 10, 'Ab23_03': 11,
                            'Ab60_02': 12, 'Ab59_02': 13, 'Ab23_01': 14, 'Ab23_06': 15}

        self.undiagnosed_shap_result_abs = model_module.abnormal_procedure_classifier_2(scenario=text_convert_int[self.cb.currentText()[:7]])[self.cb.currentText()[:7]][0]

        red_range = self.undiagnosed_shap_result_abs[self.undiagnosed_shap_result_abs['probability'] >= 10]
        orange_range = self.undiagnosed_shap_result_abs[
            [self.undiagnosed_shap_result_abs['probability'].iloc[i] < 10 and self.undiagnosed_shap_result_abs['probability'].iloc[i] > 1 for i in
             range(len(self.undiagnosed_shap_result_abs['probability']))]]
        convert_red = {0: self.cb_red1, 1: self.cb_red2, 2: self.cb_red3, 3: self.cb_red4}
        convert_orange = {0: self.cb_orange1, 1: self.cb_orange2, 2: self.cb_orange3, 3: self.cb_orange4, 4: self.cb_orange5,
                          5: self.cb_orange6, 6: self.cb_orange7, 7: self.cb_orange8, 8: self.cb_orange9, 9: self.cb_orange10,
                          10: self.cb_orange11, 11: self.cb_orange12}

        red_del = [i for i in range(len(red_range), 4)]
        orange_del = [i for i in range(len(orange_range), 12)]
        '''
                [i for i in range(0, 4)] -> [0,1,2,3]
                [i for i in range(1, 4)] -> [1,2,3]
                [i for i in range(2, 4)] -> [2,3]
                [i for i in range(3, 4)] -> [3]
                [i for i in range(4, 4)] -> []
                ※ red_range에 포함되지 않는 red_button 지역 탐색
        '''

        [convert_red[i].setText(f'{red_range["describe"].iloc[i]} \n[{round(red_range["probability"].iloc[i], 2)}%]') for i in range(len(red_range))]
        [convert_red[i].setText('None\nParameter') for i in red_del]
        for i in range(len(red_range)):
            if red_range['sign'].iloc[i] == '+':
                convert_red[i].setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: red;')
            elif red_range['sign'].iloc[i] == '-':
                convert_red[i].setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: blue;')
        # [convert_red[i].setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: blue;') for i in range(len(red_range))]
        [convert_red[i].setStyleSheet('color : black;' 'background-color: light gray;') for i in red_del]

        [convert_orange[i].setText(f'{orange_range["describe"].iloc[i]} \n[{round(orange_range["probability"].iloc[i], 2)}%]') for i in range(len(orange_range))]
        [convert_orange[i].setText('None\nParameter') for i in orange_del]
        for i in range(len(orange_range)):
            if orange_range['sign'].iloc[i] == '+':
                convert_orange[i].setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: red;')
            elif orange_range['sign'].iloc[i] == '-':
                convert_orange[i].setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: blue;')
        [convert_orange[i].setStyleSheet('color : black;' 'background-color: light gray;') for i in orange_del]

        #####################################################################################################################################
        # 각 Button에 호환되는 Plotting 데이터 구축
        # Red1 Button
        if self.cb_red1.text().split()[0] != 'None':
            if self.cb_red1.isChecked():
                self.cb_red_plot_1.clear()
                self.cb_red_plot_1.setTitle(red_range['describe'].iloc[0])
                self.cb_red_plot_1.addLegend(offset=(-30, 20))
                self.cb_red_plot_1.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[red_range['variable'].iloc[0]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.cb_red_plot_1.plot(x=symptom_db[0], y=pd.DataFrame(compare_data[self.cb.currentText()[:7]])[red_range['variable'].iloc[0]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Red2 Button
        if self.cb_red2.text().split()[0] != 'None':
            if self.cb_red2.isChecked():
                self.cb_red_plot_2.clear()
                self.cb_red_plot_2.setTitle(red_range['describe'].iloc[1])
                self.cb_red_plot_2.addLegend(offset=(-30, 20))
                self.cb_red_plot_2.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[red_range['variable'].iloc[1]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.cb_red_plot_2.plot(x=symptom_db[0], y=pd.DataFrame(compare_data[self.cb.currentText()[:7]])[red_range['variable'].iloc[1]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Red3 Button
        if self.cb_red3.text().split()[0] != 'None':
            if self.cb_red3.isChecked():
                self.cb_red_plot_3.clear()
                self.cb_red_plot_3.setTitle(red_range['describe'].iloc[2])
                self.cb_red_plot_3.addLegend(offset=(-30, 20))
                self.cb_red_plot_3.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[red_range['variable'].iloc[2]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.cb_red_plot_3.plot(x=symptom_db[0], y=pd.DataFrame(compare_data[self.cb.currentText()[:7]])[red_range['variable'].iloc[2]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Red4 Button
        if self.cb_red4.text().split()[0] != 'None':
            if self.cb_red4.isChecked():
                self.cb_red_plot_4.clear()
                self.cb_red_plot_4.setTitle(red_range['describe'].iloc[3])
                self.cb_red_plot_4.addLegend(offset=(-30, 20))
                self.cb_red_plot_4.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[red_range['variable'].iloc[3]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.cb_red_plot_4.plot(x=symptom_db[0], y=pd.DataFrame(compare_data[self.cb.currentText()[:7]])[red_range['variable'].iloc[3]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange1 Button
        if self.cb_orange1.text().split()[0] != 'None':
            if self.cb_orange1.isChecked():
                self.cb_orange_plot_1.clear()
                self.cb_orange_plot_1.setTitle(orange_range['describe'].iloc[0])
                self.cb_orange_plot_1.addLegend(offset=(-30, 20))
                self.cb_orange_plot_1.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[0]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.cb_orange_plot_1.plot(x=symptom_db[0], y=pd.DataFrame(compare_data[self.cb.currentText()[:7]])[orange_range['variable'].iloc[0]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange2 Button
        if self.cb_orange2.text().split()[0] != 'None':
            if self.cb_orange2.isChecked():
                self.cb_orange_plot_2.clear()
                self.cb_orange_plot_2.setTitle(orange_range['describe'].iloc[1])
                self.cb_orange_plot_2.addLegend(offset=(-30, 20))
                self.cb_orange_plot_2.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[1]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.cb_orange_plot_2.plot(x=symptom_db[0], y=pd.DataFrame(compare_data[self.cb.currentText()[:7]])[orange_range['variable'].iloc[1]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange3 Button
        if self.cb_orange3.text().split()[0] != 'None':
            if self.cb_orange3.isChecked():
                self.cb_orange_plot_3.clear()
                self.cb_orange_plot_3.setTitle(orange_range['describe'].iloc[2])
                self.cb_orange_plot_3.addLegend(offset=(-30, 20))
                self.cb_orange_plot_3.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[2]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.cb_orange_plot_3.plot(x=symptom_db[0], y=pd.DataFrame(compare_data[self.cb.currentText()[:7]])[orange_range['variable'].iloc[2]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange4 Button
        if self.cb_orange4.text().split()[0] != 'None':
            if self.cb_orange4.isChecked():
                self.cb_orange_plot_4.clear()
                self.cb_orange_plot_4.setTitle(orange_range['describe'].iloc[3])
                self.cb_orange_plot_4.addLegend(offset=(-30, 20))
                self.cb_orange_plot_4.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[3]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.cb_orange_plot_4.plot(x=symptom_db[0], y=pd.DataFrame(compare_data[self.cb.currentText()[:7]])[orange_range['variable'].iloc[3]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange5 Button
        if self.cb_orange5.text().split()[0] != 'None':
            if self.cb_orange5.isChecked():
                self.cb_orange_plot_5.clear()
                self.cb_orange_plot_5.setTitle(orange_range['describe'].iloc[4])
                self.cb_orange_plot_5.addLegend(offset=(-30, 20))
                self.cb_orange_plot_5.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[4]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.cb_orange_plot_5.plot(x=symptom_db[0], y=pd.DataFrame(compare_data[self.cb.currentText()[:7]])[orange_range['variable'].iloc[4]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange6 Button
        if self.cb_orange6.text().split()[0] != 'None':
            if self.cb_orange6.isChecked():
                self.cb_orange_plot_6.clear()
                self.cb_orange_plot_6.setTitle(orange_range['describe'].iloc[5])
                self.cb_orange_plot_6.addLegend(offset=(-30, 20))
                self.cb_orange_plot_6.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[5]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.cb_orange_plot_6.plot(x=symptom_db[0], y=pd.DataFrame(compare_data[self.cb.currentText()[:7]])[orange_range['variable'].iloc[5]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange7 Button
        if self.cb_orange7.text().split()[0] != 'None':
            if self.cb_orange7.isChecked():
                self.cb_orange_plot_7.clear()
                self.cb_orange_plot_7.setTitle(orange_range['describe'].iloc[6])
                self.cb_orange_plot_7.addLegend(offset=(-30, 20))
                self.cb_orange_plot_7.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[6]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.cb_orange_plot_7.plot(x=symptom_db[0], y=pd.DataFrame(compare_data[self.cb.currentText()[:7]])[orange_range['variable'].iloc[6]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange8 Button
        if self.cb_orange8.text().split()[0] != 'None':
            if self.cb_orange8.isChecked():
                self.cb_orange_plot_8.clear()
                self.cb_orange_plot_8.setTitle(orange_range['describe'].iloc[7])
                self.cb_orange_plot_8.addLegend(offset=(-30, 20))
                self.cb_orange_plot_8.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[7]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.cb_orange_plot_8.plot(x=symptom_db[0], y=pd.DataFrame(compare_data[self.cb.currentText()[:7]])[orange_range['variable'].iloc[7]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange9 Button
        if self.cb_orange9.text().split()[0] != 'None':
            if self.cb_orange9.isChecked():
                self.cb_orange_plot_9.clear()
                self.cb_orange_plot_9.setTitle(orange_range['describe'].iloc[8])
                self.cb_orange_plot_9.addLegend(offset=(-30, 20))
                self.cb_orange_plot_9.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[8]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.cb_orange_plot_9.plot(x=symptom_db[0], y=pd.DataFrame(compare_data[self.cb.currentText()[:7]])[orange_range['variable'].iloc[8]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange10 Button
        if self.cb_orange10.text().split()[0] != 'None':
            if self.cb_orange10.isChecked():
                self.cb_orange_plot_10.clear()
                self.cb_orange_plot_10.setTitle(orange_range['describe'].iloc[9])
                self.cb_orange_plot_10.addLegend(offset=(-30, 20))
                self.cb_orange_plot_10.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[9]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.cb_orange_plot_10.plot(x=symptom_db[0], y=pd.DataFrame(compare_data[self.cb.currentText()[:7]])[orange_range['variable'].iloc[9]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange11 Button
        if self.cb_orange11.text().split()[0] != 'None':
            if self.cb_orange11.isChecked():
                self.cb_orange_plot_11.clear()
                self.cb_orange_plot_11.setTitle(orange_range['describe'].iloc[10])
                self.cb_orange_plot_11.addLegend(offset=(-30, 20))
                self.cb_orange_plot_11.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[10]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.cb_orange_plot_11.plot(x=symptom_db[0], y=pd.DataFrame(compare_data[self.cb.currentText()[:7]])[orange_range['variable'].iloc[10]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange12 Button
        if self.cb_orange12.text().split()[0] != 'None':
            if self.cb_orange12.isChecked():
                self.cb_orange_plot_12.clear()
                self.cb_orange_plot_12.setTitle(orange_range['describe'].iloc[11])
                self.cb_orange_plot_12.addLegend(offset=(-30, 20))
                self.cb_orange_plot_12.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['variable'].iloc[11]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
                self.cb_orange_plot_12.plot(x=symptom_db[0], y=pd.DataFrame(compare_data[self.cb.currentText()[:7]])[orange_range['variable'].iloc[11]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        [convert_red[i].setCheckable(True) for i in range(4)]
        [convert_orange[i].setCheckable(True) for i in range(12)]

    @pyqtSlot(str, name='show_another_result_table')
    @pyqtSlot(object)
    def show_another_result_table(self, line):
        # shap_add_des['variable'] : 변수 이름 / shap_add_des[0] : shap value
        # shap_add_des['describe'] : 변수에 대한 설명 / shap_add_des['probability'] : shap value를 확률로 환산한 값
        # text_convert_int = {'Normal': 0, 'Ab21_01': 1, 'Ab21_02': 2, 'Ab20_04': 3, 'Ab15_07': 4, 'Ab15_08': 5,
        #                     'Ab63_04': 6, 'Ab63_02': 7, 'Ab21_12': 8, 'Ab19_02': 9, 'Ab21_11': 10, 'Ab23_03': 11,
        #                     'Ab60_02': 12, 'Ab59_02': 13, 'Ab23_01': 14, 'Ab23_06': 15}
        # self.undiagnosed_shap_result_abs = model_module.abnormal_procedure_classifier_2(scenario=text_convert_int[self.cb.currentText()[:7]])[self.cb.currentText()[:7]][0]

        self.combo_tableWidget.clear()
        self.combo_tableWidget.setRowCount(len(self.undiagnosed_shap_result_abs))
        self.combo_tableWidget.setColumnCount(3)
        self.combo_tableWidget.setHorizontalHeaderLabels(["value_name", 'probability', 'describe'])

        header = self.combo_tableWidget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

        [self.combo_tableWidget.setItem(i, 0, QTableWidgetItem(f"{self.undiagnosed_shap_result_abs['variable'][i]}")) for i in
         range(len(self.undiagnosed_shap_result_abs['variable']))]
        [self.combo_tableWidget.setItem(i, 1, QTableWidgetItem(f"{round(self.undiagnosed_shap_result_abs['probability'][i], 2)}%")) for i in
         range(len(self.undiagnosed_shap_result_abs['probability']))]
        [self.combo_tableWidget.setItem(i, 2, QTableWidgetItem(f"{self.undiagnosed_shap_result_abs['describe'][i]}")) for i in
         range(len(self.undiagnosed_shap_result_abs['describe']))]

        delegate = AlignDelegate(self.combo_tableWidget)
        self.combo_tableWidget.setItemDelegate(delegate)

    @pyqtSlot(bool)
    def show_anoter_table(self):
        self.combo_tableWidget.show()

    def cb_red1_plot(self):
        if self.cb_red1.isChecked():
            if self.cb_red1.text().split()[0] != 'None':
                self.cb_red_plot_1.show()


    def cb_red2_plot(self):
        if self.cb_red2.isChecked():
            if self.cb_red2.text().split()[0] != 'None':
                self.cb_red_plot_2.show()

    def cb_red3_plot(self):
        if self.cb_red3.isChecked():
            if self.cb_red3.text().split()[0] != 'None':
                self.cb_red_plot_3.show()

    def cb_red4_plot(self):
        if self.cb_red4.isChecked():
            if self.cb_red4.text().split()[0] != 'None':
                self.cb_red_plot_4.show()

    def cb_orange1_plot(self):
        if self.cb_orange1.isChecked():
            if self.cb_orange1.text().split()[0] != 'None':
                self.cb_orange_plot_1.show()

    def cb_orange2_plot(self):
        if self.cb_orange2.isChecked():
            if self.cb_orange2.text().split()[0] != 'None':
                self.cb_orange_plot_2.show()

    def cb_orange3_plot(self):
        if self.cb_orange3.isChecked():
            if self.cb_orange3.text().split()[0] != 'None':
                self.cb_orange_plot_3.show()

    def cb_orange4_plot(self):
        if self.cb_orange4.isChecked():
            if self.cb_orange4.text().split()[0] != 'None':
                self.cb_orange_plot_4.show()

    def cb_orange5_plot(self):
        if self.cb_orange5.isChecked():
            if self.cb_orange5.text().split()[0] != 'None':
                self.cb_orange_plot_5.show()

    def cb_orange6_plot(self):
        if self.cb_orange6.isChecked():
            if self.cb_orange6.text().split()[0] != 'None':
                self.cb_orange_plot_6.show()

    def cb_orange7_plot(self):
        if self.cb_orange7.isChecked():
            if self.cb_orange7.text().split()[0] != 'None':
                self.cb_orange_plot_7.show()

    def cb_orange8_plot(self):
        if self.cb_orange8.isChecked():
            if self.cb_orange8.text().split()[0] != 'None':
                self.cb_orange_plot_8.show()

    def cb_orange9_plot(self):
        if self.cb_orange9.isChecked():
            if self.cb_orange9.text().split()[0] != 'None':
                self.cb_orange_plot_9.show()

    def cb_orange10_plot(self):
        if self.cb_orange10.isChecked():
            if self.cb_orange10.text().split()[0] != 'None':
                self.cb_orange_plot_10.show()

    def cb_orange11_plot(self):
        if self.cb_orange11.isChecked():
            if self.cb_orange11.text().split()[0] != 'None':
                self.cb_orange_plot_11.show()

    def cb_orange12_plot(self):
        if self.cb_orange12.isChecked():
            if self.cb_orange12.text().split()[0] != 'None':
                self.cb_orange_plot_12.show()

class log_system(QWidget):
    def __init__(self):
        super().__init__()
        # 서브 인터페이스 초기 설정
        self.setObjectName('LogWidget')
        self.setStyleSheet("""#LogWidget {background-color: black;}""")
        self.setWindowTitle('Log Display')
        self.setWindowIcon(QIcon('./icon/log_icon.png'))
        self.setGeometry(300, 300, 1000, 600)
        font = QFont('Times New Roman')
        # 레이아웃 구성
        final_layout = QVBoxLayout()
        summury_layout = QHBoxLayout()
        total_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        graph_layout = QVBoxLayout()
        self.log_display = QTextEdit()
        # self.log_display.setStyleSheet('background-color: black;')
        self.log_display.setReadOnly(True)
        self.log_display.ensureCursorVisible()
        self.log_display.setFont(QFont('Times new roman'))
        self.trained_status = QPushButton('Training Status')
        self.trained_status.setFont(QFont('Times new roman'))
        self.trained_status.setCheckable(True)
        self.abnormal_status = QPushButton('Scenario Diagnosis')
        self.abnormal_status.setFont(QFont('Times new roman'))
        self.abnormal_status.setCheckable(True)
        self.verif_status = QPushButton("Verification Result")
        self.verif_status.setFont(QFont('Times new roman'))
        self.verif_status.setCheckable(True)
        self.train_plot = pyqtgraph.PlotWidget(title='Training Status Plot')
        self.train_plot.showGrid(x=True, y=True, alpha=0.2)
        train_item = self.train_plot.getPlotItem()
        train_item.titleLabel.item.setFont(font)
        train_item.getAxis("bottom").setStyle(tickFont=font)
        train_item.getAxis("left").setStyle(tickFont=font)
        self.scenario_plot = pyqtgraph.PlotWidget(title='Diagnosis Scenario Plot')
        self.scenario_plot.showGrid(x=True, y=True, alpha=0.2)
        scenario_item = self.scenario_plot.getPlotItem()
        scenario_item.titleLabel.item.setFont(font)
        scenario_item.getAxis("bottom").setStyle(tickFont=font)
        scenario_item.getAxis("left").setStyle(tickFont=font)
        self.verif_plot = pyqtgraph.PlotWidget(title='Verification Result Plot')
        self.verif_plot.showGrid(x=True, y=True, alpha=0.2)
        verif_item = self.verif_plot.getPlotItem()
        verif_item.titleLabel.item.setFont(font)
        verif_item.getAxis("bottom").setStyle(tickFont=font)
        verif_item.getAxis("left").setStyle(tickFont=font)

        btn_layout.addWidget(self.trained_status)
        btn_layout.addWidget(self.abnormal_status)
        btn_layout.addWidget(self.verif_status)
        graph_layout.addWidget(self.train_plot)
        graph_layout.addWidget(self.scenario_plot)
        graph_layout.addWidget(self.verif_plot)
        total_layout.addWidget(self.log_display)
        summury_layout.addLayout(total_layout)
        summury_layout.addLayout(graph_layout)
        final_layout.addLayout(summury_layout)
        final_layout.addLayout(btn_layout)
        self.setLayout(final_layout)

        # Training status log system
        self.train_log = QTextEdit()
        self.train_log.resize(400,500)
        self.train_log.setWindowTitle('Training Status Log')
        self.train_log.setReadOnly(True)
        self.train_log.setFont(QFont('Times new roman'))
        self.trained_status.clicked.connect(self.train_log_control)
        # Scenario Diagnosis log system
        self.scenario_log = QTextEdit()
        self.scenario_log.resize(400,500)
        self.scenario_log.setWindowTitle('Scenario Diagnosis Log')
        self.scenario_log.setReadOnly(True)
        self.scenario_log.setFont(QFont('Times new roman'))
        self.abnormal_status.clicked.connect(self.scenario_log_control)
        # Verification Result log system
        self.verif_log = QTextEdit()
        self.verif_log.resize(400,500)
        self.verif_log.setWindowTitle('Diagnosis Verification Log')
        self.verif_log.setReadOnly(True)
        self.verif_log.setFont(QFont('Times new roman'))
        self.verif_status.clicked.connect(self.verif_log_control)

    @pyqtSlot(object)
    def total_log(self, logging_db):
        # logging_db = {'time':[], 'train':[], 'scenario':{'name':[], 'probability':[]}, 'success':[]} Structure
        alertHtml = "<font color=\"Red\">";
        notifyHtml = "<font color=\"Green\">";
        updateHtml = "<font color=\"Blue\">";
        text_set = {0: {'name': 'Normal', 'contents': 'Normal'}, 1: {'name': 'Ab21_01', 'contents': '가압기 압력 채널 고장 "고"'}, 2: {'name': 'Ab21_02', 'contents': '가압기 압력 채널 고장 "저"'},
                    3: {'name': 'Ab20_04', 'contents': '가압기 수위 채널 고장 "저"'}, 4: {'name': 'Ab15_07', 'contents': '증기발생기 수위 채널 고장 "저"'}, 5: {'name': 'Ab15_08', 'contents': '증기발생기 수위 채널 고장 "고"'},
                    6: {'name': 'Ab63_04', 'contents': '제어봉 낙하'}, 7: {'name': 'Ab63_02', 'contents': '제어봉의 계속적인 삽입'}, 8: {'name': 'Ab21_12', 'contents': 'Pressurizer PORV opening'},
                    9: {'name': 'Ab19_02', 'contents': '가압기 안전밸브 고장'}, 10: {'name': 'Ab21_11', 'contents': 'Opening of PRZ spray valve'}, 11: {'name': 'Ab23_03', 'contents': '1차기기 냉각수 계통으로 누설 "CVCS->CCW"'},
                    12: {'name': 'Ab60_02', 'contents': '재생열교환기 전단부위 파열'}, 13: {'name': 'Ab59_02', 'contents': '충전수 유량조절밸브 후단 누설'}, 14: {'name': 'Ab23_01', 'contents': '1차기기 냉각수 계통으로 누설 "RCS->CCW"'},
                    15: {'name': 'Ab23_06', 'contents': 'Steam generator tube rupture'}}
        train_convert = {0: 'Trained condition', 1: 'Untrained condition'}
        verif_convert = {0: 'Diagnosis Success', 1: 'Diagnosis Failure'}
        self.log_display.clear()
        self.train_log.clear()
        self.scenario_log.clear()
        self.verif_log.clear()
        self.train_plot.clear()
        self.scenario_plot.clear()
        self.verif_plot.clear()
        self.train_plot.plot(x=logging_db["time"], y=logging_db["train"], pen=pyqtgraph.mkPen('k', width=2))
        self.scenario_plot.plot(x=logging_db["time"], y=logging_db["scenario"]["name"], pen=pyqtgraph.mkPen('k', width=2))
        self.verif_plot.plot(x=logging_db["time"], y=logging_db["success"], pen=pyqtgraph.mkPen('k', width=2))
        for i in range(len(logging_db["time"])):
            if i == 0:
                self.log_display.append('Log system start')
                self.log_display.append('Initial Result')
                if logging_db["train"][i] == 0: # trained condition
                    self.log_display.append(f'Time: {logging_db["time"][i]}sec,  {train_convert[logging_db["train"][i]]}')
                    self.log_display.append(notifyHtml+'Stable State')
                elif logging_db["train"][i] == 1:
                    self.log_display.append(f'Time: {logging_db["time"][i]}sec,  {train_convert[logging_db["train"][i]]}')
                    self.log_display.append(alertHtml+'Warning: Untrained Condition')
                if logging_db["scenario"]["name"][i] == 0:
                    self.log_display.append(f'Time: {logging_db["time"][i]}sec,  {text_set[logging_db["scenario"]["name"][i]]["contents"]} [{logging_db["scenario"]["probability"][i]}%]')
                    self.log_display.append(notifyHtml+'Stable State')
                    if logging_db["scenario"]["probability"][i] <= 90:
                        self.log_display.append(alertHtml+'Warning: The probability of diagnosis is low')
                else:
                    self.log_display.append(f'Time: {logging_db["time"][i]}sec,  {text_set[logging_db["scenario"]["name"][i]]["contents"]} [{logging_db["scenario"]["probability"][i]}%]')
                    self.log_display.append(alertHtml+'Abnormal operating condition')
                    if logging_db["scenario"]["probability"][i] <= 90:
                        self.log_display.append(alertHtml+'Warning: The probability of diagnosis is low')
                if logging_db["success"][i] == 0:
                    self.log_display.append(f'Time: {logging_db["time"][i]}sec,  {verif_convert[logging_db["success"][i]]}')
                    self.log_display.append(notifyHtml+'Stable State')
                elif logging_db["success"][i] == 1:
                    self.log_display.append(f'Time: {logging_db["time"][i]}sec,  {verif_convert[logging_db["success"][i]]}')
                    self.log_display.append(alertHtml + 'Warning: Diagnosis Failure')
                self.log_display.append('------------------------------------------------------------------------------')
            else:
                if logging_db["train"][i-1] != logging_db["train"][i]:
                    self.log_display.append(updateHtml+'Log Update')
                    if logging_db["train"][i] == 0:
                        self.log_display.append(f'Time: {logging_db["time"][i]}sec,  {train_convert[logging_db["train"][i]]}')
                        self.log_display.append(notifyHtml+'Stable State')
                    elif logging_db["train"][i] == 1:
                        self.log_display.append(f'Time: {logging_db["time"][i]}sec,  {train_convert[logging_db["train"][i]]}')
                        self.log_display.append(alertHtml + 'Warning: Untrained Condition')
                if logging_db["scenario"]["name"][i-1] != logging_db["scenario"]["name"][i]:
                    self.log_display.append(updateHtml + 'Log Update')
                    if logging_db["scenario"]["name"][i] == 0:
                        self.log_display.append(f'Time: {logging_db["time"][i]}sec,  {text_set[logging_db["scenario"]["name"][i]]["contents"]} [{logging_db["scenario"]["probability"][i]}%]')
                        self.log_display.append(notifyHtml+'Stable State')
                        if logging_db["scenario"]["probability"][i] <= 90:
                            self.log_display.append(alertHtml + 'Warning: The probability of diagnosis is low')
                    else:
                        self.log_display.append(f'Time: {logging_db["time"][i]}sec,  {text_set[logging_db["scenario"]["name"][i]]["contents"]} [{logging_db["scenario"]["probability"][i]}%]')
                        self.log_display.append(alertHtml + 'Abnormal operating condition')
                        if logging_db["scenario"]["probability"][i] <= 90:
                            self.log_display.append(alertHtml + 'Warning: The probability of diagnosis is low')
                if logging_db["success"][i-1] != logging_db["success"][i]:
                    self.log_display.append(updateHtml + 'Log Update')
                    if logging_db["success"][i] == 0:
                        self.log_display.append(f'Time: {logging_db["time"][i]}sec,  {verif_convert[logging_db["success"][i]]}')
                        self.log_display.append(notifyHtml+'Stable State')
                    elif logging_db["success"][i] == 1:
                        self.log_display.append(f'Time: {logging_db["time"][i]}sec,  {verif_convert[logging_db["success"][i]]}')
                        self.log_display.append(alertHtml + 'Warning: Diagnosis Failure')
            if self.trained_status.isChecked():
                if logging_db["train"][i] == 0:
                    self.train_log.append(notifyHtml+f'Time: {logging_db["time"][i]}sec,  {train_convert[logging_db["train"][i]]}')
                elif logging_db["train"][i] == 1:
                    self.train_log.append(alertHtml + f'Time: {logging_db["time"][i]}sec,  {train_convert[logging_db["train"][i]]}')
            if self.abnormal_status.isChecked():
                if logging_db["scenario"]["name"][i] == 0:
                    self.scenario_log.append(notifyHtml+f'Time: {logging_db["time"][i]}sec,  {text_set[logging_db["scenario"]["name"][i]]["contents"]} [{logging_db["scenario"]["probability"][i]}%]')
                else:
                    self.scenario_log.append(alertHtml+f'Time: {logging_db["time"][i]}sec,  {text_set[logging_db["scenario"]["name"][i]]["contents"]} [{logging_db["scenario"]["probability"][i]}%]')
            if self.verif_status.isChecked():
                if logging_db["success"][i] == 0:
                    self.verif_log.append(notifyHtml+f'Time: {logging_db["time"][i]}sec,  {verif_convert[logging_db["success"][i]]}')
                elif logging_db["success"][i] == 1:
                    self.verif_log.append(alertHtml+f'Time: {logging_db["time"][i]}sec,  {verif_convert[logging_db["success"][i]]}')

    def train_log_control(self):
        if self.trained_status.isChecked():
            self.train_log.show()
        else:
            self.train_log.close()

    def scenario_log_control(self):
        if self.abnormal_status.isChecked():
            self.scenario_log.show()
        else:
            self.scenario_log.close()

    def verif_log_control(self):
        if self.verif_status.isChecked():
            self.verif_log.show()
        else:
            self.verif_log.close()



if __name__ == "__main__":
    # 데이터 로드 Fnc
    def Load_pickle(file_name):
        with open(f'./DataBase/CNS_db/pkl/{file_name}', 'rb') as f:
            load_pkl = pickle.load(f)
        return load_pkl

    # 변수 명, Pkl 파일 명
    Want_ = [('Normal', 'normal.pkl'), ('Ab21_01', 'ab21-01_170.pkl'), ('Ab21_02', 'ab21-02-152.pkl'), ('Ab20_04', 'ab20-04_6.pkl'), ('Ab15_07', 'ab15-07_1002.pkl'), ('Ab15_08', 'ab15-08_1071.pkl')
             , ('Ab63_04', 'ab63-04_113.pkl'), ('Ab63_02', 'ab63-02_5(4;47트립).pkl'), ('Ab21_12', 'ab21-12_34 (06;06트립).pkl'), ('Ab19_02', 'ab19-02-17(05;55트립).pkl'), ('Ab21_11', 'ab21-11_62(06;29트립).pkl')
             , ('Ab23_03', 'ab23-03-77.pkl'), ('Ab60_02', 'ab60-02_306.pkl'), ('Ab59_02', 'ab59-02-1055.pkl'), ('Ab23_01', 'ab23-01_30004(3;11_trip).pkl'), ('Ab23_06', 'ab23-06_30004(03;28 trip).pkl')]  # TODO ... 아래 파일명, 변수명 튜플로 입력.
    # 데이터 로드
    Loaded_DB = {Val_name: Load_pickle(Loaded_pkl) for Val_name, Loaded_pkl in Want_}

    print('데이터 불러오기를 완료하여 모델 불러오기로 이행합니다.')

    # 모델 로드
    model_module = Model_module()
    model_module.load_model()
    print('모델 불러오기를 완료하여 GUI로 이행합니다.')
    app = QApplication(sys.argv)
    form = Mainwindow(Loaded_DB)
    exit(app.exec_())