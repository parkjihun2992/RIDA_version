import numpy as np
import pandas as pd
import sys
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pyqtgraph
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtTest import *
from Model_module import Model_module
from Data_module import Data_module
# from Sub_widget import another_result_explain


class Worker(QObject):
    # Signal을 보낼 그릇을 생성# #############
    train_value = pyqtSignal(object)
    # nor_ab_value = pyqtSignal(object)
    procedure_value = pyqtSignal(object)
    verif_value = pyqtSignal(object)
    timer = pyqtSignal(object)
    symptom_db = pyqtSignal(object)
    shap = pyqtSignal(object)
    plot_db = pyqtSignal(object)
    display_ex = pyqtSignal(object, object, object)
    another_shap = pyqtSignal(object, object, object)
    another_shap_table = pyqtSignal(object)

    ##########################################

    @pyqtSlot(object)
    def generate_db(self):
        test_db = input('구현할 시나리오를 입력해주세요 : ')
        print(f'입력된 시나리오 : {test_db}를 실행합니다.')

        Model_module()  # model module 내의 빈행렬 초기화
        data_module = Data_module()
        db, check_db = data_module.load_data(file_name=test_db)  # test_db 불러오기
        data_module.data_processing()  # Min-Max o, 2 Dimension
        liner = []
        plot_data = []
        normal_data = []
        compare_data = {'Normal':[], 'Ab21-01':[], 'Ab21-02':[], 'Ab20-04':[], 'Ab15-07':[], 'Ab15-08':[], 'Ab63-04':[], 'Ab63-02':[], 'Ab21-12':[], 'Ab19-02':[], 'Ab21-11':[], 'Ab23-03':[], 'Ab60-02':[], 'Ab59-02':[], 'Ab23-01':[], 'Ab23-06':[]}
        for line in range(np.shape(db)[0]):
            QTest.qWait(0.01)
            print(np.shape(db)[0], line)
            data = np.array([data_module.load_real_data(row=line)])
            liner.append(line)
            check_data, check_parameter = data_module.load_real_check_data(row=line)
            plot_data.append(check_data[0])
            try: normal_data.append(normal_db.iloc[line])
            except: pass
            try: compare_data['Normal'].append(normal_db.iloc[line])
            except: pass
            try: compare_data['Ab21-01'].append(ab21_01.iloc[line])
            except: pass
            try: compare_data['Ab21-02'].append(ab21_02.iloc[line])
            except: pass
            try: compare_data['Ab20-04'].append(ab20_04.iloc[line])
            except: pass
            try: compare_data['Ab15-07'].append(ab15_07.iloc[line])
            except: pass
            try: compare_data['Ab15-08'].append(ab15_08.iloc[line])
            except: pass
            try: compare_data['Ab63-04'].append(ab63_04.iloc[line])
            except: pass
            try: compare_data['Ab63-02'].append(ab63_02.iloc[line])
            except: pass
            try: compare_data['Ab21-12'].append(ab21_12.iloc[line])
            except: pass
            try: compare_data['Ab19-02'].append(ab19_02.iloc[line])
            except: pass
            try: compare_data['Ab21-11'].append(ab21_11.iloc[line])
            except: pass
            try: compare_data['Ab23-03'].append(ab23_03.iloc[line])
            except: pass
            try: compare_data['Ab60-02'].append(ab60_02.iloc[line])
            except: pass
            try: compare_data['Ab59-02'].append(ab59_02.iloc[line])
            except: pass
            try: compare_data['Ab23-01'].append(ab23_01.iloc[line])
            except: pass
            try: compare_data['Ab23-06'].append(ab23_06.iloc[line])
            except: pass
            if np.shape(data) == (1, 10, 46):
                dim2 = np.array(data_module.load_scaled_data(row=line - 9))  # 2차원 scale
                # check_data, check_parameter = data_module.load_real_check_data(row=line - 8)
                # plot_data.append(check_data[0])
                train_untrain_reconstruction_error, train_untrain_error = model_module.train_untrain_classifier(data=data)
                # normal_abnormal_reconstruction_error = model_module.normal_abnormal_classifier(data=data)
                abnormal_procedure_result, abnormal_procedure_prediction, shap_add_des, shap_value = model_module.abnormal_procedure_classifier(data=dim2)
                abnormal_verif_reconstruction_error, verif_threshold, abnormal_verif_error = model_module.abnormal_procedure_verification(data=data)
                self.train_value.emit(train_untrain_error)
                # self.nor_ab_value.emit(np.argmax(abnormal_procedure_result[line-9], axis=1)[0])
                self.procedure_value.emit(np.argmax(abnormal_procedure_prediction, axis=1)[0])
                self.verif_value.emit([abnormal_verif_error, verif_threshold])
                self.timer.emit([line, check_parameter])
                self.symptom_db.emit([np.argmax(abnormal_procedure_prediction, axis=1)[0], check_parameter])
                self.shap.emit(shap_add_des)
                self.plot_db.emit([liner, plot_data])
                self.display_ex.emit(shap_add_des, [liner, plot_data], normal_data)
                self.another_shap.emit(shap_value, [liner, plot_data], compare_data)
                self.another_shap_table.emit(shap_value)


class AlignDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super(AlignDelegate, self).initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter


class Mainwindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Abnormal Diagnosis for NPP")
        self.setGeometry(150, 50, 1700, 800)
        # 그래프 초기조건
        pyqtgraph.setConfigOption("background", "w")
        pyqtgraph.setConfigOption("foreground", "k")
        #############################################
        self.selected_para = pd.read_csv('./DataBase/Final_parameter.csv')
        # GUI part 1 Layout (진단 부분 통합)
        layout_left = QVBoxLayout()

        # 영 번째 그룹 설정 (Time and Power)
        gb_0 = QGroupBox("Training Status")  # 영 번째 그룹 이름 설정
        layout_left.addWidget(gb_0)  # 전체 틀에 영 번째 그룹 넣기
        gb_0_layout = QBoxLayout(QBoxLayout.LeftToRight)  # 영 번째 그룹 내용을 넣을 레이아웃 설정

        # 첫 번째 그룹 설정
        gb_1 = QGroupBox("Training Status")  # 첫 번째 그룹 이름 설정
        layout_left.addWidget(gb_1)  # 전체 틀에 첫 번째 그룹 넣기
        gb_1_layout = QBoxLayout(QBoxLayout.LeftToRight)  # 첫 번째 그룹 내용을 넣을 레이아웃 설정

        # 두 번째 그룹 설정
        gb_2 = QGroupBox('NPP Status')
        layout_left.addWidget(gb_2)
        gb_2_layout = QBoxLayout(QBoxLayout.LeftToRight)

        # 세 번째 그룹 설정
        gb_3 = QGroupBox(self)
        layout_left.addWidget(gb_3)
        gb_3_layout = QBoxLayout(QBoxLayout.LeftToRight)

        # 네 번째 그룹 설정
        gb_4 = QGroupBox('Predicted Result Verification')
        layout_left.addWidget(gb_4)
        gb_4_layout = QBoxLayout(QBoxLayout.LeftToRight)

        # 다섯 번째 그룹 설정
        gb_5 = QGroupBox('Symptom check in scenario')
        layout_left.addWidget(gb_5)
        gb_5_layout = QBoxLayout(QBoxLayout.TopToBottom)

        # Spacer 추가
        # layout_part1.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # 영 번째 그룹 내용
        self.time_label = QLabel(self)
        self.power_label = QPushButton(self)

        # 첫 번째 그룹 내용
        # Trained / Untrained condition label
        self.trained_label = QPushButton('Trained')
        self.Untrained_label = QPushButton('Untrained')

        # 두 번째 그룹 내용
        self.normal_label = QPushButton('Normal')
        self.abnormal_label = QPushButton('Abnormal')

        # 세 번째 그룹 내용
        self.name_procedure = QLabel('Number of Procedure: ')
        self.num_procedure = QLineEdit(self)
        self.num_procedure.setAlignment(Qt.AlignCenter)
        self.name_scnario = QLabel('Name of Procedure: ')
        self.num_scnario = QLineEdit(self)
        self.num_scnario.setAlignment(Qt.AlignCenter)

        # 네 번째 그룹 내용
        self.success_label = QPushButton('Diagnosis Success')
        self.failure_label = QPushButton('Diagnosis Failure')

        # 다섯 번째 그룹 내용
        self.symptom_name = QLabel(self)
        self.symptom1 = QCheckBox(self)
        self.symptom2 = QCheckBox(self)
        self.symptom3 = QCheckBox(self)
        self.symptom4 = QCheckBox(self)
        self.symptom5 = QCheckBox(self)
        self.symptom6 = QCheckBox(self)

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
        gb_2.setLayout(gb_2_layout)

        # 세 번째 그룹 내용 입력
        gb_3_layout.addWidget(self.name_procedure)
        gb_3_layout.addWidget(self.num_procedure)
        gb_3_layout.addWidget(self.name_scnario)
        gb_3_layout.addWidget(self.num_scnario)
        gb_3.setLayout(gb_3_layout)

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
        # layout_part1.addWidget(self.start_btn)

        self.tableWidget = QTableWidget(0, 0)
        self.tableWidget.setFixedHeight(500)
        self.tableWidget.setFixedWidth(800)

        # Plot 구현
        self.plot_1 = pyqtgraph.PlotWidget(title=self)
        self.plot_2 = pyqtgraph.PlotWidget(title=self)
        self.plot_3 = pyqtgraph.PlotWidget(title=self)
        self.plot_4 = pyqtgraph.PlotWidget(title=self)

        # Explanation Alarm 구현
        red_alarm = QGroupBox('Main basis for diagnosis')
        red_alarm_layout = QGridLayout()
        orange_alarm = QGroupBox('Sub basis for diagnosis')
        orange_alarm_layout = QGridLayout()
        # Display Button 생성
        self.red1 = QPushButton(self)
        self.red2 = QPushButton(self)
        self.red3 = QPushButton(self)
        self.red4 = QPushButton(self)
        self.orange1 = QPushButton(self)
        self.orange2 = QPushButton(self)
        self.orange3 = QPushButton(self)
        self.orange4 = QPushButton(self)
        self.orange5 = QPushButton(self)
        self.orange6 = QPushButton(self)
        self.orange7 = QPushButton(self)
        self.orange8 = QPushButton(self)
        self.orange9 = QPushButton(self)
        self.orange10 = QPushButton(self)
        self.orange11 = QPushButton(self)
        self.orange12 = QPushButton(self)
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
        layout_part1 = QVBoxLayout()
        detail_part = QHBoxLayout()
        detailed_table = QPushButton('Detail Explanation [Table]')
        self.another_classification = QPushButton('Why other scenarios were not chosen')
        detail_part.addWidget(detailed_table)
        detail_part.addWidget(self.another_classification)
        alarm_main = QVBoxLayout()
        alarm_main.addWidget(red_alarm)
        alarm_main.addWidget(orange_alarm)
        layout_part1.addLayout(layout_left)
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
        total_layout.addLayout(layout_base)
        total_layout.addWidget(self.start_btn)

        self.setLayout(total_layout)  # setLayout : 최종 출력될 GUI 화면을 결정


        # Threading Part##############################################################################################################
        # 데이터 연산 부분 Thread화
        self.worker = Worker()
        self.worker_thread = QThread()

        # Signal을 Main Thread 내의 함수와 연결
        self.worker.train_value.connect(self.Determine_train)
        self.worker.procedure_value.connect(self.Determine_abnormal)
        self.worker.procedure_value.connect(self.Determine_procedure)
        self.worker.verif_value.connect(self.verifit_result)
        self.worker.timer.connect(self.time_display)
        self.worker.symptom_db.connect(self.procedure_satisfaction)
        # self.worker.shap.connect(self.explain_result)
        self.worker.plot_db.connect(self.plotting)
        self.worker.display_ex.connect(self.display_explain)





        self.worker.moveToThread(self.worker_thread) # Worker class를 Thread로 이동
        # self.worker_thread.started.connect(lambda: self.worker.generate_db())
        self.start_btn.clicked.connect(lambda: self.worker.generate_db())  # 누르면 For문 실행
        self.worker_thread.start()
        # Threading Part##############################################################################################################

        # 이벤트 처리 ----------------------------------------------------------------------------------------------------
        detailed_table.clicked.connect(self.show_table)
        self.another_classification.clicked.connect(self.show_another_result)

        # Button 클릭 연동 이벤트 처리
        convert_red_btn = {0: self.red1, 1: self.red2, 2: self.red3, 3: self.red4} # Red Button
        convert_red_plot = {0: self.red1_plot, 1: self.red2_plot, 2: self.red3_plot, 3: self.red4_plot} #

        convert_orange_btn = {0: self.orange1, 1: self.orange2, 2: self.orange3, 3: self.orange4, 4: self.orange5,
                              5: self.orange6, 6: self.orange7, 7: self.orange8, 8: self.orange9, 9: self.orange10,
                              10: self.orange11, 11: self.orange12} # Orange Button
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
        # Grid setting
        self.red_plot_1.showGrid(x=True, y=True, alpha=0.3)
        self.red_plot_2.showGrid(x=True, y=True, alpha=0.3)
        self.red_plot_3.showGrid(x=True, y=True, alpha=0.3)
        self.red_plot_4.showGrid(x=True, y=True, alpha=0.3)
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


        self.show() # UI show command


    def time_display(self, display_variable):
        # display_variable[0] : time, display_variable[1].iloc[1]
        self.time_label.setText(f'<b>Time :<b/> {display_variable[0]} sec')
        self.time_label.setFont(QFont('Times new roman', 15))
        self.time_label.setAlignment(Qt.AlignCenter)
        self.power_label.setText(f'Power : {round(display_variable[1].iloc[1]["QPROREL"]*100, 2)}%')
        if round(display_variable[1].iloc[1]["QPROREL"]*100, 2) < 95:
            self.power_label.setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: red;')
        else:
            self.power_label.setStyleSheet('color : black;' 'background-color: light gray;')

    def Determine_train(self, train_untrain_reconstruction_error):
        if train_untrain_reconstruction_error[0] <= 0.00225299: # Trained Data
            self.trained_label.setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: green;')
            self.Untrained_label.setStyleSheet('color : black;' 'background-color: light gray;')
        else: # Untrianed Data
            self.Untrained_label.setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: red;')
            self.trained_label.setStyleSheet('color : black;' 'background-color: light gray;')

    def Determine_abnormal(self, abnormal_diagnosis):
        if abnormal_diagnosis == 0: # 정상상태
            self.normal_label.setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: green;')
            self.abnormal_label.setStyleSheet('color : black;' 'background-color: light gray;')
        else: # 비정상상태
            self.abnormal_label.setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: red;')
            self.normal_label.setStyleSheet('color : black;' 'background-color: light gray;')

    def Determine_procedure(self, abnormal_procedure_result):
        if abnormal_procedure_result == 0:
            self.num_procedure.setText('Normal')
            self.num_scnario.setText('Normal')
        elif abnormal_procedure_result == 1:
            self.num_procedure.setText('Ab21-01')
            self.num_scnario.setText('가압기 압력 채널 고장 "고"')
        elif abnormal_procedure_result == 2:
            self.num_procedure.setText('Ab21-02')
            self.num_scnario.setText('가압기 압력 채널 고장 "저"')
        elif abnormal_procedure_result == 3:
            self.num_procedure.setText('Ab20-04')
            self.num_scnario.setText('가압기 수위 채널 고장 "저"')
        elif abnormal_procedure_result == 4:
            self.num_procedure.setText('Ab15-07')
            self.num_scnario.setText('증기발생기 수위 채널 고장 "저"')
        elif abnormal_procedure_result == 5:
            self.num_procedure.setText('Ab15-08')
            self.num_scnario.setText('증기발생기 수위 채널 고장 "고"')
        elif abnormal_procedure_result == 6:
            self.num_procedure.setText('Ab63-04')
            self.num_scnario.setText('제어봉 낙하')
        elif abnormal_procedure_result == 7:
            self.num_procedure.setText('Ab63-02')
            self.num_scnario.setText('제어봉의 계속적인 삽입')
        elif abnormal_procedure_result == 8:
            self.num_procedure.setText('Ab21-12')
            # self.num_scnario.setText('가압기 PORV 열림')
            self.num_scnario.setText('Pressurizer PORV opening')
        elif abnormal_procedure_result == 9:
            self.num_procedure.setText('Ab19-02')
            self.num_scnario.setText('가압기 안전밸브 고장')
        elif abnormal_procedure_result == 10:
            self.num_procedure.setText('Ab21-11')
            self.num_scnario.setText('가압기 살수밸브 고장 "열림"')
        elif abnormal_procedure_result == 11:
            self.num_procedure.setText('Ab23-03')
            self.num_scnario.setText('1차기기 냉각수 계통으로 누설 "CVCS->CCW"')
        elif abnormal_procedure_result == 12:
            self.num_procedure.setText('Ab60-02')
            self.num_scnario.setText('재생열교환기 전단부위 파열')
        elif abnormal_procedure_result == 13:
            self.num_procedure.setText('Ab59-02')
            self.num_scnario.setText('충전수 유량조절밸브 후단 누설')
        elif abnormal_procedure_result == 14:
            self.num_procedure.setText('Ab23-01')
            self.num_scnario.setText('1차기기 냉각수 계통으로 누설 "RCS->CCW"')
        elif abnormal_procedure_result == 15:
            self.num_procedure.setText('Ab23-06')
            self.num_scnario.setText('증기발생기 전열관 누설')

    def verifit_result(self, verif_value):
        if verif_value[0] <= verif_value[1]: # 진단 성공
            self.success_label.setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: green;')
            self.failure_label.setStyleSheet('color : black;' 'background-color: light gray;')
        else: # 진단 실패
            self.failure_label.setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: red;')
            self.success_label.setStyleSheet('color : black;' 'background-color: light gray;')

    def procedure_satisfaction(self, symptom_db):
        # symptom_db[0] : classification result [0~15]
        # symptom_db[1] : check_db [2,2222] -> 현시점과 이전시점 비교를 위함.
        # symptom_db[1].iloc[0] : 이전 시점  # symptom_db[1].iloc[1] : 현재 시점
        if symptom_db[0] == 0: # 정상 상태
            self.symptom_name.setText('Diagnosis Result : Normal → Symptoms : 0')
            self.symptom1.setText('')
            self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")
            self.symptom2.setText('')
            self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")
            self.symptom3.setText('')
            self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")
            self.symptom4.setText('')
            self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")
            self.symptom5.setText('')
            self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")
            self.symptom6.setText('')
            self.symptom6.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

        elif symptom_db[0] == 1:
            self.symptom_name.setText('Diagnosis Result : Ab21-01 Pressurizer pressure channel failure "High" → Symptoms : 6')

            self.symptom1.setText("채널 고장으로 인한 가압기 '고' 압력 지시")
            if symptom_db[1].iloc[1]['PPRZN'] > symptom_db[1].iloc[1]['CPPRZH']:
                self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            else:
                self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

            self.symptom2.setText("가압기 살수밸브 '열림' 지시")
            if symptom_db[1].iloc[1]['BPRZSP'] > 0:
                self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            else:
                self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

            self.symptom3.setText("가압기 비례전열기 꺼짐")
            if symptom_db[1].iloc[1]['QPRZP'] == 0:
                self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            else:
                self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

            self.symptom4.setText("가압기 보조전열기 꺼짐")
            if symptom_db[1].iloc[1]['QPRZB'] == 0:
                self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            else:
                self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

            self.symptom5.setText("실제 가압기 '저' 압력 지시")
            if symptom_db[1].iloc[1]['PPRZ'] < symptom_db[1].iloc[1]['CPPRZL']:
                self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            else:
                self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

            self.symptom6.setText("가압기 PORV 차단밸브 닫힘")
            if symptom_db[1].iloc[1]['BHV6'] == 0:
                self.symptom6.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            else:
                self.symptom6.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

        elif symptom_db[0] == 2:
            self.symptom_name.setText('진단 : Ab21-02 가압기 압력 채널 고장 "저" → 증상 : 5')

            self.symptom1.setText("채널 고장으로 인한 가압기 '저' 압력 지시")
            if symptom_db[1].iloc[1]['PPRZN'] < symptom_db[1].iloc[1]['CPPRZL']:
                self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            else:
                self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

            self.symptom2.setText('가압기 저압력으로 인한 보조 전열기 켜짐 지시 및 경보 발생')
            if (symptom_db[1].iloc[1]['PPRZN'] < symptom_db[1].iloc[1]['CQPRZB']) and (symptom_db[1].iloc[1]['KBHON'] == 1):
                self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            else:
                self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

            self.symptom3.setText("실제 가압기 '고' 압력 지시")
            if symptom_db[1].iloc[1]['PPRZ'] > symptom_db[1].iloc[1]['CPPRZH']:
                self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            else:
                self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

            self.symptom4.setText('가압기 PORV 열림 지시 및 경보 발생')
            if symptom_db[1].iloc[1]['BPORV'] > 0:
                self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            else:
                self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

            self.symptom5.setText('실제 가압기 압력 감소로 가압기 PORV 닫힘') # 가압기 압력 감소에 대해 해결해야함.
            if symptom_db[1].iloc[1]['BPORV'] == 0 and (symptom_db[1].iloc[0]['PPRZ'] > symptom_db[1].iloc[1]['PPRZ']):
                self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            else:
                self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

        elif symptom_db[0] == 3:
            self.symptom_name.setText('진단 : Ab20-04 가압기 수위 채널 고장 "저" → 증상 : 5')

            self.symptom1.setText("채널 고장으로 인한 가압기 '저' 수위 지시")
            if symptom_db[1].iloc[1]['ZINST63'] < 17: # 나중에 다시 확인해야함.
                self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            # else:
            #     self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

            self.symptom2.setText('"LETDN HX OUTLET FLOW LOW" 경보 발생')
            if symptom_db[1].iloc[1]['UNRHXUT'] > symptom_db[1].iloc[1]['CULDHX']:
                self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            # else:
            #     self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

            self.symptom3.setText('"CHARGING LINE FLOW HI/LO" 경보 발생')
            if (symptom_db[1].iloc[1]['WCHGNO'] < symptom_db[1].iloc[1]['CWCHGL']) or (symptom_db[1].iloc[1]['WCHGNO'] > symptom_db[1].iloc[1]['CWCHGH']):
                self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            # else:
            #     self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

            self.symptom4.setText('충전 유량 증가')
            if symptom_db[1].iloc[0]['WCHGNO'] < symptom_db[1].iloc[1]['WCHGNO']:
                self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            # else:
            #     self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

            self.symptom5.setText('건전한 수위지시계의 수위 지시치 증가')
            if symptom_db[1].iloc[0]['ZPRZNO'] < symptom_db[1].iloc[1]['ZPRZNO']:
                self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            # else:
            #     self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

        elif symptom_db[0] == 4:
            self.symptom_name.setText('진단 : Ab15-07 증기발생기 수위 채널 고장 "저" → 증상 : ')
            self.symptom1.setText('증기발생기 수위 "저" 경보 발생')
            if symptom_db[1].iloc[1]['ZINST78']*0.01 < symptom_db[1].iloc[1]['CZSGW'] or symptom_db[1].iloc[1]['ZINST77']*0.01 < symptom_db[1].iloc[1]['CZSGW'] or symptom_db[1].iloc[1]['ZINST76']*0.01 < symptom_db[1].iloc[1]['CZSGW']:
                self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            else:
                self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")
            self.symptom2.setText('해당 SG MFCV 열림 방향으로 진행 및 해당 SG 실제 급수유량 증가')


        elif symptom_db[0] == 8:
            # self.symptom_name.setText('진단 : Ab21-12 가압기 PORV 열림 → 증상 : 5')
            self.symptom_name.setText('Diagnosis result : Ab21-12 Pressurizer PORV opening → Symptoms : 5')

            # self.symptom1.setText('가압기 PORV 열림 지시 및 경보 발생')
            self.symptom1.setText('Pressurizer PORV open indication and alarm')
            if symptom_db[1].iloc[1]['BPORV'] > 0:
                self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            else:
                self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

            # self.symptom2.setText('가압기 저압력으로 인한 보조 전열기 켜짐 지시 및 경보 발생')
            self.symptom2.setText('Aux. heater turn on instruction and alarm due to pressurizer low pressure')
            if (symptom_db[1].iloc[1]['PPRZN'] < symptom_db[1].iloc[1]['CQPRZB']) and (symptom_db[1].iloc[1]['KBHON'] == 1):
                self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            else:
                self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

            # self.symptom3.setText("가압기 '저' 압력 지시 및 경보 발생")
            self.symptom3.setText("pressurizer 'low' pressure indication and alarm")
            if symptom_db[1].iloc[1]['PPRZ'] < symptom_db[1].iloc[1]['CPPRZL'] :
                self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            else:
                self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

            # self.symptom4.setText("PRT 고온 지시 및 경보 발생")
            self.symptom4.setText("PRT high temperature indication and alarm")
            if symptom_db[1].iloc[1]['UPRT'] > symptom_db[1].iloc[1]['CUPRT'] :
                self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            else:
                self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

            # self.symptom5.setText("PRT 고압 지시 및 경보 발생")
            self.symptom5.setText("PRT high pressure indication and alarm")
            if (symptom_db[1].iloc[1]['PPRT'] - 0.98E5) > symptom_db[1].iloc[1]['CPPRT']:
                self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            else:
                self.symptom5.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

            self.symptom6.setText("Blank")
            self.symptom6.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

        elif symptom_db[0] == 10:
            self.symptom_name.setText("진단 : Ab21-11 가압기 살수밸브 고장 '열림'  → 증상 : 4")

            self.symptom1.setText("가압기 살수밸브 '열림' 지시 및 상태 표시등 점등")
            if symptom_db[1].iloc[1]['BPRZSP'] > 0:
                self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            else:
                self.symptom1.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

            self.symptom2.setText("가압기 보조전열기 켜짐 지시 및 경보 발생")
            if (symptom_db[1].iloc[1]['PPRZN'] < symptom_db[1].iloc[1]['CQPRZB']) and (symptom_db[1].iloc[1]['KBHON'] == 1):
                self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            else:
                self.symptom2.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

            self.symptom3.setText("가압기 '저' 압력 지시 및 경보 발생")
            if symptom_db[1].iloc[1]['PPRZ'] < symptom_db[1].iloc[1]['CPPRZL']:
                self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            else:
                self.symptom3.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

            self.symptom4.setText("가압기 수위 급격한 증가") # 급격한 증가에 대한 수정은 필요함 -> 추후 수정
            if symptom_db[1].iloc[0]['ZINST63'] < symptom_db[1].iloc[1]['ZINST63']:
                self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : red;""}")
            else:
                self.symptom4.setStyleSheet("QCheckBo" "x::indicator" "{""background-color : white;""}")

    def explain_result(self, shap_add_des):
        '''
        # shap_add_des['index'] : 변수 이름 / shap_add_des[0] : shap value
        # shap_add_des['describe'] : 변수에 대한 설명 / shap_add_des['probability'] : shap value를 확률로 환산한 값
        '''

        self.tableWidget.setRowCount(len(shap_add_des))
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setHorizontalHeaderLabels(["value_name", 'probability', 'describe', 'system'])

        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)

        [self.tableWidget.setItem(i, 0, QTableWidgetItem(f"{shap_add_des['index'][i]}")) for i in range(len(shap_add_des['index']))]
        [self.tableWidget.setItem(i, 1, QTableWidgetItem(f"{round(shap_add_des['probability'][i],2)}%")) for i in range(len(shap_add_des['probability']))]
        [self.tableWidget.setItem(i, 2, QTableWidgetItem(f"{shap_add_des['describe'][i]}")) for i in range(len(shap_add_des['describe']))]
        [self.tableWidget.setItem(i, 3, QTableWidgetItem(f"{shap_add_des['system'][i]}")) for i in range(len(shap_add_des['system']))]

        delegate = AlignDelegate(self.tableWidget)
        self.tableWidget.setItemDelegate(delegate)

    def show_table(self):
        self.worker.shap.connect(self.explain_result)
        # 클릭시 Thread를 통해 신호를 전달하기 때문에 버퍼링이 발생함. 2초 정도? 이 부분은 나중에 생각해서 초기에 불러올지 고민해봐야할듯.
        self.tableWidget.show()

    def plotting(self, symptom_db):
        #  symptom_db[0] : liner : appended time (axis-x) / symptom_db[1].iloc[1] : check_db (:line,2222)[1]
        # -- scatter --
        # time = []
        # value1, value2, value3 = [], [], []
        # time.append(symptom_db[0])
        # value1.append(round(symptom_db[1].iloc[1]['ZVCT'],2))
        # value2.append(round(symptom_db[1].iloc[1]['BPORV'],2))
        # value3.append(round(symptom_db[1].iloc[1]['UPRZ'],2))

        # self.plotting_1 = self.plot_1.plot(pen=None, symbol='o', symbolBrush='w', symbolPen='w', symbolSize=5)
        # self.plotting_2 = self.plot_2.plot(pen=None, symbol='o', symbolBrush='w', symbolPen='w', symbolSize=5)
        # self.plotting_3 = self.plot_3.plot(pen=None, symbol='o', symbolBrush='w', symbolPen='w', symbolSize=5)

        # -- Line plotting --
        # self.plotting_1 = self.plot_1.plot(pen='w')
        # self.plotting_2 = self.plot_2.plot(pen='w')
        # self.plotting_3 = self.plot_3.plot(pen='w')
        # self.plotting_4 = self.plot_4.plot(pen='w')
        self.plot_1.showGrid(x=True, y=True, alpha=0.3)
        self.plot_2.showGrid(x=True, y=True, alpha=0.3)
        self.plot_3.showGrid(x=True, y=True, alpha=0.3)
        self.plot_4.showGrid(x=True, y=True, alpha=0.3)

        self.plotting_1 = self.plot_1.plot(pen=pyqtgraph.mkPen('k',width=3))
        self.plotting_2 = self.plot_2.plot(pen=pyqtgraph.mkPen('k',width=3))
        self.plotting_3 = self.plot_3.plot(pen=pyqtgraph.mkPen('k',width=3))
        self.plotting_4 = self.plot_4.plot(pen=pyqtgraph.mkPen('k',width=3))

        self.plotting_1.setData(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])['BPORV'])
        self.plot_1.setTitle('PORV open state')
        self.plotting_2.setData(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])['PPRZN'])
        self.plot_2.setTitle('Pressurizer pressure')
        self.plotting_3.setData(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])['UPRT'])
        self.plot_3.setTitle('PRT temperature')
        self.plotting_4.setData(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])['PPRT'])
        self.plot_4.setTitle('PRT pressure')

        # red_range = display_db[display_db['probability'] >= 10] # 10% 이상의 확률을 가진 변수
        #
        # print(bool(red_range["describe"].iloc[3]))
        # try :
        #     self.plotting_1.setData(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[red_range['index'].iloc[0]])
        #     if red_range["describe"].iloc[0] == None:
        #         self.plot_1.setTitle(self)
        #     else:
        #         self.plot_1.setTitle(f'{red_range["describe"].iloc[0]}')
        #     # self.plot_1.clear()
        # except:
        #     print('plot1 fail')
        # try:
        #     self.plotting_2.setData(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[red_range['index'].iloc[1]])
        #     if red_range["describe"].iloc[1] == None:
        #         self.plot_2.setTitle(self)
        #     else:
        #         self.plot_2.setTitle(f'{red_range["describe"].iloc[1]}')
        #     # self.plot_2.clear()
        # except:
        #     print('plot2 fail')
        # try:
        #     self.plotting_3.setData(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[red_range['index'].iloc[2]])
        #     if red_range["describe"].iloc[2] == None:
        #         self.plot_3.setTitle(self)
        #     else:
        #         self.plot_3.setTitle(f'{red_range["describe"].iloc[2]}')
        #     # self.plot_3.clear()
        # except:
        #     print('plot3 fail')
        # try:
        #     self.plotting_4.setData(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[red_range['index'].iloc[3]])
        #     if red_range["describe"].iloc[3] == None:
        #         self.plot_4.setTitle(self)
        #     else:
        #         self.plot_4.setTitle(f'{red_range["describe"].iloc[3]}')
        #     # self.plot_4.clear()
        # except:
        #     print('plot4 fail')

    def display_explain(self, display_db, symptom_db, normal_db):
        '''
        # display_db['index'] : 변수 이름 / display_db[0] : shap value
        # display_db['describe'] : 변수에 대한 설명 / display_db['probability'] : shap value를 확률로 환산한 값
        #  symptom_db[0] : liner : appended time (axis-x) / symptom_db[1].iloc[1] : check_db (:line,2222)[1]
        '''
        red_range = display_db[display_db['probability'] >=10]
        orange_range = display_db[[display_db['probability'].iloc[i]<10 and display_db['probability'].iloc[i]>1 for i in range(len(display_db['probability']))]]
        convert_red = {0: self.red1, 1: self.red2, 2: self.red3, 3: self.red4}
        convert_orange = {0: self.orange1, 1: self.orange2, 2: self.orange3, 3: self.orange4, 4: self.orange5, 5: self.orange6, 6: self.orange7, 7: self.orange8, 8: self.orange9, 9: self.orange10, 10: self.orange11, 11: self.orange12}
        if 4-len(red_range) == 0:
            red_del = []
        elif 4-len(red_range) == 1:
            red_del = [3]
        elif 4-len(red_range) == 2:
            red_del = [2,3]
        elif 4-len(red_range) == 3:
            red_del = [1,2,3]
        elif 4-len(red_range) == 4:
            red_del = [0,1,2,3]

        if 12-len(orange_range) == 0:
            orange_del = []
        elif 12-len(orange_range) == 1:
            orange_del = [11]
        elif 12-len(orange_range) == 2:
            orange_del = [10,11]
        elif 12-len(orange_range) == 3:
            orange_del = [9,10,11]
        elif 12-len(orange_range) == 4:
            orange_del = [8,9,10,11]
        elif 12-len(orange_range) == 5:
            orange_del = [7,8,9,10,11]
        elif 12-len(orange_range) == 6:
            orange_del = [6,7,8,9,10,11]
        elif 12-len(orange_range) == 7:
            orange_del = [5,6,7,8,9,10,11]
        elif 12-len(orange_range) == 8:
            orange_del = [4,5,6,7,8,9,10,11]
        elif 12-len(orange_range) == 9:
            orange_del = [3,4,5,6,7,8,9,10,11]
        elif 12-len(orange_range) == 10:
            orange_del = [2,3,4,5,6,7,8,9,10,11]
        elif 12-len(orange_range) == 11:
            orange_del = [1,2,3,4,5,6,7,8,9,10,11]
        elif 12-len(orange_range) == 12:
            orange_del = [0,1,2,3,4,5,6,7,8,9,10,11]

        [convert_red[i].setText(f'{red_range["describe"].iloc[i]} \n[{round(red_range["probability"].iloc[i],2)}%]') for i in range(len(red_range))]
        [convert_red[i].setText('None\nParameter') for i in red_del]
        [convert_red[i].setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: blue;') for i in range(len(red_range))]
        [convert_red[i].setStyleSheet('color : black;' 'background-color: light gray;') for i in red_del]

        [convert_orange[i].setText(f'{orange_range["describe"].iloc[i]} \n[{round(orange_range["probability"].iloc[i],2)}%]') for i in range(len(orange_range))]
        [convert_orange[i].setText('None\nParameter') for i in orange_del]
        # [convert_orange[i].setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: orange;') for i in range(len(orange_range))]
        # [convert_orange[i].setStyleSheet('color : black;' 'background-color: light gray;') for i in orange_del]

        # 각 Button에 호환되는 Plotting 데이터 구축
        # Red1 Button
        if self.red1.text().split()[0] != 'None':
            self.red_plot_1.clear()
            self.red_plot_1.setTitle(red_range['describe'].iloc[0])
            self.red_plot_1.addLegend(offset=(-30,20))
            self.red_plot_1.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[red_range['index'].iloc[0]], pen=pyqtgraph.mkPen('b', width=3), name = 'Real Data')
            self.red_plot_1.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[red_range['index'].iloc[0]], pen=pyqtgraph.mkPen('k', width=3), name = 'Normal Data')

        # Red2 Button
        if self.red2.text().split()[0] != 'None':
            self.red_plot_2.clear()
            self.red_plot_2.setTitle(red_range['describe'].iloc[1])
            self.red_plot_2.addLegend(offset=(-30, 20))
            self.red_plot_2.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[red_range['index'].iloc[1]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.red_plot_2.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[red_range['index'].iloc[1]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Red3 Button
        if self.red3.text().split()[0] != 'None':
            self.red_plot_3.clear()
            self.red_plot_3.setTitle(red_range['describe'].iloc[2])
            self.red_plot_3.addLegend(offset=(-30, 20))
            self.red_plot_3.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[red_range['index'].iloc[2]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.red_plot_3.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[red_range['index'].iloc[2]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Red4 Button
        if self.red4.text().split()[0] != 'None':
            self.red_plot_4.clear()
            self.red_plot_4.setTitle(red_range['describe'].iloc[3])
            self.red_plot_4.addLegend(offset=(-30, 20))
            self.red_plot_4.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[red_range['index'].iloc[3]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.red_plot_4.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[red_range['index'].iloc[3]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange1 Button
        if self.orange1.text().split()[0] != 'None':
            self.orange_plot_1.clear()
            self.orange_plot_1.setTitle(orange_range['describe'].iloc[0])
            self.orange_plot_1.addLegend(offset=(-30, 20))
            self.orange_plot_1.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[0]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.orange_plot_1.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['index'].iloc[0]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange2 Button
        if self.orange2.text().split()[0] != 'None':
            self.orange_plot_2.clear()
            self.orange_plot_2.setTitle(orange_range['describe'].iloc[1])
            self.orange_plot_2.addLegend(offset=(-30, 20))
            self.orange_plot_2.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[1]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.orange_plot_2.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['index'].iloc[1]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange3 Button
        if self.orange3.text().split()[0] != 'None':
            self.orange_plot_3.clear()
            self.orange_plot_3.setTitle(orange_range['describe'].iloc[2])
            self.orange_plot_3.addLegend(offset=(-30, 20))
            self.orange_plot_3.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[2]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.orange_plot_3.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['index'].iloc[2]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange4 Button
        if self.orange4.text().split()[0] != 'None':
            self.orange_plot_4.clear()
            self.orange_plot_4.setTitle(orange_range['describe'].iloc[3])
            self.orange_plot_4.addLegend(offset=(-30, 20))
            self.orange_plot_4.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[3]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.orange_plot_4.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['index'].iloc[3]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange5 Button
        if self.orange5.text().split()[0] != 'None':
            self.orange_plot_5.clear()
            self.orange_plot_5.setTitle(orange_range['describe'].iloc[4])
            self.orange_plot_5.addLegend(offset=(-30, 20))
            self.orange_plot_5.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[4]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.orange_plot_5.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['index'].iloc[4]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange6 Button
        if self.orange6.text().split()[0] != 'None':
            self.orange_plot_6.clear()
            self.orange_plot_6.setTitle(orange_range['describe'].iloc[5])
            self.orange_plot_6.addLegend(offset=(-30, 20))
            self.orange_plot_6.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[5]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.orange_plot_6.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['index'].iloc[5]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange7 Button
        if self.orange7.text().split()[0] != 'None':
            self.orange_plot_7.clear()
            self.orange_plot_7.setTitle(orange_range['describe'].iloc[6])
            self.orange_plot_7.addLegend(offset=(-30, 20))
            self.orange_plot_7.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[6]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.orange_plot_7.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['index'].iloc[6]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange8 Button
        if self.orange8.text().split()[0] != 'None':
            self.orange_plot_8.clear()
            self.orange_plot_8.setTitle(orange_range['describe'].iloc[7])
            self.orange_plot_8.addLegend(offset=(-30, 20))
            self.orange_plot_8.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[7]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.orange_plot_8.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['index'].iloc[7]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange9 Button
        if self.orange9.text().split()[0] != 'None':
            self.orange_plot_9.clear()
            self.orange_plot_9.setTitle(orange_range['describe'].iloc[8])
            self.orange_plot_9.addLegend(offset=(-30, 20))
            self.orange_plot_9.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[8]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.orange_plot_9.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['index'].iloc[8]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange10 Button
        if self.orange10.text().split()[0] != 'None':
            self.orange_plot_10.clear()
            self.orange_plot_10.setTitle(orange_range['describe'].iloc[9])
            self.orange_plot_10.addLegend(offset=(-30, 20))
            self.orange_plot_10.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[9]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.orange_plot_10.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['index'].iloc[9]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange11 Button
        if self.orange11.text().split()[0] != 'None':
            self.orange_plot_11.clear()
            self.orange_plot_11.setTitle(orange_range['describe'].iloc[10])
            self.orange_plot_11.addLegend(offset=(-30, 20))
            self.orange_plot_11.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[10]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.orange_plot_11.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['index'].iloc[10]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        # Orange12 Button
        if self.orange12.text().split()[0] != 'None':
            self.orange_plot_12.clear()
            self.orange_plot_12.setTitle(orange_range['describe'].iloc[11])
            self.orange_plot_12.addLegend(offset=(-30, 20))
            self.orange_plot_12.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[11]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.orange_plot_12.plot(x=symptom_db[0], y=pd.DataFrame(normal_db)[orange_range['index'].iloc[11]], pen=pyqtgraph.mkPen('k', width=3), name='Normal Data')

        [convert_red[i].setCheckable(True) for i in range(4)]
        [convert_orange[i].setCheckable(True) for i in range(12)]

    def red1_plot(self):
        if self.red1.isChecked():
            if self.red1.text().split()[0] != 'None':
                self.red_plot_1.show()
                self.red1.setCheckable(False)

    def red2_plot(self):
        if self.red2.isChecked():
            if self.red2.text().split()[0] != 'None':
                self.red_plot_2.show()
                self.red2.setCheckable(False)

    def red3_plot(self):
        if self.red3.isChecked():
            if self.red3.text().split()[0] != 'None':
                self.red_plot_3.show()
                self.red3.setCheckable(False)

    def red4_plot(self):
        if self.red4.isChecked():
            if self.red4.text().split()[0] != 'None':
                self.red_plot_4.show()
                self.red4.setCheckable(False)

    def orange1_plot(self):
        if self.orange1.isChecked():
            if self.orange1.text().split()[0] != 'None':
                self.orange_plot_1.show()
                self.orange1.setCheckable(False)

    def orange2_plot(self):
        if self.orange2.isChecked():
            if self.orange2.text().split()[0] != 'None':
                self.orange_plot_2.show()
                self.orange2.setCheckable(False)

    def orange3_plot(self):
        if self.orange3.isChecked():
            if self.orange3.text().split()[0] != 'None':
                self.orange_plot_3.show()
                self.orange3.setCheckable(False)

    def orange4_plot(self):
        if self.orange4.isChecked():
            if self.orange4.text().split()[0] != 'None':
                self.orange_plot_4.show()
                self.orange4.setCheckable(False)

    def orange5_plot(self):
        if self.orange5.isChecked():
            if self.orange5.text().split()[0] != 'None':
                self.orange_plot_5.show()
                self.orange5.setCheckable(False)

    def orange6_plot(self):
        if self.orange6.isChecked():
            if self.orange6.text().split()[0] != 'None':
                self.orange_plot_6.show()
                self.orange6.setCheckable(False)

    def orange7_plot(self):
        if self.orange7.isChecked():
            if self.orange7.text().split()[0] != 'None':
                self.orange_plot_7.show()
                self.orange7.setCheckable(False)

    def orange8_plot(self):
        if self.orange8.isChecked():
            if self.orange8.text().split()[0] != 'None':
                self.orange_plot_8.show()
                self.orange8.setCheckable(False)

    def orange9_plot(self):
        if self.orange9.isChecked():
            if self.orange9.text().split()[0] != 'None':
                self.orange_plot_9.show()
                self.orange9.setCheckable(False)

    def orange10_plot(self):
        if self.orange10.isChecked():
            if self.orange10.text().split()[0] != 'None':
                self.orange_plot_10.show()
                self.orange10.setCheckable(False)

    def orange11_plot(self):
        if self.orange11.isChecked():
            if self.orange11.text().split()[0] != 'None':
                self.orange_plot_11.show()
                self.orange11.setCheckable(False)

    def orange12_plot(self):
        if self.orange12.isChecked():
            if self.orange12.text().split()[0] != 'None':
                self.orange_plot_12.show()
                self.orange12.setCheckable(False)

    def show_another_result(self):
        self.other = another_result_explain()
        self.worker.another_shap_table.connect(self.other.show_another_result_table)
        self.worker.another_shap.connect(self.other.show_shap)
        self.other.show()


class another_result_explain(QWidget):
    def __init__(self):
        super().__init__()
        # 서브 인터페이스 초기 설정
        self.setWindowTitle('Another Result Explanation')
        self.setGeometry(300, 300, 800, 500)
        self.selected_para = pd.read_csv('./DataBase/Final_parameter_200825.csv')

        # 레이아웃 구성
        combo_layout = QVBoxLayout()
        self.title_label = QLabel("<b>선택되지 않은 시나리오에 대한 결과 해석<b/>")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.blank = QLabel(self) # Enter를 위한 라벨

        self.show_table = QPushButton("Show Table")

        self.cb = QComboBox(self)
        self.cb.addItem('Normal')
        self.cb.addItem('Ab21-01: Pressurizer pressure channel failure (High)')
        self.cb.addItem('Ab21-02: Pressurizer pressure channel failure (Low)')
        self.cb.addItem('Ab20-04: Pressurizer level channel failure (Low)')
        self.cb.addItem('Ab15-07: Steam generator level channel failure (High)')
        self.cb.addItem('Ab15-08: Steam generator level channel failure (Low)')
        self.cb.addItem('Ab63-04: Control rod fall')
        self.cb.addItem('Ab63-02: Continuous insertion of control rod')
        self.cb.addItem('Ab21-12: Pressurizer PORV opening')
        self.cb.addItem('Ab19-02: Pressurizer safety valve failure')
        self.cb.addItem('Ab21-11: Pressurizer spray valve failed opening')
        self.cb.addItem('Ab23-03: Leakage from CVCS to RCS')
        self.cb.addItem('Ab60-02: Rupture of the front end of the regenerative heat exchanger')
        self.cb.addItem('Ab59-02: Leakage at the rear end of the charging flow control valve')
        self.cb.addItem('Ab23-01: Leakage from CVCS to CCW')
        self.cb.addItem('Ab23-06: Steam generator u-tube leakage')

        # Explanation Alarm 구현
        cb_red_alarm = QGroupBox('Main basis for diagnosis')
        cb_red_alarm_layout = QGridLayout()
        cb_orange_alarm = QGroupBox('Sub basis for diagnosis')
        cb_orange_alarm_layout = QGridLayout()

        # Display Button 생성
        self.cb_red1 = QPushButton(self)
        self.cb_red2 = QPushButton(self)
        self.cb_red3 = QPushButton(self)
        self.cb_red4 = QPushButton(self)
        self.cb_orange1 = QPushButton(self)
        self.cb_orange2 = QPushButton(self)
        self.cb_orange3 = QPushButton(self)
        self.cb_orange4 = QPushButton(self)
        self.cb_orange5 = QPushButton(self)
        self.cb_orange6 = QPushButton(self)
        self.cb_orange7 = QPushButton(self)
        self.cb_orange8 = QPushButton(self)
        self.cb_orange9 = QPushButton(self)
        self.cb_orange10 = QPushButton(self)
        self.cb_orange11 = QPushButton(self)
        self.cb_orange12 = QPushButton(self)
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
        # combo_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        combo_layout.addWidget(cb_red_alarm)
        combo_layout.addWidget(cb_orange_alarm)
        combo_layout.addWidget(self.blank)
        combo_layout.addWidget(self.show_table)
        combo_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.setLayout(combo_layout)

        self.combo_tableWidget = QTableWidget(0, 0)
        self.combo_tableWidget.setFixedHeight(500)
        self.combo_tableWidget.setFixedWidth(800)

        # self.combo_tableWidget = QTableWidget(0, 0)

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
        # Grid setting
        self.cb_red_plot_1.showGrid(x=True, y=True, alpha=0.3)
        self.cb_red_plot_2.showGrid(x=True, y=True, alpha=0.3)
        self.cb_red_plot_3.showGrid(x=True, y=True, alpha=0.3)
        self.cb_red_plot_4.showGrid(x=True, y=True, alpha=0.3)
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
        ################################################################################################################
        self.show() # Sub UI show command

    def show_shap(self, all_shap, symptom_db, compare_data):
        # all_shap : 전체 시나리오에 해당하는 shap_value를 가지고 있음.
        # symptom_db[0] : liner : appended time (axis-x) / symptom_db[1].iloc[1] : check_db (:line,2222)[1]

        if self.cb.currentText() == 'Normal':
            step1 = pd.DataFrame(all_shap[0], columns=self.selected_para['0'].tolist())
            compared_db = compare_data[self.cb.currentText()]
        elif self.cb.currentText() == 'Ab21-01: Pressurizer pressure channel failure (High)':
            step1 = pd.DataFrame(all_shap[1], columns=self.selected_para['0'].tolist())
            compared_db = compare_data[self.cb.currentText()[:7]]
        elif self.cb.currentText() == 'Ab21-02: Pressurizer pressure channel failure (Low)':
            step1 = pd.DataFrame(all_shap[2], columns=self.selected_para['0'].tolist())
            compared_db = compare_data[self.cb.currentText()[:7]]
        elif self.cb.currentText() == 'Ab20-04: Pressurizer level channel failure (Low)':
            step1 = pd.DataFrame(all_shap[3], columns=self.selected_para['0'].tolist())
            compared_db = compare_data[self.cb.currentText()[:7]]
        elif self.cb.currentText() == 'Ab15-07: Steam generator level channel failure (High)':
            step1 = pd.DataFrame(all_shap[4], columns=self.selected_para['0'].tolist())
            compared_db = compare_data[self.cb.currentText()[:7]]
        elif self.cb.currentText() == 'Ab15-08: Steam generator level channel failure (Low)':
            step1 = pd.DataFrame(all_shap[5], columns=self.selected_para['0'].tolist())
            compared_db = compare_data[self.cb.currentText()[:7]]
        elif self.cb.currentText() == 'Ab63-04: Control rod fall':
            step1 = pd.DataFrame(all_shap[6], columns=self.selected_para['0'].tolist())
            compared_db = compare_data[self.cb.currentText()[:7]]
        elif self.cb.currentText() == 'Ab63-02: Continuous insertion of control rod':
            step1 = pd.DataFrame(all_shap[7], columns=self.selected_para['0'].tolist())
            compared_db = compare_data[self.cb.currentText()[:7]]
        elif self.cb.currentText() == 'Ab21-12: Pressurizer PORV opening':
            step1 = pd.DataFrame(all_shap[8], columns=self.selected_para['0'].tolist())
            compared_db = compare_data[self.cb.currentText()[:7]]
        elif self.cb.currentText() == 'Ab19-02: Pressurizer safety valve failure':
            step1 = pd.DataFrame(all_shap[9], columns=self.selected_para['0'].tolist())
            compared_db = compare_data[self.cb.currentText()[:7]]
        elif self.cb.currentText() == 'Ab21-11: Pressurizer spray valve failed opening':
            step1 = pd.DataFrame(all_shap[10], columns=self.selected_para['0'].tolist())
            compared_db = compare_data[self.cb.currentText()[:7]]
        elif self.cb.currentText() == 'Ab23-03: Leakage from CVCS to RCS':
            step1 = pd.DataFrame(all_shap[11], columns=self.selected_para['0'].tolist())
            compared_db = compare_data[self.cb.currentText()[:7]]
        elif self.cb.currentText() == 'Ab60-02: Rupture of the front end of the regenerative heat exchanger':
            step1 = pd.DataFrame(all_shap[12], columns=self.selected_para['0'].tolist())
            compared_db = compare_data[self.cb.currentText()[:7]]
        elif self.cb.currentText() == 'Ab59-02: Leakage at the rear end of the charging flow control valve':
            step1 = pd.DataFrame(all_shap[13], columns=self.selected_para['0'].tolist())
            compared_db = compare_data[self.cb.currentText()[:7]]
        elif self.cb.currentText() == 'Ab23-01: Leakage from CVCS to CCW':
            step1 = pd.DataFrame(all_shap[14], columns=self.selected_para['0'].tolist())
            compared_db = compare_data[self.cb.currentText()[:7]]
        elif self.cb.currentText() == 'Ab23-06: Steam generator u-tube leakage':
            step1 = pd.DataFrame(all_shap[15], columns=self.selected_para['0'].tolist())
            compared_db = compare_data[self.cb.currentText()[:7]]

        step2 = step1.sort_values(by=0, ascending=True, axis=1)
        step3 = step2[step2.iloc[:] < 0].dropna(axis=1).T
        self.step4 = step3.reset_index()
        col = self.step4['index']
        var = [self.selected_para['0'][self.selected_para['0'] == col_].index for col_ in col]
        val_col = [self.selected_para['1'][var_].iloc[0] for var_ in var]
        proba = [(self.step4[0][val_num] / sum(self.step4[0])) * 100 for val_num in range(len(self.step4[0]))]
        val_system = [self.selected_para['2'][var_].iloc[0] for var_ in var]
        self.step4['describe'] = val_col
        self.step4['probability'] = proba
        self.step4['system'] = val_system

        red_range = self.step4[self.step4['probability'] >= 10]
        orange_range = self.step4[
            [self.step4['probability'].iloc[i] < 10 and self.step4['probability'].iloc[i] > 1 for i in
             range(len(self.step4['probability']))]]
        convert_red = {0: self.cb_red1, 1: self.cb_red2, 2: self.cb_red3, 3: self.cb_red4}
        convert_orange = {0: self.cb_orange1, 1: self.cb_orange2, 2: self.cb_orange3, 3: self.cb_orange4, 4: self.cb_orange5,
                          5: self.cb_orange6, 6: self.cb_orange7, 7: self.cb_orange8, 8: self.cb_orange9, 9: self.cb_orange10,
                          10: self.cb_orange11, 11: self.cb_orange12}
        if 4 - len(red_range) == 0:
            red_del = []
        elif 4 - len(red_range) == 1:
            red_del = [3]
        elif 4 - len(red_range) == 2:
            red_del = [2, 3]
        elif 4 - len(red_range) == 3:
            red_del = [1, 2, 3]
        elif 4 - len(red_range) == 4:
            red_del = [0, 1, 2, 3]

        if 12 - len(orange_range) == 0:
            orange_del = []
        elif 12 - len(orange_range) == 1:
            orange_del = [11]
        elif 12 - len(orange_range) == 2:
            orange_del = [10, 11]
        elif 12 - len(orange_range) == 3:
            orange_del = [9, 10, 11]
        elif 12 - len(orange_range) == 4:
            orange_del = [8, 9, 10, 11]
        elif 12 - len(orange_range) == 5:
            orange_del = [7, 8, 9, 10, 11]
        elif 12 - len(orange_range) == 6:
            orange_del = [6, 7, 8, 9, 10, 11]
        elif 12 - len(orange_range) == 7:
            orange_del = [5, 6, 7, 8, 9, 10, 11]
        elif 12 - len(orange_range) == 8:
            orange_del = [4, 5, 6, 7, 8, 9, 10, 11]
        elif 12 - len(orange_range) == 9:
            orange_del = [3, 4, 5, 6, 7, 8, 9, 10, 11]
        elif 12 - len(orange_range) == 10:
            orange_del = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        elif 12 - len(orange_range) == 11:
            orange_del = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        elif 12 - len(orange_range) == 12:
            orange_del = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        [convert_red[i].setText(f'{red_range["describe"].iloc[i]} \n[{round(red_range["probability"].iloc[i], 2)}%]') for i in range(len(red_range))]
        [convert_red[i].setText('None\nParameter') for i in red_del]
        [convert_red[i].setStyleSheet('color : white;' 'font-weight: bold;' 'background-color: blue;') for i in range(len(red_range))]
        [convert_red[i].setStyleSheet('color : black;' 'background-color: light gray;') for i in red_del]

        [convert_orange[i].setText(f'{orange_range["describe"].iloc[i]} \n[{round(orange_range["probability"].iloc[i], 2)}%]') for i in range(len(orange_range))]
        [convert_orange[i].setText('None\nParameter') for i in orange_del]

#####################################################################################################################################
        # 각 Button에 호환되는 Plotting 데이터 구축
        # Red1 Button
        if self.cb_red1.text().split()[0] != 'None':
            self.cb_red_plot_1.clear()
            self.cb_red_plot_1.setTitle(red_range['describe'].iloc[0])
            self.cb_red_plot_1.addLegend(offset=(-30,20))
            self.cb_red_plot_1.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[red_range['index'].iloc[0]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.cb_red_plot_1.plot(x=symptom_db[0], y=pd.DataFrame(compared_db)[red_range['index'].iloc[0]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])


        # Red2 Button
        if self.cb_red2.text().split()[0] != 'None':
            self.cb_red_plot_2.clear()
            self.cb_red_plot_2.setTitle(red_range['describe'].iloc[1])
            self.cb_red_plot_2.addLegend(offset=(-30, 20))
            self.cb_red_plot_2.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[red_range['index'].iloc[1]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.cb_red_plot_2.plot(x=symptom_db[0], y=pd.DataFrame(compared_db)[red_range['index'].iloc[1]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Red3 Button
        if self.cb_red3.text().split()[0] != 'None':
            self.cb_red_plot_3.clear()
            self.cb_red_plot_3.setTitle(red_range['describe'].iloc[2])
            self.cb_red_plot_3.addLegend(offset=(-30, 20))
            self.cb_red_plot_3.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[red_range['index'].iloc[2]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.cb_red_plot_3.plot(x=symptom_db[0], y=pd.DataFrame(compared_db)[red_range['index'].iloc[2]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Red4 Button
        if self.cb_red4.text().split()[0] != 'None':
            self.cb_red_plot_4.clear()
            self.cb_red_plot_4.setTitle(red_range['describe'].iloc[3])
            self.cb_red_plot_4.addLegend(offset=(-30, 20))
            self.cb_red_plot_4.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[red_range['index'].iloc[3]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.cb_red_plot_4.plot(x=symptom_db[0], y=pd.DataFrame(compared_db)[red_range['index'].iloc[3]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange1 Button
        if self.cb_orange1.text().split()[0] != 'None':
            self.cb_orange_plot_1.clear()
            self.cb_orange_plot_1.setTitle(orange_range['describe'].iloc[0])
            self.cb_orange_plot_1.addLegend(offset=(-30, 20))
            self.cb_orange_plot_1.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[0]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.cb_orange_plot_1.plot(x=symptom_db[0], y=pd.DataFrame(compared_db)[orange_range['index'].iloc[0]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange2 Button
        if self.cb_orange2.text().split()[0] != 'None':
            self.cb_orange_plot_2.clear()
            self.cb_orange_plot_2.setTitle(orange_range['describe'].iloc[1])
            self.cb_orange_plot_2.addLegend(offset=(-30, 20))
            self.cb_orange_plot_2.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[1]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.cb_orange_plot_2.plot(x=symptom_db[0], y=pd.DataFrame(compared_db)[orange_range['index'].iloc[1]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange3 Button
        if self.cb_orange3.text().split()[0] != 'None':
            self.cb_orange_plot_3.clear()
            self.cb_orange_plot_3.setTitle(orange_range['describe'].iloc[2])
            self.cb_orange_plot_3.addLegend(offset=(-30, 20))
            self.cb_orange_plot_3.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[2]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.cb_orange_plot_3.plot(x=symptom_db[0], y=pd.DataFrame(compared_db)[orange_range['index'].iloc[2]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange4 Button
        if self.cb_orange4.text().split()[0] != 'None':
            self.cb_orange_plot_4.clear()
            self.cb_orange_plot_4.setTitle(orange_range['describe'].iloc[3])
            self.cb_orange_plot_4.addLegend(offset=(-30, 20))
            self.cb_orange_plot_4.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[3]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.cb_orange_plot_4.plot(x=symptom_db[0], y=pd.DataFrame(compared_db)[orange_range['index'].iloc[3]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange5 Button
        if self.cb_orange5.text().split()[0] != 'None':
            self.cb_orange_plot_5.clear()
            self.cb_orange_plot_5.setTitle(orange_range['describe'].iloc[4])
            self.cb_orange_plot_5.addLegend(offset=(-30, 20))
            self.cb_orange_plot_5.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[4]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.cb_orange_plot_5.plot(x=symptom_db[0], y=pd.DataFrame(compared_db)[orange_range['index'].iloc[4]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange6 Button
        if self.cb_orange6.text().split()[0] != 'None':
            self.cb_orange_plot_6.clear()
            self.cb_orange_plot_6.setTitle(orange_range['describe'].iloc[5])
            self.cb_orange_plot_6.addLegend(offset=(-30, 20))
            self.cb_orange_plot_6.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[5]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.cb_orange_plot_6.plot(x=symptom_db[0], y=pd.DataFrame(compared_db)[orange_range['index'].iloc[5]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange7 Button
        if self.cb_orange7.text().split()[0] != 'None':
            self.cb_orange_plot_7.clear()
            self.cb_orange_plot_7.setTitle(orange_range['describe'].iloc[6])
            self.cb_orange_plot_7.addLegend(offset=(-30, 20))
            self.cb_orange_plot_7.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[6]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.cb_orange_plot_7.plot(x=symptom_db[0], y=pd.DataFrame(compared_db)[orange_range['index'].iloc[6]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange8 Button
        if self.cb_orange8.text().split()[0] != 'None':
            self.cb_orange_plot_8.clear()
            self.cb_orange_plot_8.setTitle(orange_range['describe'].iloc[7])
            self.cb_orange_plot_8.addLegend(offset=(-30, 20))
            self.cb_orange_plot_8.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[7]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.cb_orange_plot_8.plot(x=symptom_db[0], y=pd.DataFrame(compared_db)[orange_range['index'].iloc[7]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange9 Button
        if self.cb_orange9.text().split()[0] != 'None':
            self.cb_orange_plot_9.clear()
            self.cb_orange_plot_9.setTitle(orange_range['describe'].iloc[8])
            self.cb_orange_plot_9.addLegend(offset=(-30, 20))
            self.cb_orange_plot_9.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[8]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.cb_orange_plot_9.plot(x=symptom_db[0], y=pd.DataFrame(compared_db)[orange_range['index'].iloc[8]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange10 Button
        if self.cb_orange10.text().split()[0] != 'None':
            self.cb_orange_plot_10.clear()
            self.cb_orange_plot_10.setTitle(orange_range['describe'].iloc[9])
            self.cb_orange_plot_10.addLegend(offset=(-30, 20))
            self.cb_orange_plot_10.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[9]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.cb_orange_plot_10.plot(x=symptom_db[0], y=pd.DataFrame(compared_db)[orange_range['index'].iloc[9]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange11 Button
        if self.cb_orange11.text().split()[0] != 'None':
            self.cb_orange_plot_11.clear()
            self.cb_orange_plot_11.setTitle(orange_range['describe'].iloc[10])
            self.cb_orange_plot_11.addLegend(offset=(-30, 20))
            self.cb_orange_plot_11.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[10]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.cb_orange_plot_11.plot(x=symptom_db[0], y=pd.DataFrame(compared_db)[orange_range['index'].iloc[10]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        # Orange12 Button
        if self.cb_orange12.text().split()[0] != 'None':
            self.cb_orange_plot_12.clear()
            self.cb_orange_plot_12.setTitle(orange_range['describe'].iloc[11])
            self.cb_orange_plot_12.addLegend(offset=(-30, 20))
            self.cb_orange_plot_12.plot(x=symptom_db[0], y=pd.DataFrame(symptom_db[1])[orange_range['index'].iloc[11]], pen=pyqtgraph.mkPen('b', width=3), name='Real Data')
            self.cb_orange_plot_12.plot(x=symptom_db[0], y=pd.DataFrame(compared_db)[orange_range['index'].iloc[11]], pen=pyqtgraph.mkPen('k', width=3), name=self.cb.currentText()[:7])

        [convert_red[i].setCheckable(True) for i in range(4)]
        [convert_orange[i].setCheckable(True) for i in range(12)]

    def show_another_result_table(self, all_shap):
        '''
               # shap_add_des['index'] : 변수 이름 / shap_add_des[0] : shap value
               # shap_add_des['describe'] : 변수에 대한 설명 / shap_add_des['probability'] : shap value를 확률로 환산한 값
               '''

        if self.cb.currentText() == 'Normal':
            step1 = pd.DataFrame(all_shap[0], columns=self.selected_para['0'].tolist())
        elif self.cb.currentText() == 'Ab21-01: Pressurizer pressure channel failure (High)':
            step1 = pd.DataFrame(all_shap[1], columns=self.selected_para['0'].tolist())
        elif self.cb.currentText() == 'Ab21-02: Pressurizer pressure channel failure (Low)':
            step1 = pd.DataFrame(all_shap[2], columns=self.selected_para['0'].tolist())
        elif self.cb.currentText() == 'Ab20-04: Pressurizer level channel failure (Low)':
            step1 = pd.DataFrame(all_shap[3], columns=self.selected_para['0'].tolist())
        elif self.cb.currentText() == 'Ab15-07: Steam generator level channel failure (High)':
            step1 = pd.DataFrame(all_shap[4], columns=self.selected_para['0'].tolist())
        elif self.cb.currentText() == 'Ab15-08: Steam generator level channel failure (Low)':
            step1 = pd.DataFrame(all_shap[5], columns=self.selected_para['0'].tolist())
        elif self.cb.currentText() == 'Ab63-04: Control rod fall':
            step1 = pd.DataFrame(all_shap[6], columns=self.selected_para['0'].tolist())
        elif self.cb.currentText() == 'Ab63-02: Continuous insertion of control rod':
            step1 = pd.DataFrame(all_shap[7], columns=self.selected_para['0'].tolist())
        elif self.cb.currentText() == 'Ab21-12: Pressurizer PORV opening':
            step1 = pd.DataFrame(all_shap[8], columns=self.selected_para['0'].tolist())
        elif self.cb.currentText() == 'Ab19-02: Pressurizer safety valve failure':
            step1 = pd.DataFrame(all_shap[9], columns=self.selected_para['0'].tolist())
        elif self.cb.currentText() == 'Ab21-11: Pressurizer spray valve failed opening':
            step1 = pd.DataFrame(all_shap[10], columns=self.selected_para['0'].tolist())
        elif self.cb.currentText() == 'Ab23-03: Leakage from CVCS to RCS':
            step1 = pd.DataFrame(all_shap[11], columns=self.selected_para['0'].tolist())
        elif self.cb.currentText() == 'Ab60-02: Rupture of the front end of the regenerative heat exchanger':
            step1 = pd.DataFrame(all_shap[12], columns=self.selected_para['0'].tolist())
        elif self.cb.currentText() == 'Ab59-02: Leakage at the rear end of the charging flow control valve':
            step1 = pd.DataFrame(all_shap[13], columns=self.selected_para['0'].tolist())
        elif self.cb.currentText() == 'Ab23-01: Leakage from CVCS to CCW':
            step1 = pd.DataFrame(all_shap[14], columns=self.selected_para['0'].tolist())
        elif self.cb.currentText() == 'Ab23-06: Steam generator u-tube leakage':
            step1 = pd.DataFrame(all_shap[15], columns=self.selected_para['0'].tolist())

        step2 = step1.sort_values(by=0, ascending=True, axis=1)
        step3 = step2[step2.iloc[:] < 0].dropna(axis=1).T
        self.step4 = step3.reset_index()
        col = self.step4['index']
        var = [self.selected_para['0'][self.selected_para['0'] == col_].index for col_ in col]
        val_col = [self.selected_para['1'][var_].iloc[0] for var_ in var]
        proba = [(self.step4[0][val_num] / sum(self.step4[0])) * 100 for val_num in range(len(self.step4[0]))]
        val_system = [self.selected_para['2'][var_].iloc[0] for var_ in var]
        self.step4['describe'] = val_col
        self.step4['probability'] = proba
        self.step4['system'] = val_system

        self.combo_tableWidget.setRowCount(len(self.step4))
        self.combo_tableWidget.setColumnCount(4)
        self.combo_tableWidget.setHorizontalHeaderLabels(["value_name", 'probability', 'describe', 'system'])

        header = self.combo_tableWidget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)

        [self.combo_tableWidget.setItem(i, 0, QTableWidgetItem(f"{self.step4['index'][i]}")) for i in
         range(len(self.step4['index']))]
        [self.combo_tableWidget.setItem(i, 1, QTableWidgetItem(f"{round(self.step4['probability'][i], 2)}%")) for i in
         range(len(self.step4['probability']))]
        [self.combo_tableWidget.setItem(i, 2, QTableWidgetItem(f"{self.step4['describe'][i]}")) for i in
         range(len(self.step4['describe']))]
        [self.combo_tableWidget.setItem(i, 3, QTableWidgetItem(f"{self.step4['system'][i]}")) for i in
         range(len(self.step4['system']))]

        delegate = AlignDelegate(self.combo_tableWidget)
        self.combo_tableWidget.setItemDelegate(delegate)

    def show_anoter_table(self):
        self.combo_tableWidget.show()

    def cb_red1_plot(self):
        if self.cb_red1.isChecked():
            if self.cb_red1.text().split()[0] != 'None':
                self.cb_red_plot_1.show()
                self.cb_red1.setCheckable(False)

    def cb_red2_plot(self):
        if self.cb_red2.isChecked():
            if self.cb_red2.text().split()[0] != 'None':
                self.cb_red_plot_2.show()
                self.cb_red2.setCheckable(False)

    def cb_red3_plot(self):
        if self.cb_red3.isChecked():
            if self.cb_red3.text().split()[0] != 'None':
                self.cb_red_plot_3.show()
                self.cb_red3.setCheckable(False)

    def cb_red4_plot(self):
        if self.cb_red4.isChecked():
            if self.cb_red4.text().split()[0] != 'None':
                self.cb_red_plot_4.show()
                self.cb_red4.setCheckable(False)

    def cb_orange1_plot(self):
        if self.cb_orange1.isChecked():
            if self.cb_orange1.text().split()[0] != 'None':
                self.cb_orange_plot_1.show()
                self.cb_orange1.setCheckable(False)

    def cb_orange2_plot(self):
        if self.cb_orange2.isChecked():
            if self.cb_orange2.text().split()[0] != 'None':
                self.cb_orange_plot_2.show()
                self.cb_orange2.setCheckable(False)

    def cb_orange3_plot(self):
        if self.cb_orange3.isChecked():
            if self.cb_orange3.text().split()[0] != 'None':
                self.cb_orange_plot_3.show()
                self.cb_orange3.setCheckable(False)

    def cb_orange4_plot(self):
        if self.cb_orange4.isChecked():
            if self.cb_orange4.text().split()[0] != 'None':
                self.cb_orange_plot_4.show()
                self.cb_orange4.setCheckable(False)

    def cb_orange5_plot(self):
        if self.cb_orange5.isChecked():
            if self.cb_orange5.text().split()[0] != 'None':
                self.cb_orange_plot_5.show()
                self.cb_orange5.setCheckable(False)

    def cb_orange6_plot(self):
        if self.cb_orange6.isChecked():
            if self.cb_orange6.text().split()[0] != 'None':
                self.cb_orange_plot_6.show()
                self.cb_orange6.setCheckable(False)

    def cb_orange7_plot(self):
        if self.cb_orange7.isChecked():
            if self.cb_orange7.text().split()[0] != 'None':
                self.cb_orange_plot_7.show()
                self.cb_orange7.setCheckable(False)

    def cb_orange8_plot(self):
        if self.cb_orange8.isChecked():
            if self.cb_orange8.text().split()[0] != 'None':
                self.cb_orange_plot_8.show()
                self.cb_orange8.setCheckable(False)

    def cb_orange9_plot(self):
        if self.cb_orange9.isChecked():
            if self.cb_orange9.text().split()[0] != 'None':
                self.cb_orange_plot_9.show()
                self.cb_orange9.setCheckable(False)

    def cb_orange10_plot(self):
        if self.cb_orange10.isChecked():
            if self.cb_orange10.text().split()[0] != 'None':
                self.cb_orange_plot_10.show()
                self.cb_orange10.setCheckable(False)

    def cb_orange11_plot(self):
        if self.cb_orange11.isChecked():
            if self.cb_orange11.text().split()[0] != 'None':
                self.cb_orange_plot_11.show()
                self.cb_orange11.setCheckable(False)

    def cb_orange12_plot(self):
        if self.cb_orange12.isChecked():
            if self.cb_orange12.text().split()[0] != 'None':
                self.cb_orange_plot_12.show()
                self.cb_orange12.setCheckable(False)


if __name__ == "__main__":
    # 데이터 로드
    with open(f'./DataBase/CNS_db/pkl/normal.pkl', 'rb') as f:
        normal_db = pickle.load(f)
    with open(f'./DataBase/CNS_db/pkl/ab21-01_170.pkl', 'rb') as f:
        ab21_01 = pickle.load(f)
    with open(f'./DataBase/CNS_db/pkl/ab21-02-152.pkl', 'rb') as f:
        ab21_02 = pickle.load(f)
    with open(f'./DataBase/CNS_db/pkl/ab20-04_6.pkl', 'rb') as f:
        ab20_04 = pickle.load(f)
    with open(f'./DataBase/CNS_db/pkl/ab15-07_1002.pkl', 'rb') as f:
        ab15_07 = pickle.load(f)
    with open(f'./DataBase/CNS_db/pkl/ab15-08_1071.pkl', 'rb') as f:
        ab15_08 = pickle.load(f)
    with open(f'./DataBase/CNS_db/pkl/ab63-04_113.pkl', 'rb') as f:
        ab63_04 = pickle.load(f)
    with open(f'./DataBase/CNS_db/pkl/ab63-02_5(4;47트립).pkl', 'rb') as f:
        ab63_02 = pickle.load(f)
    with open(f'./DataBase/CNS_db/pkl/ab21-12_34 (06;06트립).pkl', 'rb') as f:
        ab21_12 = pickle.load(f)
    with open(f'./DataBase/CNS_db/pkl/ab19-02-17(05;55트립).pkl', 'rb') as f:
        ab19_02 = pickle.load(f)
    with open(f'./DataBase/CNS_db/pkl/ab21-11_62(06;29트립).pkl', 'rb') as f:
        ab21_11 = pickle.load(f)
    with open(f'./DataBase/CNS_db/pkl/ab23-03-77.pkl', 'rb') as f:
        ab23_03 = pickle.load(f)
    with open(f'./DataBase/CNS_db/pkl/ab60-02_306.pkl', 'rb') as f:
        ab60_02 = pickle.load(f)
    with open(f'./DataBase/CNS_db/pkl/ab59-02-1055.pkl', 'rb') as f:
        ab59_02 = pickle.load(f)
    with open(f'./DataBase/CNS_db/pkl/ab23-01_30004(3;11_trip).pkl', 'rb') as f:
        ab23_01 = pickle.load(f)
    with open(f'./DataBase/CNS_db/pkl/ab23-06_30004(03;28 trip).pkl', 'rb') as f:
        ab23_06 = pickle.load(f)
    print('데이터 불러오기를 완료하여 모델 불러오기로 이행합니다.')
    # 모델 로드
    model_module = Model_module()
    model_module.load_model()
    print('모델 불러오기를 완료하여 GUI로 이행합니다.')
    app = QApplication(sys.argv)
    form = Mainwindow()
    exit(app.exec_())