import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
from collections import deque
import matplotlib.pyplot as plt
import matplotlib

class Rule_module:
    def __init__(self):
        pass

    def percent_procedure(self, check_data, check_parameter):
        # procedure = [f'{pro[:4]}_{pro[5:7]}' for pro in test_db]
        procedure = ['ab21_01', 'ab21_02', 'ab20_01', 'ab20_04', 'ab15_07', 'ab15_08', 'ab63_04', 'ab63_02',
                     'ab63_03', 'ab21_12', 'ab19_02', 'ab21_11', 'ab59_01', 'ab80_02', 'ab64_03', 'ab60_02', 'ab23_03',
                     'ab59_02', 'ab23_01', 'ab23_06']
        hist = []
        for file_name in procedure: globals()[file_name] = []  # 비정상 시나리오 이름별 변수 생성
        s253 = bool(int(check_data['KZBANK4'] >= 220))  # KLAMPO253 : ALARM: CONTROL BANK D FULL ROD WITHDRAWL(220 STEPS)
        s254 = bool(sum([bool(int(check_data[f'KZROD{_}'] < check_data['KRIL'])) for _ in range(1, 9, 1)]))  # KLAMPO254
        s255 = bool(sum([bool(int(check_data[f'KZROD{_}'])) for _ in range(1, 53, 1)]) >= 2)  # KLAMPO255
        s260 = bool(int(check_data['WNETLD'] < check_data['CWLHXL']))  # KLAMPO260
        s262 = bool(int(check_data['URHXUT'] > check_data['CURHX']))  # KLAMPO262
        s263 = bool(int(check_data['ZVCT'] < check_data['CZVCT2']))  # KLAMPO263
        s266 = bool(int(check_data['WCHGNO'] < check_data['CWCHGL']))  # KLAMPO266
        s268 = bool(int(check_data['WNETLD'] > check_data['CWLHXH']))  # KLAMPO268
        s271 = bool(int(check_data['ZVCT'] > check_data['CZVCT6']))  # KLAMPO271
        s274 = bool(int(check_data['WCHGNO'] > check_data['CWCHGH']))  # KLAMPO274
        s307 = bool(int(check_data['PPRZN'] > check_data['CPPRZH']))  # KLAMPO307
        s308 = bool(int(check_data['PPRZN'] < check_data['CPPRZL']))  # KLAMPO308
        s309 = bool(int(check_data['BPORV'] > 0.01))  # KLAMPO309
        s310 = bool(int(check_data['ZPRZNO'] > (check_data['ZPRZSP'] + check_data['CZPRZH'])) and int(check_data['QPRZB'] > check_data['CQPRZP']))  # KLAMPO310
        s312 = bool(int(check_data['PPRZN'] < check_data['CQPRZB']) and int(check_data['KBHON'] == 1))  # KLAMPO312
        s313 = bool(int((check_data['UAVLEGS'] - check_data['UAVLEGM']) > check_data['CUTDEV']))  # KLAMPO313
        s314 = bool(int(check_data['UAVLEGS'] > check_data['CUTAVG']))  # KLAMPO314
        s315 = bool(sum([bool(int(check_data[f'UAVLEG{_}'] > float(pd.DataFrame([check_data['UAVLEG1'], check_data['UAVLEG2'], check_data['UAVLEG3']]).max()))) for _ in range(1, 4, 1)]))  # KLAMPO315
        s317 = bool(int(check_data['UPRT'] > check_data['CUPRT']))  # KLAMPO317
        s320 = bool(int((check_data['WSTM1'] - check_data['WFWLN1']) > (check_data['WSTM1'] * 0.1)) or int((check_data['WSTM2'] - check_data['WFWLN2']) > (check_data['WSTM2'] * 0.1)) or int((check_data['WSTM3'] - check_data['WFWLN3']) > (check_data['WSTM3'] * 0.1)))  # KLAMPO320
        s325 = bool(sum([bool(int(check_data[f'BHV{_}08'])) for _ in range(1, 4, 1)]))  # KLAMPO325
        s338 = bool(sum([bool(int(check_data[f'ZSGNOR{_}'] > check_data['CPERMS'])) for _ in range(1, 4, 1)]) and int(check_data['KTBTRIP'] == 1))  # KLAMPO338
        ab21_01_para1 = bool(int(check_data['ZINST66'] > 0))  # PRZ SPRAY FLOW
        ab21_01_para2 = bool(int(check_data['QPRZH'] == 0))  # PROPORTIONAL HEATER FRACTIONAL POWER
        ab21_01_para3 = bool(int(check_data['KLAMPO118'] == 0))  # BACK-UP HEATER ON
        ab21_01_para4 = bool(int(check_data['BPORV'] == 0))  # PORV
        ab21_02_para1 = bool(int(check_data['BPORV'] == 0))  # PORV
        ab20_01_para1 = bool(int(check_parameter.iloc[0]['WCHGNO'] >= check_parameter.iloc[1]['WCHGNO']))  # 충전유량 감소
        ab20_01_para2 = bool(int(check_parameter.iloc[0]['ZPRZNO'] >= check_parameter.iloc[1]['ZPRZNO']))  # 실제 가압기 수위 감소
        ab20_04_para1 = bool(int(check_parameter.iloc[0]['WCHGNO'] <= check_parameter.iloc[1]['WCHGNO']))  # 충전유량 증가
        ab20_04_para2 = bool(int(check_parameter.iloc[0]['ZPRZNO'] <= check_parameter.iloc[1]['ZPRZNO']))  # 실제 가압기 수위 증가
        ab63_04_para1 = bool(int(check_parameter.iloc[0]['QPROREL'] >= check_parameter.iloc[1]['QPROREL']))  # 원자로 출력 감소
        ab63_04_para2 = bool(int(check_parameter.iloc[0]['UAVLEGM'] >= check_parameter.iloc[1]['UAVLEGM']))  # Tavg 감소
        ab63_02_para1 = bool(int(check_parameter.iloc[0]['QPROREL'] >= check_parameter.iloc[1]['QPROREL']))  # 원자로 출력 감소
        ab63_02_para2 = bool(sum([check_parameter.iloc[0][f'KBCDO{_}'] > check_parameter.iloc[1][f'KBCDO{_}'] for _ in range(7, 11, 1)]))  # 계속적인 제어봉 삽입
        ab63_03_para1 = bool(sum([check_parameter.iloc[0][f'KBCDO{_}'] < check_parameter.iloc[1][f'KBCDO{_}'] for _ in range(7, 11, 1)]))  # 계속적인 제어봉 인출 (뭔가 좀 이상한 부분임)
        ab63_03_para2 = bool(int(check_parameter.iloc[0]['QPROREL'] <= check_parameter.iloc[1]['QPROREL']))  # 원자로 출력 증가
        ab19_02_para1 = bool(int(check_parameter.iloc[0]['ZVCT'] >= check_parameter.iloc[1]['ZVCT']))  # VCT 수위 감소
        ab19_02_para2 = bool(int(check_parameter.iloc[0]['WCHGNO'] <= check_parameter.iloc[1]['WCHGNO']))  # 충전유량 증가
        ab21_11_para1 = bool(int(check_parameter.iloc[0]['ZINST63'] <= check_parameter.iloc[1]['ZINST63']))  # 가압기 수위 증가
        ab21_11_para2 = bool(int(check_data['KLAMPO216'] > 0.01))  # KLAMPO309
        ab23_03_para1 = bool(int(check_parameter.iloc[0]['ZVCT'] >= check_parameter.iloc[1]['ZVCT']))  # VCT 수위 감소
        ab23_03_para2 = bool(int(check_parameter.iloc[0]['WCHGNO'] <= check_parameter.iloc[1]['WCHGNO']))  # 충전유량 증가
        ab80_02_para1 = bool(int(check_data['KSWO28'] == 1) and int(sum([check_parameter.iloc[0][f'KBCDO{_}'] > check_parameter.iloc[1][f'KBCDO{_}'] for _ in range(7, 11, 1)])))  # 제어봉 AUTO의 경우, 제어봉이 자동 삽입
        ab60_02_para1 = bool(int(check_data['QPRZH'] == 0) and int(check_data['KLAMPO118'] == 0))  # 가압기 모든 전열기 꺼짐
        ab60_02_para2 = bool(int(check_parameter.iloc[0]['UNRHXUT'] >= check_parameter.iloc[1]['UNRHXUT']))  # 재생열교환기 후단 유출수 온도 감소
        ab60_02_para3 = bool(int(check_parameter.iloc[0]['UCHGUT'] >= check_parameter.iloc[1]['UCHGUT']))  # 재생열교환기 후단 충전수 온도 감소
        ab59_02_para1 = bool(int(check_parameter.iloc[0]['WCHGNO'] <= check_parameter.iloc[1]['WCHGNO']))  # 충전유량 증가
        ab59_02_para2 = bool(int(check_parameter.iloc[0]['ZINST63'] >= check_parameter.iloc[1]['ZINST63']))  # 가압기 수위 감소
        ab59_02_para3 = bool(int(check_parameter.iloc[0]['ZVCT'] >= check_parameter.iloc[1]['ZVCT']))  # VCT 수위 감소
        ab23_01_para1 = bool(int(check_parameter.iloc[0]['ZINST63'] >= check_parameter.iloc[1]['ZINST63']))  # 가압기 수위 감소
        ab23_01_para2 = bool(int(check_parameter.iloc[0]['ZINST65'] >= check_parameter.iloc[1]['ZINST65']))  # 가압기 압력 감소
        ab23_01_para3 = bool(int(check_parameter.iloc[0]['ZVCT'] >= check_parameter.iloc[1]['ZVCT']))  # VCT 수위 감소
        ab23_01_para4 = bool(int(check_parameter.iloc[0]['WCHGNO'] <= check_parameter.iloc[1]['WCHGNO']))  # 충전유량 증가
        ab23_01_para5 = bool(int(check_parameter.iloc[0]['ZINST22'] <= check_parameter.iloc[1]['ZINST22']))  # 격납용기 대기 방사선 증가
        ab23_06_para1 = bool(int(check_parameter.iloc[0]['ZINST63'] >= check_parameter.iloc[1]['ZINST63']))  # 가압기 수위 감소
        ab23_06_para2 = bool(int(check_parameter.iloc[0]['ZINST65'] >= check_parameter.iloc[1]['ZINST65']))  # 가압기 압력 감소
        ab23_06_para3 = bool(int(check_parameter.iloc[0]['ZVCT'] >= check_parameter.iloc[1]['ZVCT']))  # VCT 수위 감소
        ab59_01_para1 = bool(int(check_parameter.iloc[0]['WCHGNO'] >= check_parameter.iloc[1]['WCHGNO']))  # 충전유량 감소
        ab59_01_para2 = bool(int(check_parameter.iloc[0]['ZINST63'] >= check_parameter.iloc[1]['ZINST63']))  # 가압기 수위 감소
        ab59_01_para3 = bool(int(check_parameter.iloc[0]['ZVCT'] <= check_parameter.iloc[1]['ZVCT']))  # VCT 수위 증가
        ab59_01_para4 = bool(sum([bool(int(check_parameter.iloc[0][f'WRCPSI{_}'] <= check_parameter.iloc[1][f'WRCPSI{_}'])) for _ in range(1, 4, 1)]))  # RCP 밀봉수 주입유량 감소
        ab21_01.append([s308, ab21_01_para1, ab21_01_para2, ab21_01_para3, ab21_01_para4])
        ab21_02.append([s312, s307, s309, ab21_02_para1])
        ab20_01.append([s310, s312, s266, s274, ab20_01_para1, ab20_01_para2])
        ab20_04.append([s274, s266, ab20_04_para1, ab20_04_para2])
        ab15_07.append([s320])
        ab15_08.append([s320])
        ab63_04.append([s313, s255, ab63_04_para1, ab63_04_para2])
        ab63_02.append([s313, s254, ab63_02_para1, ab63_02_para2])
        ab63_03.append([s313, s253, ab63_03_para1, ab63_03_para2])
        ab21_12.append([s309, s312, s308])
        ab19_02.append([s312, s308, s317, ab19_02_para1, ab19_02_para2])
        ab21_11.append([s312, s308, s309, ab21_11_para1, ab21_11_para2])
        ab23_03.append([ab23_03_para1, ab23_03_para2])
        ab80_02.append([s320, s314, s315, s310, ab80_02_para1])
        ab60_02.append([s260, s268, s263, s271, ab60_02_para1, ab60_02_para2, ab60_02_para3])
        ab59_02.append([s274, s266, s271, s263, ab59_02_para1, ab59_02_para2, ab59_02_para3])
        ab23_01.append([ab23_01_para1, ab23_01_para2, ab23_01_para3, ab23_01_para4, ab23_01_para5])
        ab23_06.append([s320, ab23_06_para1, ab23_06_para2, ab23_06_para3])
        ab59_01.append([s266, s274, s263, s271, s262, ab59_01_para1, ab59_01_para2, ab59_01_para3, ab59_01_para4])
        ab64_03.append([s325, s338, s320])

        for file_name in procedure:
            globals()[f'percent_{file_name}'] = (sum(globals()[file_name][0]) / len(
                globals()[file_name][0])) * 100  # 비정상 시나리오 이름별 확률계산
            hist.append(globals()[f'percent_{file_name}'])
        temp = pd.DataFrame(hist, index=procedure).sort_values(by=0, ascending=True)
        return temp
