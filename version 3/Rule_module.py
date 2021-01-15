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
        hist, total, part = [], [], []
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
            globals()[f'percent_{file_name}'] = (sum(globals()[file_name][0]) / len(globals()[file_name][0])) * 100  # 비정상 시나리오 이름별 확률계산
            hist.append(globals()[f'percent_{file_name}'])
            total.append(len(globals()[file_name][0]))
            part.append(sum(globals()[file_name][0]))
        # temp = pd.DataFrame(hist, index=procedure).sort_values(by=0, ascending=True)
        temp = pd.DataFrame([np.array(hist), np.array(part), np.array(total)-np.array(part)], columns=procedure, index=['Prob', 'Satisfaction', 'Total']).T.sort_values(by='Prob', ascending=True)

        return temp

    def system_daignosis(self, check_data):
        procedure = ['reactivity_control_system', 'rod_contol_system', 'component_cooling_system', 'safety_injection_system', 'chemical_volume_control_system', 'containment_system', 'reactor_coolant_system', 'main_steam_system', 'feed_water_system', 'aux_feed_water_system', 'condenser_system']
        sys, total, part = [], [], []
        for file_name in procedure: globals()[file_name] = []  # 비정상 시나리오 이름별 변수 생성
        # reactivity_control_system
        reactivity1 = bool(int(check_data['XPIRM'] > check_data['CIRFH'])) #KLAMPO251
        reactivity2 = bool(int(check_data['QPROREL'] > check_data['CPRFH'])) #KLAMPO252
        reactivity3 = bool(int(check_data['CAXOFF'] >= check_data['CAXOFDL']) or int(check_data['CAXOFF'] < check_data['CAXOFDL']-0.15)) #KLAMPO256

        # rod_contol_system
        rod1 = bool(int(check_data['KZBANK4'] >= 220)) #KLAMPO253
        rod2 = bool(sum([bool(int(check_data[f'KZROD{_}'] < check_data['KRIL'])) for _ in range(1, 9, 1)]))  # KLAMPO254
        rod3 = bool(sum([bool(int(check_data[f'KZROD{_}'])) for _ in range(1, 53, 1)]) >= 2)  # KLAMPO255

        # component_cooling_system
        ccw1 = bool(int(check_data['UCCWIN'] >= check_data['CUCCWH'])) #KLAMPO257

        # safety_injection_system
        sis1 = bool(int(check_data['ZRWST'] <= check_data['CZRWSLL'])) #KLAMPO259
        sis2 = bool(int(check_data['PACCTK'] < check_data['CPACCL']) or int(check_data['PACCTK'] > check_data['CPACCH'])) #KLAMPO305 & KLAMPO306

        # chemical_volume_control_system
        cvcs1 = bool(int(check_data['WNETLD'] < check_data['CWLHXL']))  #KLAMPO260
        cvcs2 = bool(int(check_data['URHXUT'] > check_data['CURHX']))  #KLAMPO262
        cvcs3 = bool(int(check_data['ZVCT'] < check_data['CZVCT2']))  #KLAMPO263
        cvcs4 = bool(int(check_data['PVCT'] < check_data['CPVCTL'])) #KLAMPO264
        cvcs5 = bool(int(check_data['WCHGNO'] < check_data['CWCHGL']))  #KLAMPO266
        cvcs6 = bool(int(check_data['WNETLD'] > check_data['CWLHXH']))  #KLAMPO268
        cvcs7 = bool(int(check_data['ZVCT'] > check_data['CZVCT6']))  #KLAMPO271
        cvcs8 = bool(int(check_data['PVCT'] > check_data['CPVCTH'])) #KLAMPO272
        cvcs9 = bool(int(check_data['WCHGNO'] > check_data['CWCHGH'])) # KLAMPO274

        # containment_system
        ctmt1 = bool(int(check_data['KCSAS'] == 1)) #KLAMPO270
        ctmt2 = bool(int(check_data['KCISOB'] == 1)) #KLAMPO273
        ctmt3 = bool(int(check_data['ZSUMP'] > check_data['CZSMPH'])) #KLAMPO295 & KLAMPO296
        ctmt4 = bool(int(check_data['UCTMT'] > check_data['CUCTMT'])) #KLAMPO297
        ctmt5 = bool(int(check_data['HUCTMT'] > check_data['CHCTMT'])) #KLAMPO298
        ctmt6 = bool(int(check_data['DCTMT'] > check_data['CRADHI'])) #KLAMPO301
        ctmt7 = bool(int((check_data['PCTMT']*check_data['PAKGCM']) > check_data['CPCMTH'])) #KLAMPO302 & KLAMPO303 & KLAMPO304

        # reactor_coolant_system
        rcs1 = bool(sum([bool(int(check_data[f'WRCPSI{_}'] < check_data['CWRCPS'])) for _ in range(1, 4, 1)]))  # KLAMPO265
        rcs2 = bool(int(check_data['PPRZN'] > check_data['CPPRZH']) or int(check_data['PPRZN'] < check_data['CPPRZL']))  #KLAMPO307 & KLAMPO308
        rcs3 = bool(int(check_data['BPORV'] > 0.01))  #KLAMPO309
        rcs4 = bool((int(check_data['ZPRZNO'] > (check_data['ZPRZSP'] + check_data['CZPRZH'])) and int(check_data['QPRZB'] > check_data['CQPRZP'])) or (int(check_data['ZPRZNO'] < check_data['CZPRZL']) and int(check_data['QPRZ'] <= check_data['CQRPZP'])))  # KLAMPO310 & KLAMPO311
        rcs5 = bool(int(check_data['PPRZN'] < check_data['CQPRZB']) and int(check_data['KBHON'] == 1))  #KLAMPO312
        rcs6 = bool(int((check_data['UAVLEGS'] - check_data['UAVLEGM']) > check_data['CUTDEV']))  #KLAMPO313
        rcs7 = bool(int(check_data['UAVLEGS'] > check_data['CUTAVG']))  #KLAMPO314
        rcs8 = bool(sum([bool(int(check_data[f'UAVLEG{_}'] > float(pd.DataFrame([check_data['UAVLEG1'], check_data['UAVLEG2'], check_data['UAVLEG3']]).max()))) for _ in range(1, 4, 1)]))  # KLAMPO315
        rcs9 = bool(sum([bool(int(check_data[f'WSGRCP{_}'] < check_data['CWSGRL'])) for _ in range(1, 4, 1)]) and sum([bool(int(check_data[f'KRCP{a}'] == 1)) for a in range(1,4,1)])) #KLAMPO316
        rcs10 = bool(int(check_data['UPRT'] > check_data['CUPRT']))  # KLAMPO317
        rcs11 = bool(int((check_data['PPRT']-98000) > check_data['CPPRT'])) #KLAMPO318
        rcs12 = bool(sum([bool(int(check_data[f'KRCP{_}'])) for _ in range(1, 4, 1)]) == 0)  #KLAMPO321

        #main_steam_system
        mss1 = bool(int((check_data['WSTM1'] - check_data['WFWLN1']) > (check_data['WSTM1'] * 0.1)) or int((check_data['WSTM2'] - check_data['WFWLN2']) > (check_data['WSTM2'] * 0.1)) or int((check_data['WSTM3'] - check_data['WFWLN3']) > (check_data['WSTM3'] * 0.1)))  # KLAMPO320
        mss2 = bool(sum([bool(int(check_data[f'BHV{_}08'])) for _ in range(1, 4, 1)]))  # KLAMPO325
        mss3 = bool(sum([bool(int((check_data['PSGLP'] - check_data[f'PSG{_}']) < check_data['CMSLH'])) for _ in range(1, 4, 1)]))  # KLAMPO326
        mss4 = bool(sum([bool(int(check_data[f'PSG{_}'] < check_data['CPSTML'])) for _ in range(1, 4, 1)]))  # KLAMPO254

        #feed_water_system
        fws1 = bool(int((check_data['WSTM1'] - check_data['WFWLN1']) > (check_data['WSTM1'] * 0.1)) or int((check_data['WSTM2'] - check_data['WFWLN2']) > (check_data['WSTM2'] * 0.1)) or int((check_data['WSTM3'] - check_data['WFWLN3']) > (check_data['WSTM3'] * 0.1)))  # KLAMPO320
        fws2 = bool(sum([bool(int(check_data[f'ZSGNOR{_}'] < check_data['CZSGW'])) for _ in range(1, 4, 1)]))  #KLAMPO319
        fws3 = bool(int(check_data['PFWPOUT'] > check_data['CPFWOH']))  #KLAMPO331
        fws4 = bool(sum([bool(int(check_data[f'KFWP{_}'] == 0)) for _ in range(1, 4, 1)])) #KLAMPO332
        fws5 = bool(int(check_data['UFDW'] > check_data['CUFWH']))  #KLAMPO333
        fws6 = bool(sum([bool(int(check_data[f'ZSGNOR{_}'] > check_data['CPERMS'])) for _ in range(1, 4, 1)]) and int(check_data['KTBTRIP'] == 1))  # KLAMPO338

        #aux_feed_water_system
        afws1 = bool(int(check_data['KAFWP1'] == 1) or int(check_data['KAFWP3'] == 1)) #KLAMPO329

        #condenser_system
        cs1 = bool(int(check_data['ZCNDTK'] < check_data['CZCTKL']) or int(check_data['ZCNDTK'] > check_data['CZCTKH'])) #KLAMPO322 and KLAMPO323 and KLAMPO324
        cs2 = bool(int(check_data['ZCOND'] < check_data['CZCNDL']) or int(check_data['ZCOND'] > check_data['CZCNDH'])) #KLAMPO330 and KLAMPO336
        cs3 = bool(int(check_data['WCDPO'] < check_data['CWCDPO'])) #KLAMPO334
        cs4 = bool(int(check_data['PVAC'] < check_data['CPVACH'])) #KLAMPO335

        reactivity_control_system.append([reactivity1, reactivity2, reactivity3])
        rod_contol_system.append([rod1, rod2, rod3])
        component_cooling_system.append([ccw1])
        safety_injection_system.append([sis1, sis2])
        chemical_volume_control_system.append([cvcs1, cvcs2, cvcs3, cvcs4, cvcs5, cvcs6, cvcs7, cvcs8, cvcs9])
        containment_system.append([ctmt1, ctmt2, ctmt3, ctmt4, ctmt5, ctmt6, ctmt7])
        reactor_coolant_system.append([rcs1, rcs2, rcs3, rcs4, rcs5,  rcs6, rcs7, rcs8, rcs9, rcs10, rcs11, rcs12])
        main_steam_system.append([mss1, mss2, mss3, mss4])
        feed_water_system.append([fws1, fws2, fws3, fws4, fws5, fws6])
        aux_feed_water_system.append([afws1])
        condenser_system.append([cs1, cs2, cs3, cs4])

        for file_name in procedure:
            globals()[f'percent_{file_name}'] = (sum(globals()[file_name][0]) / len(globals()[file_name][0])) * 100  # 비정상 시나리오 이름별 확률계산
            sys.append(globals()[f'percent_{file_name}'])
            total.append(len(globals()[file_name][0]))
            part.append(sum(globals()[file_name][0]))
        # sys_ = pd.DataFrame(sys, index=procedure).sort_values(by=0, ascending=True)
        sys_ = pd.DataFrame([np.array(sys), np.array(part), np.array(total)-np.array(part)], columns=procedure, index=['Prob', 'Satisfaction', 'Total']).T.sort_values(by='Prob', ascending=True)
        return sys_






