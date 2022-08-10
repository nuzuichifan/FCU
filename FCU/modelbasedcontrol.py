import math
import numpy as np
import xlwt
import fancoilunit_coilingload




def lambe(x,y,s):#(X风量，y水流量)x,210-1020;y210-580(X风量，y水流量)
    s_up = s + s * 0.05
    s_down = s - s * 0.05

    k0=1.71104589e+01
    k11=4.69591059e+00
    k12=4.01993422e+00
    k21=-2.89549527e-03
    k22=3.67694169e-03
    k23=-4.81336674e-03
    Q=k0 + k11 * x + k12 * y + k21 * x ** 2 + k22 * x * y + k23 * y ** 2
    if Q <= s_up and Q >= s_down:
        Punish_coefficient_1 = 0
    else:
        Punish_coefficient_1 = 1000
    Pfan = (3.22086094e-05) * (x) ** 2 + (3.05481795e-02) * x + (
        3.56900355e+01)  # 风流量-功率
    Ppump = (3.68380722e-04) * (y) ** 2 + (-2.43551880e-02) * y + (
        2.48530002e+01)  # 水流量-功率
    Punish_coefficient_1 = 150 - 150 * math.exp(-((Q - s) ** 2) / (2 * (s * 0.2) ** 2))
    return  Punish_coefficient_1 + (Pfan + Ppump)

state = np.load('state_3.npy')


workbook = xlwt.Workbook(encoding = 'utf-8')
# 创建一个worksheet

worksheet_a_water = workbook.add_sheet('a_water')
worksheet_a_air = workbook.add_sheet('a_air')
worksheet_Pfan = workbook.add_sheet('Pfan')
worksheet_Ppump = workbook.add_sheet('Ppump')
worksheet_Q = workbook.add_sheet('Q')
worksheet_total_power = workbook.add_sheet('total_power')
worksheet_s_Q = workbook.add_sheet('s_Q')
for j in range (0,10):
    for i in range(4392):#(X风量，y水流量)
        s = state[i:i + 1, j:j+1]
        min_value = 10000
        min_x = 0
        min_y = 0
        if s > 0:
            for x in range(210,1021):
                for y in range(210,581):
                    value_now=lambe(x,y,s)
                #print(x,y)
                    if value_now<=min_value:
                        min_value=value_now
                        min_x=x
                        min_y=y


            environment = fancoilunit_coilingload.fan_coil_unit(cls=s, tw1=7, V_air=min_x / 3600,
                                                                W_water=min_y / 3600, t_in_dryair=28,
                                                                t_in_relative_humidity=0.6)
            cls, Q, t2, d2, tw2, W_water, V_air, Ppump, Pfan = environment.approch()
            print(i,s,Q,min_value,min_x, min_y,Pfan+Ppump)
            worksheet_Q.write(i, j, label=float(Q))
            worksheet_a_water.write(i, j, label=float(min_y))
            worksheet_a_air.write(i, j, label=float(min_x))
            worksheet_Pfan.write(i, j, label=Pfan)
            worksheet_Ppump.write(i, j, label=Ppump)
            worksheet_total_power.write(i, j, label=Ppump + Pfan)
            worksheet_s_Q.write(i, j, label=float(s - Q))
workbook.save('bianlifa.xls')

