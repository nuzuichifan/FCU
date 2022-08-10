import numpy as np
import xlwt
import fancoilunit_coilingload
'''
(x,y):(水流量,风流量)
(200-300,200-400)  (250,300)   =(-2,2)*25+250  =(-2,2)*50+300   (250,300)2188.988 
                                (200,200)1600 不准  (200,400)2422.951  (300,200)1777 (300,400)2706
(300-400,400-600)  (350,500)   =(-2,2)*25+350  =(-2,2)*50+500   (350,500)2788.749
                                (300,400)2706 (300,600)3229 (400,400)2900 (400,600)3518
(400-500,600-800)  (450,700)   =(-2,2)*25+450  =(-2,2)*50+700
                                (400,600)3518 (400,800)3891 (500,600)3715 (500,800)4134
(500-600,800-1000) (550,900)   =(-2,2)*25+550  =(-2,2)*50+900
                                (500,800)4134 (500,1000)4516 (600,800)4282 (600,1000)4740
综上：0-2700，2700-3500，3500-4100，4100-inf
'''

a_air = 0
a_water = 0
Q= 0
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
for j in range(0,10):
    for i in range(4392):
        s = state[i:i + 1, j:j+1]
        if s>0:
            if s > 0 and s <= 2700:
                a_air = 300
                a_water = 250
                environment = fancoilunit_coilingload.fan_coil_unit(cls=s, tw1=7, V_air=a_air / 3600,
                                                                    W_water=a_water / 3600, t_in_dryair=28,
                                                                    t_in_relative_humidity=0.6)
                cls, Q, t2, d2, tw2, W_water, V_air, Ppump, Pfan = environment.approch()

            if s > 2700 and s <= 3500:
                a_air = 500
                a_water = 350
                environment = fancoilunit_coilingload.fan_coil_unit(cls=s, tw1=7, V_air=a_air / 3600,
                                                                    W_water=a_water / 3600, t_in_dryair=28,
                                                                    t_in_relative_humidity=0.6)
                cls, Q, t2, d2, tw2, W_water, V_air, Ppump, Pfan = environment.approch()
            if s > 3500 and s <= 4100:
                a_air = 700
                a_water = 450
                environment = fancoilunit_coilingload.fan_coil_unit(cls=s, tw1=7, V_air=a_air / 3600,
                                                                    W_water=a_water / 3600, t_in_dryair=28,
                                                                    t_in_relative_humidity=0.6)
                cls, Q, t2, d2, tw2, W_water, V_air, Ppump, Pfan = environment.approch()
            if s > 4100:
                a_air = 900
                a_water = 550
                environment = fancoilunit_coilingload.fan_coil_unit(cls=s, tw1=7, V_air=a_air / 3600,
                                                                    W_water=a_water / 3600, t_in_dryair=28,
                                                                    t_in_relative_humidity=0.6)
                cls, Q, t2, d2, tw2, W_water, V_air, Ppump, Pfan = environment.approch()

            worksheet_Q.write(i, j, label=float(Q))
            worksheet_a_water.write(i, j, label=float(a_water))
            worksheet_a_air.write(i, j, label=float(a_air))
            worksheet_Pfan.write(i, j, label=Pfan)
            worksheet_Ppump.write(i, j, label=Ppump)
            worksheet_total_power.write(i, j, label=Ppump + Pfan)
            worksheet_s_Q.write(i, j, label=float(s - Q))
workbook.save('baseline.xls')
