import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
#
from sklearn.metrics import mean_absolute_error
import pandas as pd
import math
from CoolProp.HumidAirProp import HAPropsSI
from CoolProp.CoolProp import PropsSI
# def get_air_parameters(t,Phi_a):
#     T=t+273.15
#     B=101.325
#     ps=math.exp(-5800.2206/T+1.3914993-0.048640239*T+0.000041764768*T**2-0.000000014452093*T**3+6.5459673*math.log(T))/1000
#     pv=ps*Phi_a       #kpa
#     d=0.621945*pv/(B-pv)
#     h=1.006*t+d*(2501+1.86*t)
#     rho_air=(0.003484*B/T-0.00134*pv/T)*1000
#     td=6.54+14.526*math.log(pv)+0.7389*math.log(pv)**2+0.09486*math.log(pv)**3+0.4569*pv**0.1984
#     mu_a=0.000017894*((T/288.15)**1.5)*(288.15+B)/(T+B)
#     lambda_a=(((T/273.16)**1.5)*(273.16+194)/(T+194))*0.000017161
#     return t,Phi_a,d,h,pv,ps,rho_air,td,mu_a,lambda_a
class physical_parameter_air2:
    def __init__(self,temperature_air,relative_humidity_air,B=101325,):
        self.temperature_air=temperature_air
        self.relative_humidity_air=relative_humidity_air
        self.B=B
        self.T=self.temperature_air+273.15
    def wd(self):#温度
        return self.temperature_air
    def xdsd(self):#相对湿度
        return self.relative_humidity_air
    def hsl(self):#含湿量（kgwater/kgair）
        return HAPropsSI('W', 'T', self.T, 'P', self.B, 'R', self.relative_humidity_air)
    def bh(self):#比焓（j/kg）
        return HAPropsSI('H', 'T', self.T, 'P', self.B, 'R', self.relative_humidity_air)
    def szqfyl(self):#水蒸气分压力（pa）
        return HAPropsSI('P_w', 'T', self.T, 'P', self.B, 'R', self.relative_humidity_air)
    def md(self):#密度（kg/m3）
         return 0.003484*self.B/self.T-0.00134*self.szqfyl()/self.T
    def ldwd(self):#露点温度（°C）
        return HAPropsSI('D', 'T', self.T, 'P', self.B, 'R', self.relative_humidity_air)-273.15
    def dlnd(self):#空气动力粘度（m2/s）
        return HAPropsSI('M', 'T', self.T, 'P', self.B, 'R', self.relative_humidity_air)/self.md()
    def drl(self):#导热率（W/(mK)）
        return HAPropsSI('K', 'T', self.T, 'P', self.B, 'R', self.relative_humidity_air)

# air=physical_parameter_air2(30,0.6)
# print(air.hsl())
# print(air.bh())
# print(air.szqfyl())
# print(air.md())
# print(air.ldwd())
# print(air.dlnd())
# print(air.drl())


class physical_parameter_air:
    def __init__(self,temperature_air,relative_humidity_air,B=101.325,):
        self.temperature_air=temperature_air
        self.relative_humidity_air=relative_humidity_air
        self.B=B
        self.T=self.temperature_air+273.15
    def wd(self):#温度
        return self.temperature_air
    def xdsd(self):#相对湿度
        return self.relative_humidity_air
    def bhszqfyl(self):#饱和水蒸气分压力
        return math.exp(-5800.2206/self.T+1.3914993-0.048640239*self.T+0.000041764768*self.T**2-0.000000014452093*self.T**3+6.5459673*math.log(self.T))/1000
    def szqfyl(self):#水蒸气分压力
        return self.relative_humidity_air*self.bhszqfyl()
    def hsl(self):#含湿量
        return 0.621945*self.szqfyl()/(self.B-self.szqfyl())
    def bh(self):#比焓
        return 1.006*self.temperature_air+self.hsl()*(2501+1.86*self.temperature_air)
    def md(self):#密度
        return (0.003484*self.B/self.T-0.00134*self.szqfyl()/self.T)*1000
    def ldwd(self):#ldwd
        return 6.54+14.526*math.log(self.szqfyl())+0.7389*math.log(self.szqfyl())**2+0.09486*math.log(self.szqfyl())**3+0.4569*self.szqfyl()**0.1984
    def nzxs(self):#粘滞系数
        return 0.000017894*((self.T/288.15)**1.5)*(288.15+self.B)/(self.T+self.B)
    def drl(self):#导热率
        return (((self.T/273.16)**1.5)*(273.16+194)/(self.T+194))*0.000017161
# air=physical_parameter_air(30,0.6)
# print(air.hsl())
# print(air.bhszqfyl())
# print(air.szqfyl())
# print(air.bh())
# print(air.md())
# print(air.ldwd())
# print(air.nzxs())
# print(air.drl())
#粘滞系数和导热率的计算有问题


class physical_parameter_water:
    def __init__(self,temperature_water):
        self.temperature_water=temperature_water
    def ydnd(self):
        test_ydnd=np.load("D:/model/fancoilunit/ydnd.npy")
        return (test_ydnd[0])*self.temperature_water**3+(test_ydnd[1])*self.temperature_water**2+(test_ydnd[2])*self.temperature_water+(test_ydnd[3])
    def Pr(self):
        test_Pr=np.load("D:/model/fancoilunit/Pr.npy")
        return (test_Pr[0])*self.temperature_water**3+(test_Pr[1])*self.temperature_water**2+(test_Pr[2])*self.temperature_water+(test_Pr[3])
    def md(self):
        test_md=np.load("D:/model/fancoilunit/md.npy")
        return (test_md[0])*self.temperature_water**3+(test_md[1])*self.temperature_water**2+(test_md[2])*self.temperature_water+(test_md[3])
    def brr(self):
        test_brr=np.load("D:/model/fancoilunit/brr.npy")
        return (test_brr[0])*self.temperature_water**3+(test_brr[1])*self.temperature_water**2+(test_brr[2])*self.temperature_water+(test_brr[3])*1000
    def drl(self):
        test_drl=np.load("D:/model/fancoilunit/drl.npy")
        return (test_drl[0])*self.temperature_water**3+(test_drl[1])*self.temperature_water**2+(test_drl[2])*self.temperature_water+(test_drl[3])
    def dlnd(self):
        test_dlnd=np.load("D:/model/fancoilunit/dlnd.npy")
        return (test_dlnd[0])*self.temperature_water**3+(test_dlnd[1])*self.temperature_water**2+(test_dlnd[2])*self.temperature_water+(test_dlnd[3])
# water=physical_parameter_water(30)
# print(water.Pr())
class fan_coil_unit:
    def __init__(self,cls,tw1,V_air,W_water,t_in_dryair,t_in_relative_humidity,delta_f=0.0002,delta_t=0.001,do=0.01,di=0.008,s1=0.025,s2=0.02,n_p=2,nv=8,ndis=2,sf=0.0022,Le=1,lamda_t=379.14,Ly=0.2,lamda_f=236):
        self.tw1=tw1    #进水温度
        self.cls=cls
        self.V_air=V_air#循环风量 m3/s
        self.W_water=W_water#(水的质量流量 kg/s)
        self.t_in_dryair=t_in_dryair#(进风干球温度 °C)
        self.t_in_relative_humidity=t_in_relative_humidity#（进风相对湿度 无量纲）
        self.delta_f=delta_f#翅片厚度 m
        self.delta_t=delta_t#管壁厚度 m
        self.do=do#管子外径 m
        self.di=di#管子内径 m
        self.s1=s1#垂直气流方向管间距m
        self.s2=s2#沿七六方向管间距m
        self.n_p=n_p#沿气流方向管排数
        self.nv=nv#v垂直七六方向管排数
        self.ndis=ndis#水路数
        self.sf=sf#翅片间距m
        self.Le=Le#换热管有效长度m
        self.Ly=Ly#宽m
        self.Lt=Le*ndis*nv#
        self.fi=Le*di*math.pi
        self.ft=Le*do*math.pi
        self.lamda_t=lamda_t#紫铜管导热系数w/（m*k）
        self.lamda_f=lamda_f#铝翅片导热系数w/（m*k）


    # def Q_waterside(self):
    #     water = physical_parameter_water(0.5*(self.tw1+self.tw2))
    #     ref=self.di*self.vw/water.ydnd()
    #     nuf=0.023*(ref**0.8)*((water.Pr())**0.4)
    #     ai=water.drl()*nuf/self.di
    #     Q=ai*(self.ndis*self.nv*self.Le)*(self.di*math.pi)*(self.twallin-0.5*(self.tw1+self.tw2))
    def a_water(self,tw2):#水侧换热系数
        water=physical_parameter_water(0.5*(self.tw1+tw2))
        v_water=self.W_water/(1000*((self.di/2)**2)*math.pi)
        ref=self.di*v_water/water.ydnd()
        prf=water.Pr()
        nuf=0.023*(ref**0.8)*(prf**0.4)
#        print(v_water,prf)
        return water.drl()*nuf/self.di

    def a_air(self):#空气侧换热系数
        air_in = physical_parameter_air2(self.t_in_dryair, self.t_in_relative_humidity)
        db=self.do+2*self.delta_f
        Lp=self.n_p*self.s2
        deq=2*(self.s1-db)*(self.sf-self.delta_f)/((self.s1-db)+(self.sf-self.delta_f))
        omig_max=self.s1*self.sf*((self.V_air/(self.Le*self.Ly)))/(self.s1-self.do-2*self.delta_f)/(self.sf-self.delta_f)
        ref=deq*omig_max/air_in.dlnd()
        A=0.518-0.02315*(Lp/deq)+0.000425*((Lp/deq)**2)-0.000003*((Lp/deq)**3)
        n=0.45+0.0066*Lp/deq
        m=-0.28+0.00008*ref
        C=A*(1.36-0.00024*ref)
        nuf=C*(ref**n)*((Lp/deq)**m)
        return air_in.drl()/deq*nuf
    def yita_s(self):
        L=self.s1
        B=self.s1
        pe=B/self.do
        pe_=1.27*pe*((L/B-0.3)**0.5)
        m=(2*self.a_air()/self.lamda_f/self.delta_f)**0.5
        hf=0.5*(self.do+2*self.delta_f)*(pe_-1)*(1+0.35*math.log(pe_))
        nc=1/self.sf+1/self.Le
        fb=math.pi*(self.do+2*self.delta_f)*(1-nc*self.delta_f)
        ff=nc*2*(self.s1*self.s2-math.pi/4*self.do**2)
        yita_f=math.tanh(m*hf)/(m*hf)
        return 1-ff/self.ft*(1-yita_f)
    def amendment(self,v):

            test_amendment = np.load("D:/model/fancoilunit/amendment.npy")
            return (test_amendment[0]) * v ** 3 + (test_amendment[1]) * v ** 2 + (test_amendment[2]) * v + (test_amendment[3])

    def approch(self):
        tw2 = self.tw1 + 0.01
        delat_Q_last = 10000000
        while True:
            water = physical_parameter_water(0.5 * (self.tw1 + tw2))
            air_in = physical_parameter_air2(self.t_in_dryair, self.t_in_relative_humidity)
            t1 = air_in.wd()
            h1 = air_in.bh()
            Q1 = self.W_water * water.brr() * (tw2 - self.tw1)
            h2 = air_in.bh() - Q1 / (air_in.md() * self.V_air)
            t_wallin = Q1 / (self.a_water(tw2) * self.Lt * self.fi) + 0.5 * (self.tw1 + tw2)

            t_wallout = (Q1 * math.log(self.do / self.di)) / (2 * self.lamda_t * self.Lt * math.pi) + t_wallin

            if t_wallout < air_in.ldwd():  # 湿工况
                dw = HAPropsSI('W', 'T', t_wallout + 273.15, 'P', 101325, 'R', 1)
                hw = HAPropsSI('H', 'T', t_wallout + 273.15, 'P', 101325, 'R', 1)
                d2 = air_in.hsl() - (air_in.hsl() - dw) * (air_in.bh() - h2) / (air_in.bh() - hw)
                #                t2=h2/(1.005+1.86*d2/1000)
                t2 = HAPropsSI('T', 'H', h2, 'P', 101325, 'W', d2) - 273.15
                kexi = (h1 - h2) / (1000 * (t1 - t2))  # 1000为空气密度
                Q2 = kexi * self.amendment(self.V_air) * self.Lt * self.ft * (0.5 * (t1 + t2) - t_wallout)
                # print(self.yita_s(),self.a_air(),)
                delat_Q = math.fabs(Q1 - Q2)
            else:  # 干工况
                d2 = air_in.hsl()
                #                t2=h2/(1.005+1.86*d2)
                t2 = HAPropsSI('T', 'H', h2, 'P', 101325, 'W', d2) - 273.15
                Q2 = self.amendment(self.V_air) * self.Lt * self.ft * (0.5 * (t1 + t2) - t_wallout)

                delat_Q = math.fabs(Q1 - Q2)
            if delat_Q - delat_Q_last > 0:
                # print(tw2, Q1, Q2, math.fabs(Q1 - Q2))
                break

            # # if math.fabs(Q1-Q2)<0.0010:
            # #     break
            else:
                tw2 = tw2 + 0.01
                delat_Q_last = delat_Q
            # tw2 = tw2 + 0.01
            # print(kexi,self.amendment(self.V_air),self.Lt*self.ft,t1,t2,t_wallout)
            # print(tw2,Q1,Q2,math.fabs(Q1-Q2))
        # Ppump = 0.015 * (self.W_water * 3600 / 12) ** 2 - 0.6 * (self.W_water * 3600 / 12) + 8.5
        # Pfan = 0.06 * (self.V_air * 3600 / 24) ** 2 - 3.5 * (self.V_air * 3600 / 24) + 65

        Pfan = (3.22086094e-05) * (self.V_air*3600) ** 2 + (3.05481795e-02) * self.V_air*3600 + (3.56900355e+01)  # 风流量-功率
        Ppump = (3.68380722e-04) * (self.W_water*3600) ** 2 + (-2.43551880e-02) * self.W_water*3600 + (2.48530002e+01)  # 水流量-功率
        reward = -math.fabs(self.cls-Q2)
        #        COP=Q2/(Ppump+Pfan)
        #return t2, d2, tw2, reward, Ppump, Pfan
        #return reward
        #return reward,-math.fabs(self.cls-Q2),Q2,t2, d2, tw2, reward, self.W_water*3600,self.V_air*3600,Ppump, Pfan
        return self.cls,Q2,t2, d2, tw2,self.W_water*3600,self.V_air*3600,Ppump, Pfan
def excel_one_line_to_list(i):
    df = pd.read_excel("D:/数据集/fancoli.xls", usecols=[i],names=None)  # 读取项目名称列,不要列名
    df_li = df.values.tolist()
    result = []
    for s_li in df_li:
        result.append(s_li[0])
    return result
def mae( action_predict, action_true):
    N = len(action_true)
    sum = 0
    for i in range(N):
        sum = sum + abs(action_true[i] - action_predict[i])
    MAE = sum / N
    print("评价指标MAE的值为：", str(MAE))

def mape(action_predict, action_true):
    sum = 0
    N = len(action_true)
    for i in range(N):
        sum = sum+(np.abs(action_true[i] - action_predict[i]) / action_true[i])
        # print(sum,(X1[i] - X2[i]) / X1[i],X1[i],X2[i])
    MAPE = sum/N*100

    print("评价指标MAPE的值为：", str(MAPE))
if __name__ == "__main__":
    v = excel_one_line_to_list(0)
    t1 = excel_one_line_to_list(1)
    ts1 = excel_one_line_to_list(2)
    t2 = excel_one_line_to_list(3)
    ts2 = excel_one_line_to_list(4)
    w = excel_one_line_to_list(5)
    tw1 = excel_one_line_to_list(6)
    tw2=excel_one_line_to_list(7)
    cls=excel_one_line_to_list(8)
    h2=excel_one_line_to_list(10)
    h_test=excel_one_line_to_list(19)
    data=np.load("state.npy")
#   aa={}
#   test=fan_coil_unit(tw1=6.99,V_air=556/3600,W_water=363.1/3600,t_in_dryair=28,t_in_relative_humidity=0.45)
#   test=fan_coil_unit(tw1=7.01,V_air=1010/3600,W_water=362/3600,t_in_dryair=28,t_in_relative_humidity=0.6)
#   test=fan_coil_unit(tw1=7.02,V_air=555/3600,W_water=575/3600,t_in_dryair=28.01,t_in_relative_humidity=0.6)
#   test=fan_coil_unit(tw1=6.99,V_air=556/3600,W_water=362/3600,t_in_dryair=27.99,t_in_relative_humidit                                          y=0.6)
#   test=fan_coil_unit(tw1=7,V_air=851/3600,W_water=360/3600,t_in_dryair=28,t_in_relative_humidity=0.6)
#   test=fan_coil_unit(tw1=7,V_air=693/3600,W_water=363/3600,t_in_dryair=28.05,t_in_relative_humidity=0.6)
#   test=fan_coil_unit(tw1=7.01,V_air=291/3600,W_water=361/3600,t_in_dryair=27.99,t_in_relative_humidity=0.6)
#   test=fan_coil_unit(tw1=7,V_air=376/3600,W_water=360/3600,t_in_dryair=28.01,t_in_relative_humidity=0.6)
#   test=fan_coil_unit(tw1=7.03,V_air=532/3600,W_water=362/3600,t_in_dryair=28,t_in_relative_humidity=0.6)
#   test=fan_coil_unit(tw1=7,V_air=693/3600,W_water=363/3600,t_in_dryair=28.05,t_in_relative_humidity=0.6)
#   test=fan_coil_unit(tw1=7,V_air=851/3600,W_water=360/3600,t_in_dryair=28,t_in_relative_humidity=0.6)
#   test=fan_coil_unit(tw1=7.01,V_air=1010/3600,W_water=362/3600,t_in_dryair=28,t_in_relative_humidity=0.6)
#    test = fan_coil_unit(cls=592,tw1=7, V_air=1010 / 3600, W_water=362 / 3600, t_in_dryair=28, t_in_relative_humidity=0.67878)
#    print(test.approch())
    for i in range(0,24):
        test = fan_coil_unit(cls=cls[i],tw1=tw1[i], V_air=v[i] / 3600, W_water=w[i] / 3600, t_in_dryair=t1[i],
                             t_in_relative_humidity=HAPropsSI('R', 'T', t1[i] + 273.15, 'P', 101325, 'B',
                                                              ts1[i] + 273.15))
        print(test.approch()[1])
        print(test.approch())


    # h_test = np.array(h_test)
    #mae(h2,h_test)
