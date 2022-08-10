import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import xlrd
import pandas as pd
f, ax = plt.subplots(3, 1)
book_1 = xlrd.open_workbook("../../DDPG_FCU/FCU_DDPG_1.xls")
sheet_1 = book_1.sheet_by_name("reward")
total_reward_1= sheet_1.row_values(4392)

book_2 = xlrd.open_workbook("../../DDPG_FCU/FCU_DDPG_2.xls")
sheet_2 = book_2.sheet_by_name("reward")
total_reward_2= sheet_2.row_values(4392)

book_3 = xlrd.open_workbook("../../DDPG_FCU/FCU_DDPG_3.xls")
sheet_3 = book_3.sheet_by_name("reward")
total_reward_3= sheet_3.row_values(4392)

book_4 = xlrd.open_workbook("../../DDPG_FCU/FCU_DDPG_4.xls")
sheet_4 = book_4.sheet_by_name("reward")
total_reward_4= sheet_4.row_values(4392)

book_5 = xlrd.open_workbook("../../DDPG_FCU/FCU_DDPG_5.xls")
sheet_5 = book_5.sheet_by_name("reward")
total_reward_5= sheet_5.row_values(4392)

book_MAC_1 = xlrd.open_workbook("../../DDPG_FCU_KN/FCU_DDPG_PK_1.xls")
sheet_MAC_1 = book_MAC_1.sheet_by_name("reward")
total_reward_MAC_1= sheet_MAC_1.row_values(4392)

book_MAC_2 = xlrd.open_workbook("../../DDPG_FCU_KN/FCU_DDPG_PK_2.xls")
sheet_MAC_2 = book_MAC_2.sheet_by_name("reward")
total_reward_MAC_2= sheet_MAC_2.row_values(4392)

book_MAC_3 = xlrd.open_workbook("../../DDPG_FCU_KN/FCU_DDPG_PK_3.xls")
sheet_MAC_3 = book_MAC_3.sheet_by_name("reward")
total_reward_MAC_3= sheet_MAC_3.row_values(4392)

book_MAC_4 = xlrd.open_workbook("../../DDPG_FCU_KN/FCU_DDPG_PK_4.xls")
sheet_MAC_4 = book_MAC_4.sheet_by_name("reward")
total_reward_MAC_4= sheet_MAC_4.row_values(4392)

book_MAC_5 = xlrd.open_workbook("../../DDPG_FCU_KN/FCU_DDPG_PK_5.xls")
sheet_MAC_5 = book_MAC_5.sheet_by_name("reward")
total_reward_MAC_5= sheet_MAC_5.row_values(4392)
#matplotlib.rcParams['text.usetex'] = True  # 开启Latex风格
#plt.figure(figsize=(10, 10), dpi=70)  # 设置图像大小
#style.use('ggplot')  # 加载'ggplot'风格
# f, ax = plt.subplots(1, 3)  # 设置子图
# plt.subplots_adjust(wspace=0.25)#子图很有可能左右靠的很近，调整一下左右距离

X = ['1st','2nd','3th','4th','5th','6th','7th','8th','9th','10th','11st','12nd','13th','14th','15th','16th','17th','18th','19th','20th']
def fig0():
    numoral_DDPG_reward={}
    moreac_DDPG_reward={}
    for i in range(0,20):
        numoral_DDPG_reward[i]=(total_reward_1[i]+total_reward_2[i]+total_reward_3[i]+total_reward_4[i]+total_reward_5[i])/5
        moreac_DDPG_reward[i]=(total_reward_MAC_1[i]+total_reward_MAC_2[i]+total_reward_MAC_3[i]+total_reward_MAC_4[i]+total_reward_MAC_5[i])/5
    ax[0].plot(X,list(numoral_DDPG_reward.values()) ,label='DDPG' )#s点的大小
    ax[0].plot(X,list(moreac_DDPG_reward.values()),label='DDPG with PK'  )
    ax[0].legend(loc="best")
    #ax[0].set_xlabel('Number of training years')
    ax[0].set_ylabel('Reward')
    ax[0].set_title('(a) Average cumulative reward')
def fig1():#ddpg
    ax[1].plot(X, total_reward_1, label='round1')  # s点的大小
    ax[1].plot(X, total_reward_2, label='round2')
    ax[1].plot(X, total_reward_3, label='round3')
    ax[1].plot(X, total_reward_4, label='round4')
    ax[1].plot(X, total_reward_5, label='round5')
    ax[1].legend(loc="best")
    #ax[1].set_xlabel('Number of training years')
    ax[1].set_ylabel('Reward')
    ax[1].set_title('(b) Five independent experiments of DDPG')
def fig2():#ddpg with split action space
    ax[2].plot(X, total_reward_MAC_1, label='round1')  # s点的大小
    ax[2].plot(X, total_reward_MAC_2, label='round2')
    ax[2].plot(X, total_reward_MAC_3, label='round3')
    ax[2].plot(X, total_reward_MAC_4, label='round4')
    ax[2].plot(X, total_reward_MAC_5, label='round5')
    ax[2].legend(loc="best")
    #ax[2].set_xlabel('Number of training years')
    ax[2].set_ylabel('Reward')
    ax[2].set_title('(c) Five independent experiments of DDPG with PK')
#plt.title('Average cumulative rewards for five experiments ')
fig0()
fig1()
fig2()

f.set_figwidth(9)#这个是设置整个图（画布）的大小
f.set_figheight(7)#这个是设置整个图（画布）的大小
#plt.xlim(0, 10)
plt.xlabel('Number of training years')
plt.tight_layout()
#plt.savefig("Average cumulative rewards for five experiments.jpg",dpi=800)
plt.show()
