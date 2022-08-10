import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import xlrd
import pandas as pd
book_MAC_1 = xlrd.open_workbook("../../DDPG_FCU_moreACnet/FCU_DDPG_PK_1.xls")
sheet_MAC_1 = book_MAC_1.sheet_by_name("reward")
total_reward_MAC_1= sheet_MAC_1.row_values(4392)

book_MAC_2 = xlrd.open_workbook("../../DDPG_FCU_moreACnet/FCU_DDPG_PK_2.xls")
sheet_MAC_2 = book_MAC_2.sheet_by_name("reward")
total_reward_MAC_2= sheet_MAC_2.row_values(4392)

book_MAC_3 = xlrd.open_workbook("../../DDPG_FCU_moreACnet/FCU_DDPG_PK_3.xls")
sheet_MAC_3 = book_MAC_3.sheet_by_name("reward")
total_reward_MAC_3= sheet_MAC_3.row_values(4392)

book_MAC_4 = xlrd.open_workbook("../../DDPG_FCU_moreACnet/FCU_DDPG_PK_4.xls")
sheet_MAC_4 = book_MAC_4.sheet_by_name("reward")
total_reward_MAC_4= sheet_MAC_4.row_values(4392)

book_MAC_5 = xlrd.open_workbook("../../DDPG_FCU_moreACnet/FCU_DDPG_PK_5.xls")
sheet_MAC_5 = book_MAC_5.sheet_by_name("reward")
total_reward_MAC_5= sheet_MAC_5.row_values(4392)
#matplotlib.rcParams['text.usetex'] = True  # 开启Latex风格
#plt.figure(figsize=(10, 10), dpi=70)  # 设置图像大小
#style.use('ggplot')  # 加载'ggplot'风格
# f, ax = plt.subplots(1, 3)  # 设置子图
X = [1,2,3,4,5,6,7,8,9,10]
plt.plot(X,total_reward_MAC_1 ,label='1st year' )#s点的大小
plt.plot(X,total_reward_MAC_2,label='2nd year'  )
plt.plot(X,total_reward_MAC_3,label='3rd year'  )
plt.plot(X,total_reward_MAC_4,label='4th year'  )
plt.plot(X,total_reward_MAC_5,label='5th year'  )
plt.legend(loc="best")
#plt.xlim(0, 10)
plt.tight_layout()
plt.show()
#plt.savefig("reward_fiveyear_DDPG_moreACnet.jpg",dpi=600)
