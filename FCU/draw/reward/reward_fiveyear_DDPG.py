import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import xlrd
import pandas as pd
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
#matplotlib.rcParams['text.usetex'] = True  # 开启Latex风格
#plt.figure(figsize=(10, 10), dpi=70)  # 设置图像大小
#style.use('ggplot')  # 加载'ggplot'风格
# f, ax = plt.subplots(1, 3)  # 设置子图
X = [1,2,3,4,5,6,7,8,9,10]
plt.plot(X,total_reward_1 ,label='1st year' )#s点的大小
plt.plot(X,total_reward_2,label='2nd year'  )
plt.plot(X,total_reward_3,label='3rd year',color="black"  )
plt.plot(X,total_reward_4,label='4th year'  )
plt.plot(X,total_reward_5,label='5th year'  )
plt.legend(loc="best")
#plt.xlim(0, 10)
plt.show()
#plt.savefig("reward_fiveyear_DDPG.jpg",dpi=600)
