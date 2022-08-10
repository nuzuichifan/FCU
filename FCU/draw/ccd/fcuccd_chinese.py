import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import xlrd
from matplotlib import pyplot as plt
from matplotlib import ticker
import pandas as pd
fig, ax = plt.subplots()
plt.rcParams['font.sans-serif']=['simsun']
plt.rcParams['axes.unicode_minus']=False

book_1 = xlrd.open_workbook("FCU__1.xls")
sheet_1 = book_1.sheet_by_name("sheet0")
bianlifa= sheet_1.col_values(0)
baseline= sheet_1.col_values(1)
DDPG_1= sheet_1.col_values(2)
DDPG_MAC_1= sheet_1.col_values(3)

book_2 = xlrd.open_workbook("FCU__2.xls")
sheet_2 = book_2.sheet_by_name("sheet0")
DDPG_2= sheet_2.col_values(2)
DDPG_MAC_2= sheet_2.col_values(3)

book_3 = xlrd.open_workbook("FCU__3.xls")
sheet_3 = book_3.sheet_by_name("sheet0")
DDPG_3= sheet_3.col_values(2)
DDPG_MAC_3= sheet_3.col_values(3)

book_4 = xlrd.open_workbook("FCU__4.xls")
sheet_4 = book_4.sheet_by_name("sheet0")
DDPG_4= sheet_4.col_values(2)
DDPG_MAC_4= sheet_4.col_values(3)

book_5 = xlrd.open_workbook("FCU__5.xls")
sheet_5 = book_5.sheet_by_name("sheet0")
DDPG_5= sheet_5.col_values(2)
DDPG_MAC_5= sheet_5.col_values(3)

fig.set_figwidth(6.5)#这个是设置整个图（画布）的大小
fig.set_figheight(5)#这个是设置整个图（画布）的大小
#matplotlib.rcParams['text.usetex'] = True  # 开启Latex风格
#plt.figure(figsize=(10, 10), dpi=70)  # 设置图像大小
#style.use('ggplot')  # 加载'ggplot'风格
# f, ax = plt.subplots(1, 3)  # 设置子图
X = ['1st','2nd','3th','4th','5th','6th','7th','8th','9th','10th','11st','12nd','13th','14th','15th','16th','17th','18th','19th','20th']

DDPG_NEW={}
DDPG_MAC_NEW={}
for i in range(20):
  DDPG_NEW[i]=(DDPG_1[i]+DDPG_2[i]+DDPG_3[i]+DDPG_4[i]+DDPG_5[i])/5
  DDPG_MAC_NEW[i]=(DDPG_MAC_1[i]+DDPG_MAC_2[i]+DDPG_MAC_3[i]+DDPG_MAC_4[i]+DDPG_MAC_5[i])/5
print(DDPG_NEW.values())
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
ax.plot(X,bianlifa ,label='MBC' )#s点的大小
ax.plot(X,baseline,label='RBC'  )
ax.plot(X,list(DDPG_NEW.values()),label='DDPG'  )
ax.plot(X,list(DDPG_MAC_NEW.values()),label='DDPG_PK'  )
# ax.xlabel('Number of training years')
plt.xlabel("训练的年数")
plt.ylabel("年均满足率")
ax.legend(loc="best")
fig.set_figwidth(7.5)#这个是设置整个图（画布）的大小
fig.set_figheight(4)#这个是设置整个图（画布）的大小
#plt.xlim(0, 10)


plt.savefig("Percentage of annual satisfied_chinses_1.jpg",dpi=600)
plt.show()