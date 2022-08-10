import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
import xlrd
import pandas as pd

book_DDPG = xlrd.open_workbook("../../DDPG_FCU/FCU_DDPG_1.xls")
sheet_DDPG = book_DDPG.sheet_by_name("a_water")
DDPG_a_water_1= sheet_DDPG.col_values(0)
while "" in DDPG_a_water_1:# 判断是否有空值在列表中
    DDPG_a_water_1.remove("")# 如果有就直接通过remove删除

book_MAC_DDPG = xlrd.open_workbook("../../DDPG_FCU_KN/FCU_DDPG_PK_1.xls")
sheet_MAC_DDPG = book_MAC_DDPG.sheet_by_name("a_water")
DDPG_MAC_a_water_1= sheet_MAC_DDPG.col_values(0)
while "" in DDPG_MAC_a_water_1:# 判断是否有空值在列表中
    DDPG_MAC_a_water_1.remove("")# 如果有就直接通过remove删除

book_bianlifa = xlrd.open_workbook("../../modelbasedcontrol.xls")
sheet_bianlifa = book_bianlifa.sheet_by_name("a_water")
bianlifa_a_water_1= sheet_bianlifa.col_values(0)
while "" in bianlifa_a_water_1:# 判断是否有空值在列表中
    bianlifa_a_water_1.remove("")# 如果有就直接通过remove删除

book_baseline = xlrd.open_workbook("../../baseline.xls")
sheet_baseline = book_baseline.sheet_by_name("a_water")
baseline_a_water_1= sheet_baseline.col_values(0)
while "" in baseline_a_water_1:# 判断是否有空值在列表中
    baseline_a_water_1.remove("")# 如果有就直接通过remove删除

# for i in range(len(total_a_water_1)):
#     if total_a_water_1[i].astype(float)>0:
#         b.append(np.array(total_a_water_1[i]))
# print(b)
a1=sns.kdeplot(DDPG_a_water_1,DDPG_MAC_a_water_1,shade=True,legend=True,label="DDPG")
b1=sns.kdeplot(DDPG_MAC_a_water_1,baseline_a_water_1,shade=True,legend=True,label="e_DDPG")
# c1=sns.kdeplot(bianlifa_a_water_1,shade=True,legend=True,label="MBC")
# d1=sns.kdeplot(baseline_a_water_1,shade=True,legend=True,label="RBC")
a1.legend(loc="upper right")
b1.legend(loc="upper right")
# c1.legend(loc="upper right")
# d1.legend(loc="upper right")


plt.show()