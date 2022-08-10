import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
import xlrd
#水4空气3/4
book_DDPG = xlrd.open_workbook("../../DDPG_FCU/FCU_DDPG_5.xls")
book_MAC_DDPG = xlrd.open_workbook("../../DDPG_FCU_KN/FCU_DDPG_PK_5.xls")
book_bianlifa = xlrd.open_workbook("../../modelbasedcontrol.xls")
book_baseline = xlrd.open_workbook("../../baseline.xls")
sheet_DDPG = book_DDPG.sheet_by_name("a_air")
sheet_MAC_DDPG = book_MAC_DDPG.sheet_by_name("a_air")
sheet_bianlifa = book_bianlifa.sheet_by_name("a_air")
sheet_baseline = book_baseline.sheet_by_name("a_air")
DDPG_a_water_1= sheet_DDPG.col_values(0)
DDPG_MAC_a_water_1= sheet_MAC_DDPG.col_values(0)
bianlifa_a_water_1= sheet_bianlifa.col_values(0)
baseline_a_water_1= sheet_baseline.col_values(0)
DDPG_a_water_3= sheet_DDPG.col_values(2)
DDPG_MAC_a_water_3= sheet_MAC_DDPG.col_values(2)
bianlifa_a_water_3= sheet_bianlifa.col_values(2)
baseline_a_water_3= sheet_baseline.col_values(2)
DDPG_a_water_5= sheet_DDPG.col_values(4)
DDPG_MAC_a_water_5= sheet_MAC_DDPG.col_values(4)
bianlifa_a_water_5= sheet_bianlifa.col_values(4)
baseline_a_water_5= sheet_baseline.col_values(4)
DDPG_a_water_7= sheet_DDPG.col_values(6)
DDPG_MAC_a_water_7= sheet_MAC_DDPG.col_values(9)
bianlifa_a_water_7= sheet_bianlifa.col_values(6)
baseline_a_water_7= sheet_baseline.col_values(6)

a=[DDPG_a_water_1,DDPG_MAC_a_water_1,bianlifa_a_water_1,baseline_a_water_1,
   DDPG_a_water_3,DDPG_MAC_a_water_3,bianlifa_a_water_3,baseline_a_water_3,
   DDPG_a_water_5,DDPG_MAC_a_water_5,bianlifa_a_water_5,baseline_a_water_5,
   DDPG_a_water_7,DDPG_MAC_a_water_7,bianlifa_a_water_7,baseline_a_water_7]
for i in range(16):
    while "" in a[i]:# 判断是否有空值在列表中
        a[i].remove("")# 如果有就直接通过remove删除
fig, axes = plt.subplots(1, 3)

plt.subplots_adjust(wspace=0.25)#子图很有可能左右靠的很近，调整一下左右距离
fig.set_figwidth(16)#这个是设置整个图（画布）的大小
fig.set_figheight(5.4)#这个是设置整个图（画布）的大小


sns.kdeplot(baseline_a_water_1,ax=axes[0],shade=True,label="DDPG",legend=True)
sns.kdeplot(DDPG_a_water_1,ax=axes[0],shade=True,label="DDPG_PK",legend=True)
sns.kdeplot(bianlifa_a_water_1,ax=axes[0],shade=True,label="MBC",legend=True)
#sns.kdeplot(baseline_a_water_1,ax=axes[0],shade=True,label="RBC",legend=True)
axes[0].set_title("First Year")
axes[0].legend(loc="upper right")
#axes[0].tick_params(labelsize=fontsize_num)#刻度字体大小13

sns.kdeplot(baseline_a_water_3,ax=axes[1],shade=True,label="DDPG",legend=True)
sns.kdeplot(DDPG_a_water_5,ax=axes[1],shade=True,label="DDPG_PK",legend=True)
sns.kdeplot(bianlifa_a_water_3,ax=axes[1],shade=True,label="MBC",legend=True)
#sns.kdeplot(baseline_a_water_3,ax=axes[1],shade=True,label="RBC",legend=True)
axes[1].set_title("Third Year")
axes[1].set_ylabel("")
axes[1].legend(loc="upper right")


sns.kdeplot(baseline_a_water_5,ax=axes[2],shade=True,label="DDPG",legend=True)
sns.kdeplot(DDPG_a_water_5,ax=axes[2],shade=True,label="DDPG_PK",legend=True)
sns.kdeplot(bianlifa_a_water_5,ax=axes[2],shade=True,label="MBC",legend=True)
#sns.kdeplot(baseline_a_water_5,ax=axes[2],shade=True,label="RBC",legend=True)
axes[2].set_title("Fifth Year")
axes[2].set_ylabel("")
axes[2].legend(loc="upper right")

# sns.kdeplot(DDPG_a_water_7,ax=axes[3],shade=True,label="DDPG",legend=True)
# sns.kdeplot(DDPG_MAC_a_water_7,ax=axes[3],shade=True,label="DDPG_mac",legend=True)
# sns.kdeplot(bianlifa_a_water_7,ax=axes[3],shade=True,label="MBC",legend=True)
# #sns.kdeplot(baseline_a_water_7,ax=axes[3],shade=True,label="RBC",legend=True)
# axes[3].set_title("Seventh Year")
# axes[3].set_ylabel("")
# axes[3].legend(loc="upper right")
#plt.savefig("action_water.jpg",dpi=800)
plt.show()