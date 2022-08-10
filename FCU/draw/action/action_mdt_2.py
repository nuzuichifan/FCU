import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
import xlrd
#水4空气3/4
book_DDPG = xlrd.open_workbook("../../DDPG_FCU/FCU_DDPG_1.xls")
book_MAC_DDPG = xlrd.open_workbook("../../DDPG_FCU_KN/FCU_DDPG_PK_1.xls")
book_bianlifa = xlrd.open_workbook("../../modelbasedcontrol.xls")
book_baseline = xlrd.open_workbook("../../baseline.xls")
sheet_DDPG_wate = book_DDPG.sheet_by_name("a_water")
sheet_MAC_DDPG_water = book_MAC_DDPG.sheet_by_name("a_water")
sheet_bianlifa_water = book_bianlifa.sheet_by_name("a_water")
sheet_baseline_water = book_baseline.sheet_by_name("a_water")
DDPG_a_water_1= sheet_DDPG_wate.col_values(0)
DDPG_MAC_a_water_1= sheet_MAC_DDPG_water.col_values(0)
bianlifa_a_water_1= sheet_bianlifa_water.col_values(0)
baseline_a_water_1= sheet_baseline_water.col_values(0)
DDPG_a_water_3= sheet_DDPG_wate.col_values(2)
DDPG_MAC_a_water_3= sheet_MAC_DDPG_water.col_values(2)
bianlifa_a_water_3= sheet_bianlifa_water.col_values(2)
baseline_a_water_3= sheet_baseline_water.col_values(2)
DDPG_a_water_5= sheet_DDPG_wate.col_values(4)
DDPG_MAC_a_water_5= sheet_MAC_DDPG_water.col_values(4)
bianlifa_a_water_5= sheet_bianlifa_water.col_values(4)
baseline_a_water_5= sheet_baseline_water.col_values(4)
DDPG_a_water_7= sheet_DDPG_wate.col_values(6)
DDPG_MAC_a_water_7= sheet_MAC_DDPG_water.col_values(6)
bianlifa_a_water_7= sheet_bianlifa_water.col_values(6)
baseline_a_water_7= sheet_baseline_water.col_values(6)
DDPG_a_water_9= sheet_DDPG_wate.col_values(8)
DDPG_MAC_a_water_9= sheet_MAC_DDPG_water.col_values(8)
bianlifa_a_water_9= sheet_bianlifa_water.col_values(8)
baseline_a_water_9= sheet_baseline_water.col_values(8)
DDPG_a_water_11= sheet_DDPG_wate.col_values(10)
DDPG_MAC_a_water_11= sheet_MAC_DDPG_water.col_values(10)
bianlifa_a_water_11= sheet_bianlifa_water.col_values(10)
baseline_a_water_11= sheet_baseline_water.col_values(10)
DDPG_a_water_13= sheet_DDPG_wate.col_values(12)
DDPG_MAC_a_water_13= sheet_MAC_DDPG_water.col_values(12)
bianlifa_a_water_13= sheet_bianlifa_water.col_values(12)
baseline_a_water_13= sheet_baseline_water.col_values(12)
DDPG_a_water_15= sheet_DDPG_wate.col_values(14)
DDPG_MAC_a_water_15= sheet_MAC_DDPG_water.col_values(14)
bianlifa_a_water_15= sheet_bianlifa_water.col_values(14)
baseline_a_water_15= sheet_baseline_water.col_values(14)
DDPG_a_water_17= sheet_DDPG_wate.col_values(16)
DDPG_MAC_a_water_17= sheet_MAC_DDPG_water.col_values(16)
bianlifa_a_water_17= sheet_bianlifa_water.col_values(16)
baseline_a_water_17= sheet_baseline_water.col_values(16)
DDPG_a_water_19= sheet_DDPG_wate.col_values(18)
DDPG_MAC_a_water_19= sheet_MAC_DDPG_water.col_values(18)
bianlifa_a_water_19= sheet_bianlifa_water.col_values(18)
baseline_a_water_19= sheet_baseline_water.col_values(18)

sheet_DDPG_air = book_DDPG.sheet_by_name("a_air")
sheet_MAC_DDPG_air = book_MAC_DDPG.sheet_by_name("a_air")
sheet_bianlifa_air = book_bianlifa.sheet_by_name("a_air")
sheet_baseline_air = book_baseline.sheet_by_name("a_air")
DDPG_a_air_1= sheet_DDPG_air.col_values(0)
DDPG_MAC_a_air_1= sheet_MAC_DDPG_air.col_values(0)
bianlifa_a_air_1= sheet_bianlifa_air.col_values(0)
baseline_a_air_1= sheet_baseline_air.col_values(0)
DDPG_a_air_3= sheet_DDPG_air.col_values(2)
DDPG_MAC_a_air_3= sheet_MAC_DDPG_air.col_values(2)
bianlifa_a_air_3= sheet_bianlifa_air.col_values(2)
baseline_a_air_3= sheet_baseline_air.col_values(2)
DDPG_a_air_5= sheet_DDPG_air.col_values(4)
DDPG_MAC_a_air_5= sheet_MAC_DDPG_air.col_values(4)
bianlifa_a_air_5= sheet_bianlifa_air.col_values(4)
baseline_a_air_5= sheet_baseline_air.col_values(4)
DDPG_a_air_7= sheet_DDPG_air.col_values(6)
DDPG_MAC_a_air_7= sheet_MAC_DDPG_air.col_values(6)
bianlifa_a_air_7= sheet_bianlifa_air.col_values(6)
baseline_a_air_7= sheet_baseline_air.col_values(6)
DDPG_a_air_9= sheet_DDPG_air.col_values(8)
DDPG_MAC_a_air_9= sheet_MAC_DDPG_air.col_values(8)
bianlifa_a_air_9= sheet_bianlifa_air.col_values(8)
baseline_a_air_9= sheet_baseline_air.col_values(8)
DDPG_a_air_11= sheet_DDPG_air.col_values(10)
DDPG_MAC_a_air_11= sheet_MAC_DDPG_air.col_values(10)
bianlifa_a_air_11= sheet_bianlifa_air.col_values(10)
baseline_a_air_11= sheet_baseline_air.col_values(10)
DDPG_a_air_13= sheet_DDPG_air.col_values(12)
DDPG_MAC_a_air_13= sheet_MAC_DDPG_air.col_values(12)
bianlifa_a_air_13= sheet_bianlifa_air.col_values(12)
baseline_a_air_13= sheet_baseline_air.col_values(12)
DDPG_a_air_15= sheet_DDPG_air.col_values(14)
DDPG_MAC_a_air_15= sheet_MAC_DDPG_air.col_values(14)
bianlifa_a_air_15= sheet_bianlifa_air.col_values(14)
baseline_a_air_15= sheet_baseline_air.col_values(14)
DDPG_a_air_17= sheet_DDPG_air.col_values(16)
DDPG_MAC_a_air_17= sheet_MAC_DDPG_air.col_values(16)
bianlifa_a_air_17= sheet_bianlifa_air.col_values(16)
baseline_a_air_17= sheet_baseline_air.col_values(16)
DDPG_a_air_19= sheet_DDPG_air.col_values(18)
DDPG_MAC_a_air_19= sheet_MAC_DDPG_air.col_values(18)
bianlifa_a_air_19= sheet_bianlifa_air.col_values(18)
baseline_a_air_19= sheet_baseline_air.col_values(18)

a=[DDPG_a_water_1,DDPG_MAC_a_water_1,bianlifa_a_water_1,baseline_a_water_1,
   DDPG_a_water_3,DDPG_MAC_a_water_3,bianlifa_a_water_3,baseline_a_water_3,
   DDPG_a_water_5,DDPG_MAC_a_water_5,bianlifa_a_water_5,baseline_a_water_5,
   DDPG_a_water_7,DDPG_MAC_a_water_7,bianlifa_a_water_7,baseline_a_water_7,
   DDPG_a_water_9, DDPG_MAC_a_water_9, bianlifa_a_water_9, baseline_a_water_9,
DDPG_a_water_11,DDPG_MAC_a_water_11,bianlifa_a_water_11,baseline_a_water_11,
   DDPG_a_water_13,DDPG_MAC_a_water_13,bianlifa_a_water_13,baseline_a_water_13,
   DDPG_a_water_15,DDPG_MAC_a_water_15,bianlifa_a_water_15,baseline_a_water_15,
   DDPG_a_water_17,DDPG_MAC_a_water_17,bianlifa_a_water_17,baseline_a_water_17,
   DDPG_a_water_19, DDPG_MAC_a_water_19, bianlifa_a_water_19, baseline_a_water_19
   ]
b=[DDPG_a_air_1,DDPG_MAC_a_air_1,bianlifa_a_air_1,baseline_a_air_1,
   DDPG_a_air_3,DDPG_MAC_a_air_3,bianlifa_a_air_3,baseline_a_air_3,
   DDPG_a_air_5,DDPG_MAC_a_air_5,bianlifa_a_air_5,baseline_a_air_5,
   DDPG_a_air_7,DDPG_MAC_a_air_7,bianlifa_a_air_7,baseline_a_air_7,
   DDPG_a_air_9,DDPG_MAC_a_air_9,bianlifa_a_air_9,baseline_a_air_9,
DDPG_a_air_11,DDPG_MAC_a_air_11,bianlifa_a_air_11,baseline_a_air_11,
   DDPG_a_air_13,DDPG_MAC_a_air_13,bianlifa_a_air_13,baseline_a_air_13,
   DDPG_a_air_15,DDPG_MAC_a_air_15,bianlifa_a_air_15,baseline_a_air_15,
   DDPG_a_air_17,DDPG_MAC_a_air_17,bianlifa_a_air_17,baseline_a_air_17,
   DDPG_a_air_19,DDPG_MAC_a_air_19,bianlifa_a_air_19,baseline_a_air_19
   ]
for i in range(40):
    while "" in a[i]:# 判断是否有空值在列表中
        a[i].remove("")# 如果有就直接通过remove删除

for i in range(40):
    while "" in b[i]:# 判断是否有空值在列表中
        b[i].remove("")# 如果有就直接通过remove删除
fig, axes = plt.subplots(2,5)

#plt.subplots_adjust(wspace=0.25)#子图很有可能左右靠的很近，调整一下左右距离
# fig.set_figwidth(6.5)#这个是设置整个图（画布）的大小
# fig.set_figheight(7.5)#这个是设置整个图（画布）的大小



sns.kdeplot(x=DDPG_MAC_a_water_1,y=DDPG_MAC_a_air_1,ax=axes[0,0],shade=True,label="DDPG_PK",)
sns.kdeplot(x=bianlifa_a_water_1,y=bianlifa_a_air_1,ax=axes[0,0],shade=True,label="MBC",)

axes[0,0].set_title("1 Year")
axes[0,0].axis=([100,700,0,1200])
sns.kdeplot(x=DDPG_MAC_a_water_3,y=DDPG_MAC_a_air_3,ax=axes[0,1],shade=True,label="DDPG_PK",)
sns.kdeplot(x=bianlifa_a_water_3,y=bianlifa_a_air_3,ax=axes[0,1],shade=True,label="MBC",)
axes[0,1].set_title("3 Year")
axes[0,1].axis=([100,700,0,1200])
sns.kdeplot(x=DDPG_MAC_a_water_5,y=DDPG_MAC_a_air_5,ax=axes[0,2],shade=True,label="DDPG_PK",)
sns.kdeplot(x=bianlifa_a_water_5,y=bianlifa_a_air_5,ax=axes[0,2],shade=True,label="MBC",)
axes[0,2].set_title("5 Year")
axes[0,2].axis=([100,700,0,1200])
sns.kdeplot(x=DDPG_MAC_a_water_7,y=DDPG_MAC_a_air_7,ax=axes[0,3],shade=True,label="DDPG_PK")
sns.kdeplot(x=bianlifa_a_water_7,y=bianlifa_a_air_7,ax=axes[0,3],shade=True,label="MBC")
axes[0,3].set_title("7 Year")
axes[0,3].axis=([100,700,0,1200])
sns.kdeplot(x=DDPG_MAC_a_water_9,y=DDPG_MAC_a_air_9,ax=axes[0,4],shade=True,label="DDPG_PK")
sns.kdeplot(x=bianlifa_a_water_9,y=bianlifa_a_air_9,ax=axes[0,4],shade=True,label="MBC")
axes[0,4].set_title("9 Year")
axes[0,4].axis=([100,700,0,1200])



sns.kdeplot(x=DDPG_MAC_a_water_11,y=DDPG_MAC_a_air_11,ax=axes[1,0],shade=True,label="DDPG_PK",)
sns.kdeplot(x=bianlifa_a_water_11,y=bianlifa_a_air_11,ax=axes[1,0],shade=True,label="MBC",)
axes[1,0].set_title("11 Year")
axes[1,0].axis=([100,700,0,1200])
sns.kdeplot(x=DDPG_MAC_a_water_13,y=DDPG_MAC_a_air_13,ax=axes[1,1],shade=True,label="DDPG_PK",)
sns.kdeplot(x=bianlifa_a_water_13,y=bianlifa_a_air_13,ax=axes[1,1],shade=True,label="MBC",)
axes[1,1].set_title("13 Year")
axes[1,1].axis=([100,700,0,1200])
sns.kdeplot(x=DDPG_MAC_a_water_15,y=DDPG_MAC_a_air_15,ax=axes[1,2],shade=True,label="DDPG_PK",)
sns.kdeplot(x=bianlifa_a_water_15,y=bianlifa_a_air_15,ax=axes[1,2],shade=True,label="MBC",)
axes[1,2].set_title("15 Year")
axes[1,2].axis=([100,700,0,1200])
sns.kdeplot(x=DDPG_MAC_a_water_17,y=DDPG_MAC_a_air_17,ax=axes[1,3],shade=True,label="DDPG_PK")
sns.kdeplot(x=bianlifa_a_water_17,y=bianlifa_a_air_17,ax=axes[1,3],shade=True,label="MBC")
axes[1,3].set_title("17 Year")
axes[1,3].axis=([100,700,0,1200])
sns.kdeplot(x=DDPG_MAC_a_water_19,y=DDPG_MAC_a_air_19,ax=axes[1,4],shade=True,label="DDPG_PK")
sns.kdeplot(x=bianlifa_a_water_19,y=bianlifa_a_air_19,ax=axes[1,4],shade=True,label="MBC")
axes[1,4].set_title("19 Year")
axes[1,4].axis=([100,700,0,1200])




#axes[0,3].set_xlabel("Water flow")
# axes[1,1].legend(loc="upper right")
# lines = []
# labels = ['DDPG_MAC', 'MBC']
#
# fig.legend(labels, loc='lower right', bbox_to_anchor=(1,-0.1), ncol=len(labels), bbox_transform=fig.transFigure)

# fig.legend(lines, labels,
#            loc='upper right')


# sns.kdeplot(DDPG_MAC_a_water_9,DDPG_MAC_a_air_9,ax=axes[4],shade=True,label="DDPG_mac",legend=True)
# sns.kdeplot(bianlifa_a_water_9,bianlifa_a_air_9,ax=axes[4],shade=True,label="MBC",legend=True)
# #sns.kdeplot(baseline_a_water_5,ax=axes[2],shade=True,label="RBC",legend=True)
# axes[4].set_title("ninth Year")
# axes[4].set_ylabel("")
# axes[4].legend(loc="upper right")

# sns.kdeplot(DDPG_a_water_7,ax=axes[3],shade=True,label="DDPG",legend=True)
# sns.kdeplot(DDPG_MAC_a_water_7,ax=axes[3],shade=True,label="DDPG_mac",legend=True)
# sns.kdeplot(bianlifa_a_water_7,ax=axes[3],shade=True,label="MBC",legend=True)
# #sns.kdeplot(baseline_a_water_7,ax=axes[3],shade=True,label="RBC",legend=True)
# axes[3].set_title("Seventh Year")
# axes[3].set_ylabel("")
# axes[3].legend(loc="upper right")

#plt.savefig("action1.jpg",dpi=800)
plt.show()