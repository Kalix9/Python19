import matplotlib.pyplot as plt
#matplotlib inline
import numpy as np
import pandas as pd
from scipy import linalg as la
import sympy
sympy.init_printing()
from scipy import optimize
data= pd.read_csv('C:/Users/purpl/Documents/研二第一学期/python数据运算与数据可视化/WorldIndex.csv')
datanew=data.dropna()#删除缺失值
print(datanew)

#绘制人均寿命数据的直方图

max1=max(datanew.Life_expectancy)
print(max1)

min1=min(datanew.Life_expectancy)
print(min1)

range1=max1-min1
print(range1)

plt.hist(x = datanew.Life_expectancy, # 指定绘图数据
bins = 7, # 指定直方图中条块的个数
color = 'blue', # 指定直方图的填充色
edgecolor = 'black')
plt.xlabel('Age')
plt.ylabel('Number')

#绘制人均GDP直方图
max2=max(datanew.GDP_per_capita)
print(max2)


min2=min(datanew.GDP_per_capita)
print(min2)

range2=max2-min2
print(range2)


plt.hist(x = datanew.GDP_per_capita, # 指定绘图数据
bins = 30, # 指定直方图中条块的个数
color = 'blue', # 指定直方图的填充色
edgecolor = 'black')
plt.xlabel('GDP per capita')
plt.ylabel('Number of countries')
plt.title('National GDP per capita')

#绘制人均寿命箱线图
import seaborn as sns
plt.figure(figsize=(15,7))
sns.boxplot(data=datanew.Life_expectancy,orient='r')
plt.ylabel('Age')
plt.show()

#绘制每个大洲的国家个数条形图
con_number = datanew.Continent.value_counts()
print(con_number)
con_name = list(con_number.index)
print(con_name)
con_arange = np.arange(len(con_name))
print(con_arange)


plt.bar(con_arange, con_number)
# 设置横坐标
plt.xticks(con_arange, con_name, rotation=70)   # rotation 旋转横坐标标签
plt.show()

#绘制各大洲国家数量占比的饼图

plt.pie(con_number, labels=con_name, autopct='%1.f%%')  # autopct 显示占比
plt.axis('equal')  # 调整坐标轴的比例
plt.show()


#绘制人均寿命和人均GDP的关系（散点）
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.scatter(datanew.GDP_per_capita, datanew.Life_expectancy)

plt.xlabel('GDP per capita')
plt.ylabel('Number of countries')
plt.title('National GDP per capita')
plt.title('The relationship between the health and economic level of countries')  # 图标题
plt.show()

#绘制人均寿命和人均GDP的关系（散点）
map_dict = {
    'Asia':'red',
    'Europe':'yellow',
    'Africa':'blue',
    'North America':'green',
    'South America':'brown',
    'Oceania':'black'
}
colors = datanew.Continent.map(map_dict)   # 将国家按所在州对于不同的颜色

size = datanew.Population/ 1e6 * 2  # 数据点大小，正比于人口数
plt.scatter(x=datanew.GDP_per_capita, y=datanew.Life_expectancy, s=size,c=colors, alpha=0.5)  # 参数s设置点的大小
plt.xscale('log')
plt.xlabel('GDP per capita')
plt.ylabel('Life expectancy')
plt.title('Global health and income levels')

tick_val = [1000,10000,100000]
tick_lab = ['1k','10k','100k']
plt.xticks(tick_val, tick_lab)
plt.show()

import sympy
from sympy import *

x = sympy.Symbol("x")
A = 2
B = 1
b=sympy.nsolve(A * sympy.sin(x) - x+B,0)
print(b)

fig, ax = plt.subplots(figsize=(8, 4))
x1 = np.linspace(-2, 2, 100)

x2_1 = (3 - 3 * x1)/2
x2_2 = (x1-5)/2

ax.plot(x1, x2_1, 'r', lw=2, label=r"$3x_1+2x_2-3=0$")
ax.plot(x1, x2_2, 'g', lw=2, label=r"$x_1-2x_2-5=0$")

A = np.array([[3, 2], [1, -2]])
b = np.array([3, 5])
x = la.solve(A, b)

ax.plot(x[0], x[1], 'ko', lw=2)
ax.annotate("The intersection point of\nthe two lines is the solution\nto the equation system",
            xy=(x[0], x[1]), xycoords='data',
            xytext=(-120, -75), textcoords='offset points',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-.3"))

ax.set_xlabel(r"$x_1$", fontsize=18)
ax.set_ylabel(r"$x_2$", fontsize=18)
ax.legend();

fig.tight_layout()

#Symbolic approach
A = sympy.Matrix([[3, 2], [1, -2]])
b = sympy.Matrix([3, 5])

print(A.rank())
print(A.condition_number())
sympy.N(1)
print(A.norm())
L, U, _ = A.LUdecomposition()
print(L)
print(U)
print(L * U)

x = A.solve(b)
print(x)


###Numerical approach

A = np.array([[3, 2], [1, -2]])
b = np.array([3, 5])


print(np.linalg.matrix_rank(A))
print(np.linalg.cond(A))
print(np.linalg.norm(A))
P, L, U = la.lu(A)
print(L)
print(U)
print(L*U)
print(la.solve(A, b))


