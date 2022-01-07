# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 22:14:30 2021

@author: Winner
"""

import numpy as np
import os
from numpy.random import randn
import random as rd
import matplotlib.pyplot as plt


def Specdensity(ww,omegag,zetag,omegac,S0):
    Sf_0=S0*(np.power(omegag,4)+4*zetag**2*omegag**2*ww**2)/((omegag**2-ww**2)**2+4*zetag**2*omegag**2*ww**2) #金井清谱
    # Sf=Sf_0.*ww.^4/((omegaf.^2-ww.^2).^2+4*zetaf.^2*omegaf.^2.*ww.^2); #C-P谱
    return Sf_0*(ww**2/(ww**2+omegac**2)) #改进金井清谱


def funcA(omega, dt, a_th):
    # 选择参数β和γ，并计算积分常数
    gama = 0.5
    beta = 0.25
    para = []
    para.append(1 / beta / dt ** 2)
    para.append(gama / beta / dt)
    para.append(1 / beta / dt)
    para.append(1 / 2 / beta - 1)
    para.append(gama / beta - 1)
    para.append(dt / 2 * (gama / beta - 2))
    para.append(dt * (1 - gama))
    para.append(gama * dt)
    para=np.array(para)
    # 形成刚度矩阵、质量矩阵和阻尼矩阵，这里是单自由度，假设M=1
    M = 1
    K = omega ** 2 * M
    C = 2 * omega * dr * M
    LT=len(omega)#反应谱横坐标点数
    nt = len(a_th)#地震总点数
    P = np.zeros(shape=(LT,nt+1))
    P[:,:-1] =(-a_th).reshape(1,len(a_th)).repeat(LT,axis=0)
    # 存储各时间点的位移、速度以及加速度
    u = np.zeros(shape=(LT,nt+1))
    v = np.zeros(shape=(LT,nt+1))
    a = np.zeros(shape=(LT,nt+1))
    Aa = np.zeros(shape=(LT,nt))
    # 形成等效刚度矩阵K_
    K_ = K + para[0] * M + para[1] * C
    # 递推求解
    for i in range(nt):
        # 计算ti+1时刻的等效荷载
        P_ = P[:,i] + M * (para[0] * u[:,i] + para[2] * v[:,i] + para[3] * a[:,i]) + C * (
                para[1] * u[:,i] + para[4] * v[:,i] + para[5] * a[:,i])
        # 计算ti+1时刻的位移
        u[:,i+1]=(P_ / K_)
        # 计算ti+1时刻的加速度和速度
        a[:,i+1]=para[0] * (u[:,i+1] - u[:,i]) - para[2] * v[:,i] - para[3] * a[:,i]
        v[:,i+1]=v[:,i] + para[6] * a[:,i] + para[7] * a[:,i+1]
        ##
        Aa[:,i]=a[:,i + 1] + a_th[i]
    return (1/9.81)*np.max(np.abs(Aa),axis=1)

def code_rsa(omega,dr,alpha_max):
    
    '''
    建筑抗震设计规范反应谱
    '''
    gamma=0.9+(0.05-dr)/(0.3+6*dr)
    eta1=0.02+(0.05-dr)/(4+32*dr)
    eta2=1+(0.05-dr)/(0.08+1.6*dr)
    TT=2*np.pi/omega
    # first curve:
    if TT>0 and TT<=0.1:
        alpha=10*(eta2-0.45)*alpha_max*TT+0.45*alpha_max
    # second curve
    if TT>0.1 and TT<=Tg:
        alpha=eta2*alpha_max
    # third curve
    if TT>Tg and TT<=5*Tg:
        alpha=eta2*alpha_max*np.power(Tg/TT,gamma)
    # fourth curve
    if TT>5*Tg:
        alpha=(eta2*np.power(0.2,gamma)-eta1*(TT-5*Tg))*alpha_max
    
    return alpha
    # return alpha
    
# 基于双随机变量谐和函数叠加模拟非平稳人工地震波
N0=200            # 积分点个数
sample=3000     # 样本个数
dr = 0.05      #阻尼比

Apga=0.31*9.81  # m/s2
# Apga=0.31  # m/s2
alpha_max=0.72  #对应着地震加速度时程峰值为alpha_max*Apga/2.25

omegag=17.95  #Tg=0.35 s
Tg=2*np.pi/omegag
zetag=0.72     #场地土阻尼比，二类场地取0.72
gamma=2.83     #峰值因子，二类场地一、二组取2.83
S0=Apga**2/gamma**2/((np.pi*omegag*(2*zetag+1/(2*zetag)))) #基岩白噪声的谱密度，反映地震动强弱
print('S0=',S0)

# omegaf=0.5
omegac=0.5 #控制地面运动低频含量参数。该值取越大，地面运动低频含量越小
 
# zetaf=0.6
# S0=0.0049
dt=0.01           # 时间步长/秒
T=20              # 地震持续时间/秒
t=np.linspace(dt,T, int(T/dt))
#In spectrum Tend=6
# Tend=5
wu=200            # 截止频率
dw=wu/N0

listk=np.arange(N0)
ww=dw+listk*dw

#时-频调制函数参数值
a=0.25
b=0.251
c=0.005

t_x=(np.log(c*np.abs(ww-omegag)+b)-np.log(a))/(c*np.abs(ww-omegag)+b-a)
# gt=12.21*(np.exp(-0.4*t)-np.exp(-0.5*t)) # 均匀调制函数

# Ag1=np.zeros((len(t),len(ww)))

# S0=0.0049

random=np.random.RandomState(0)
theda1=np.random.uniform(0,2*np.pi,sample)
theda2=np.random.uniform(0,2*np.pi,sample)

# k=np.linspace(1,N0,N0)

Xk0=np.zeros(N0)
Yk0=np.zeros(N0)
Xk=np.zeros((sample,N0))
Yk=np.zeros((sample,N0))
for j in range(sample):
    for k in range (N0):
         Xk0[k]=np.cos((k+1)*theda1[j])+np.sin((k+1)*theda1[j])
         Yk0[k]=np.cos((k+1)*theda2[j])+np.sin((k+1)*theda2[j])
    rd.shuffle(Xk0)
    rd.shuffle(Yk0)
    Xk[j,:]=Xk0[:] 
    Yk[j,:]=Yk0[:]


#获得目标谱
St_=np.zeros(len(ww))
for ii in range (len(ww)):
    St_[ii]=code_rsa(dw*(ii+1),dr,alpha_max)

Ag2=np.zeros(len(t))
Ag=np.zeros((sample,len(t)))
Ap=np.zeros((sample,len(t)))
kk=np.arange(N0)
S_w=Specdensity(dw*(kk+1),omegag,zetag,omegac,S0)
Sa=np.zeros((sample,len(ww)))

for mk in range(30):
    print('------------第'+str(mk+1)+'次迭代-----------')
    for j in range(sample):
        for i in range(len(t)):
            gt=np.abs((np.exp(-a*(i+1)*dt)-np.exp(-(c*np.abs(ww-omegag)+b)*(i+1)*dt))/(np.exp(-a*t_x)-np.exp(-(c*np.abs(ww-omegag)+b)*t_x)))
            gt2=gt*gt
            AA=np.zeros(N0)
            # AA=np.sqrt(2*Specdensity(dw*kk,omegag,zetag,omegac,S0)*dw)*(np.cos(dw*kk*i*dt)*Xk[j,kk]+np.sin(dw*kk*i*dt)*Yk[j,kk])
            # Ag2[i]=sum(AA)
            AA=np.sqrt(2*gt2*S_w*dw)*(np.cos(dw*(kk+1)*(i+1)*dt)*Xk[j,kk]+np.sin(dw*(kk+1)*(i+1)*dt)*Yk[j,kk])
            Ag2[i]=sum(AA)

        Ag[j,:]=Ag2

        Sa[j,:]=np.array(funcA(ww, dt, Ag[j,:]))  #单条地震动的反应谱
        print('第'+str(mk+1)+'次迭代'',第'+str(j+1)+'个样本完成...')

    #生成的地震动的平均谱
    Sa_=Sa.sum(axis=0)/sample
    # Sa_alpha=Sa.sum(axis=0)/sample/Apga

    epslon_m=sum(np.abs((St_-Sa_)/St_))/N0
    epslon_max=np.abs((St_-Sa_)/St_).max()
    print('epslon_m=',epslon_m,'epslon_max=',epslon_max)

    if epslon_m<0.02 and epslon_max<0.03:
        break

    RR=St_/Sa_
    S_w=RR*S_w

# with open('XX.txt','a',encoding='utf-8') as m:
#     for i in range(len(t)):
#         m.write(str(Ag[1,i])+'\n')
#
# with open('YY.txt','a',encoding='utf-8') as m:
#     for i in range(len(t)):
#         m.write(str(0.85*Ag[2,i])+'\n')
#
# with open('ZZ.txt','a',encoding='utf-8') as m:
#     for i in range(len(t)):
#         m.write(str(0.6*Ag[3,i])+'\n')

#批量保存地震动数据
dir_name = 'D:/Master Learning/Encode/论文/机器学习/论文写作/随机地震动调整/seismic waves/' #指定路径
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

mode1 = '{}-XX' #批量命名
mode2 = '{}-YY'
mode3 = '{}-ZZ'
num = sample/3

for j in range(1,int(num+1)):
    with open(dir_name + mode1.format(j) + '.txt','w',encoding='utf-8') as m:
        for i in range(len(t)):
            m.write(str(Ag[j,i])+'\n')

for j in range(1,int(num+1)):
    with open(dir_name + mode2.format(j) + '.txt','w',encoding='utf-8') as m:
        for i in range(len(t)):
            m.write(str(0.85*Ag[int(j+num-1),i])+'\n')

for j in range(1,int(num+1)):
    with open(dir_name + mode3.format(j) + '.txt','w',encoding='utf-8') as m:
        for i in range(len(t)):
            m.write(str(0.6*Ag[int(j+2*num-1),i])+'\n')

#绘制前两条随机地震动
plt.figure(figsize=(12,8),dpi=150)
plt.plot(t,Ag[1,:],'b',label='First motion')
plt.plot(t,Ag[2,:],'r',label='Second motion')
plt.xlabel('Time / s',fontsize=14)
plt.ylabel('Acceleration / m/s^2',fontsize=14)
plt.show()

#地震影响系数
plt.figure(figsize=(12,8),dpi=150)
plt.plot(2*np.pi/ww,Sa_,'b',label='Simulation')   #2*np.pi/wk=T
plt.plot(2*np.pi/ww,St_,'r', label='Standard')
plt.xlabel('T0 / s',fontsize=14)
plt.ylabel('Seismic influence coefficient',fontsize=14)
plt.legend(fontsize=14)
plt.show()

#test
#2条地震动谱
Sa1=np.zeros(len(ww))
Sa2=np.zeros(len(ww))
 
Sa1[:]=np.array(funcA(ww, dt, Ag[1,:]))
Sa2[:]=np.array(funcA(ww, dt, Ag[2,:]))
plt.figure(figsize=(12,8),dpi=150)
plt.plot(2*np.pi/ww,Sa_,'b',label='Simulation')
plt.plot(2*np.pi/ww,Sa1,'r',label='First motion')   #2*np.pi/wk=T
plt.plot(2*np.pi/ww,Sa2,'g',label='Second motion')
plt.xlabel('T0 / s',fontsize=14)
plt.ylabel('Seismic influence coefficient',fontsize=14)
plt.legend(fontsize=14)
plt.show()

