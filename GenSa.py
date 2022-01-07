import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def myresponse(inpacc,delta,damp,Tmax):
    #Tmax是反应谱周期最大值
    inpacc=np.array(inpacc).reshape(-1,1)
    count=inpacc.shape[0]
    displace=np.zeros([count])
    velocity=np.zeros([count])
    absacce=np.zeros([count])
    ta=np.arange(delta,Tmax,delta)  # 频段范围-1/deltaHZ
    mdis=np.zeros([len(ta)])#只记录一种阻尼比的响应幅值
    mvel=np.zeros([len(ta)])
    macc=np.zeros([len(ta)])
    frcy=2*math.pi/ta
    damfrcy=frcy*np.sqrt(1-damp**2)
    e_t=np.exp(-damp*frcy*delta)
    s=np.sin(damfrcy*delta)
    c=np.cos(damfrcy*delta)
    d_f=(2*damp**2-1)/(frcy**2*delta)
    d_3t=damp/(frcy**3*delta)
    for i in range(len(ta)):
        A=np.zeros([2,2])
        A[0,0]=e_t[i]*(s[i]*damp/np.sqrt(1-damp**2)+c[i])
        A[0,1]=e_t[i]*s[i]/damfrcy[i]
        A[1,0]=-frcy[i]*e_t[i]*s[i]/np.sqrt(1-damp**2)
        A[1,1]=e_t[i]*(-s[i]*damp/np.sqrt(1-damp**2)+c[i])
        B=np.zeros([2,2])
        B[0,0]=e_t[i]*((d_f[i]+damp/frcy[i])*s[i]/damfrcy[i]+(2*d_3t[i]+1/frcy[i]**2)*c[i])-2*d_3t[i]
        B[0,1]=-e_t[i]*(d_f[i]*s[i]/damfrcy[i]+2*d_3t[i]*c[i])-1/frcy[i]**2+2*d_3t[i]
        B[1,0]=e_t[i]*((d_f[i]+damp/frcy[i])*(c[i]-damp/np.sqrt(1-damp**2)*s[i])-(2*d_3t[i]+1/frcy[i]**2)*(damfrcy[i]*s[i]+damp*frcy[i]*c[i]))+1/(frcy[i]**2*delta)
        B[1,1]=e_t[i]*(1/(frcy[i]**2*delta)*c[i]+s[i]*damp/(frcy[i]*damfrcy[i]*delta))-1/(frcy[i]**2*delta)
        for k in range(0,count-1):
            displace[k+1]=A[0,0]*displace[k]+A[0,1]*velocity[k]+B[0,0]*inpacc[k]+B[0,1]*inpacc[k+1]
            velocity[k+1]=A[1,0]*displace[k]+A[1,1]*velocity[k]+B[1,0]*inpacc[k]+B[1,1]*inpacc[k+1]
            absacce[k+1]=-2*damp*frcy[i]*velocity[k+1]-frcy[i]**2*displace[k+1]
        mdis[i]=np.max(np.abs(displace))
        mvel[i]=np.max(np.abs(velocity))
        if i==0:
            macc[i]=np.max(np.abs(inpacc))
        else:
            macc[i]=np.max(np.abs(absacce))
#     plt.plot(ta,macc)
    return macc,ta

#读取数据
inpacc=pd.read_csv('1_X.txt')  #一栏数据  需要把时间栏删除 单位采用国际单位 m/s/s

delta=0.01 #地震动步长 可根据具体的地震动而改变
damp=0.05  #阻尼比 这个一般不改变
Tmax=5 #加速度谱的最大周期
macc,ta=myresponse(inpacc,delta,damp,Tmax)

#反应谱曲线
plt.figure(figsize=(12,8),dpi=150)
plt.plot(ta,macc)
plt.show()

#加速度曲线
# plt.figure(figsize=(12,8),dpi=150)
# plt.plot(np.linspace(0,int(len(inpacc)*delta),len(inpacc)),np.array(inpacc))
# plt.show()



