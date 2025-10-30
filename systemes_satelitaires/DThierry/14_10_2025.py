import numpy as np
import matplotlib.pyplot as plt 

mu=3.98601e14
rt=6378140
alt=600000
r1=rt+alt
ainit=r1
ageo=42164178.26
a=(r1+ageo)/2

V1=np.sqrt(mu/r1)
V2=np.sqrt(mu*(2/r1-1/a))
V3=np.sqrt(mu*(2/ageo-1/a))
V4=np.sqrt(mu/ageo)

DV1=np.abs(V2-V1)
DV2=np.abs(V4-V3)
DV=DV1+DV2

#3
iinit=0
iinter=7

DV1=np.sqrt(V2**2+V1**2-2*V1*V2*np.cos(iinit-iinter))
DV2=np.sqrt(V3**2+V4**2-2*V3*V4*np.cos(iinit))
DV=DV1+DV2

print(DV1)
print(DV2)
print(DV)

#%%

def maxi(iinter):
    DV1=np.sqrt(V2**2+V1**2-2*V1*V2*np.cos(iinit-iinter))
    DV2=np.sqrt(V3**2+V4**2-2*V3*V4*np.cos(iinit))
    DV=DV1+DV2
    return DV

iinter=np.linspace(0,0.12217,100)
DV=np.zeros(100)
for i in range(100):
    DV[i]=maxi(iinter[i])
    
plt.plot(iinter,DV)

