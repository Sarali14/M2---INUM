import numpy as np
import csv

#1
def IsInTriangle(A, B, C, P, eps=1e-12):
    v0 = B - A
    v1 = C - A
    v2 = P - A

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < eps:
        return 0  # degenerate triangle

    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    w = 1 - u - v

    return 1 if (u >= -eps and v >= -eps and w >= -eps) else 0

def AireTri(A,B,C):
    return 0.5 * abs(np.cross(B - A, C - A))

A = np.array([0.0, 0.0,0.0])
B = np.array([0.0, 1.0, 0.0])
C = np.array([1.0, 0.0,0.0])

X1 = np.array([0.5, 0.5,0.0])   
X2 = np.array([0.5, 0.51,0.0])   

print(IsInTriangle(A, B, C, X1))  
print(IsInTriangle(A, B, C, X2))  

#----------------------------------------------------------------#

#2
with open('test_temperature_triangle.csv',  'r', encoding='utf-8-sig') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    Data = list(reader) 

Data = np.array(Data, dtype=float)

M= Data[0:3,:]
T= Data[3,:]

print(M)
n=M.shape[1]      
#----------------------------------------------------------------#

#3

A=np.zeros((2,n))

for i in range(n):
    A[0,i]=np.linalg.norm(M[:,i] - X1)
    A[1,i]=i
    
#----------------------------------------------------------------#
    
#4

print("unsorted : " ,A)
ind=np.argsort(A[0,:])
A=A[:,ind]
print("sorted : ", A)



