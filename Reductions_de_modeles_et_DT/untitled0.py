import numpy as np

#1 - puissance itérée
A=np.array([[1.0,1.0,0.5],[1.0,1.0,0.25],[0.5,0.25,2.0]])
x0=np.array([1,0,0])
epsilon=0.01

x=x0
lambd_anc=0
b=x/np.linalg.norm(x)
x=np.dot(A,b)
lambd=np.dot(b,x)
while (np.abs(lambd - lambd_anc)>epsilon):
    lambd_anc=lambd
    b=x/np.linalg.norm(x)
    x=np.dot(A,b)
    lambd=np.dot(b,x)
    
print(lambd)
print(np.linalg.eigvals(A))

#2 - puissance inverse

