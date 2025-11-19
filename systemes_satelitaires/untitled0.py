import numpy as np
import sympy as sp

l=sp.symbols('l')


B = np.array([[1],[1/2000]])
A=sp.Matrix([[0,1],[0,0]])
C=np.array([1,0,2])
M=np.array([[0,1/2000],[1/2000,0]])
char_poly=A.charpoly(l)


print(char_poly.as_expr())
#print(char_poly.factor())
print(np.linalg.matrix_rank(M))
#%%
print(2*np.pi*np.sqrt(((24363.57*10**3)**3)/(3.98601*10**14))/3600)

print(24363.57*(1+0.730616654))

print((398601*(86164/(2*np.pi))**2)**(1/3))

print(28+27/60 +17/3600)

print(np.sqrt((398601/24363.57)*((1+0.730616654)/(1-0.730616654))))

print(np.sqrt(398601/42164.15))

print(3.074-1.595)

