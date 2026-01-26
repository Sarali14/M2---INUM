
#####################################################################
# TD 4 - Analyse de sensibilite, Sobol
# Bertrand Iooss
# Polytech Nice Sophia

# S O L U T I O N S

#####################################################################

rm(list=ls())
graphics.off()

library(sensitivity)

# ------------ 1 Estimateurs Pick-freeze ------

modele <- function(x){
  return(x[,1]^2*(1+cos(x[,2]))^2)
}

# 1a) Calculer les indices de Sobol du modele 
# avec X1 ~ U[O,1] et X2 ~ N(0,1)

# construire 2 matrices independantes des entrees (de taille 1000)
n = 1000

x1 = runif(n,0,1)
x2 = rnorm(n,0,1)
x = cbind(x1,x2)

x1p = runif(n,0,1)
x2p = rnorm(n,0,1)
xp = cbind(x1p,x2p)

# calculer le denominateur des indices de Sobol
# en faisant tourner le modele sur x
y = modele(x)
V = var(y)

# calculez S1 et S2 avec l'estimation pick-freeze
y1 <- modele(cbind(x1,x2p))
y2 <- modele(cbind(x1p,x2))

S1 <- cov(y,y1)/V
S2 <- cov(y,y2)/V

print(c(S1,S2))

# 1b) Calculer les indices de Sobol avec la fonction sobol() 
# du package "sensitivity"

?sobol

sa <- sensitivity::sobol(model=modele,X1=x,X2=xp,order=2,nboot=100)
print(sa)
plot(sa)

# 1c)

n = 10000

x1 = runif(n,0,1)
x2 = rnorm(n,0,1)
x = cbind(x1,x2)

x1p = runif(n,0,1)
x2p = rnorm(n,0,1)
xp = cbind(x1p,x2p)

sa <- sensitivity::sobol(model=modele,X1=x,X2=xp,order=2,nboot=100)
print(sa)
plot(sa)

# ---------- 2 Robot function ----------------

rm(list=ls())
graphics.off()

library(sensitivity)

names = c("theta1", "theta2", "theta3", "theta4", "L1", "L2", "L3", "L4")
d = 8 # Number of input variables
robot <- function(xx)
{
  y <- NULL
  for (i in 1:dim(xx)[[1]]){
    theta <- xx[i,1:4] * 2 * pi
    L     <- xx[i,5:8]
    thetamat <- matrix(rep(theta,times=4), 4, 4, byrow=TRUE)
    thetamatlow <- thetamat
    thetamatlow[upper.tri(thetamatlow)] <- 0
    sumtheta <- rowSums(thetamatlow)
    u <- sum(L*cos(sumtheta))
    v <- sum(L*sin(sumtheta))
    y <- c(y,(u^2 + v^2)^(0.5))
  }
  return(y)
}

# a) src()
n = 1000
x = matrix(runif(n*d),nr=n)
colnames(x) = names
y = robot(x)

src = sensitivity::src(x,y)
print(src)
print(src$SRC^2)
sum(src$SRC^2)

# b) sobol2002()
n = 1000

X1 = matrix(runif(n*d),nr=n)
colnames(X1) = names
X2 = matrix(runif(n*d),nr=n)
colnames(X2) = names

sa = sensitivity::sobol2002(model=robot,X1=X1,X2=X2,nboot=100)
print(sa)
x11()
plot(sa)

n = 10000

X1 = matrix(runif(n*d),nr=n)
colnames(X1) = names
X2 = matrix(runif(n*d),nr=n)
colnames(X2) = names

sa = sensitivity::sobol2002(model=robot,X1=X1,X2=X2,nboot=100)
print(sa)
x11()
plot(sa)

# c) sobolmartinez()

n = 1000

X1 = matrix(runif(n*d),nr=n)
colnames(X1) = names
X2 = matrix(runif(n*d),nr=n)
colnames(X2) = names

sa = sensitivity::sobolmartinez(model=robot,X1=X1,X2=X2,nboot=100)
print(sa)
x11()
plot(sa)

n = 10000

X1 = matrix(runif(n*d),nr=n)
colnames(X1) = names
X2 = matrix(runif(n*d),nr=n)
colnames(X2) = names

sa = sensitivity::sobolmartinez(model=robot,X1=X1,X2=X2,nboot=100)
print(sa)
x11()
plot(sa)

# d) echantillons quasi-Monte carlo

library(randtoolbox)

n = 10000

X = randtoolbox::sobol(n,2*d)
X1 = X[,1:d]
colnames(X1) = names
X2 = X[,(d+1):(2*d)]
colnames(X2) = names

sa = sensitivity::sobol2002(model=robot,X1=X1,X2=X2)
print(sa)
x11()
plot(sa)

# e) Estimation par metamodele processus gaussiens

# a
library(DiceDesign)

N <- 400
desinit <- lhsDesign(N, d)
des <- discrepSA_LHS(desinit$design, c=0.95, it=2000)
x11()
plot(des$critValues, type="l")

x <- des$design
colnames(x) <- names
y <- robot(x)

# b
library(DiceKriging)
?km
krig <- km(formula=~1, x, y)
# Calcul du Q2 (coef de predictivite du metamodele) par technique leane-one-out
res2 <- (leaveOneOut.km(krig, type="UK")$mean - y)^2
Q2 <- 1 - mean(res2) / var(y)
print(Q2)

# c
library(sensitivity)

n = 10000

X1 = matrix(runif(n*d),nr=n)
colnames(X1) = names
X2 = matrix(runif(n*d),nr=n)
colnames(X2) = names

sa = sensitivity::sobolmartinez(model=NULL, X1=X1, X2=X2, nboot=100)
yy <- predict.km(krig, sa$X, type = "UK", se.compute = FALSE)
tell(sa, yy$mean)
print(sa)
x11()
plot(sa)

# d
?sobolGP

n = 1000

X1 = matrix(runif(n*d),nr=n)
colnames(X1) = names
X2 = matrix(runif(n*d),nr=n)
colnames(X2) = names

sa <- sobolGP(krig, type="UK", MCmethod="sobol2002", X1=X1, X2=X2)
plot(sa)

########################################################
# Materiel annexe : calcul d'indices aux ordres + eleves

sa12T = sensitivity::sobolSalt(model=NULL, X1=X1, X2=X2, scheme="B", nboot=100)
yy <- predict.km(krig, sa12T$X, type = "UK", se.compute = FALSE)
tell(sa12T, yy$mean)
print(sa12T)
x11()
plot(sa12T,choice=2)

# interaction d'ordre 3 (theta2,theta3,theta4) de 15%
sa = sensitivity::sobol(model=NULL, X1=X1, X2=X2, nboot=100, order=list(2,3,4,2:3,c(2,4),3:4,2:4))
yy <- predict.km(krig, sa$X, type = "UK", se.compute = FALSE)
tell(sa, yy$mean)
print(sa$V[-1,]/sa$V[1,1])
print(sa12T$T)

n = 10000

X1 = matrix(runif(n*d),nr=n)
colnames(X1) = names
X2 = matrix(runif(n*d),nr=n)
colnames(X2) = names

# interaction entre longueurs
sa = sensitivity::sobol(model=NULL, X1=X1, X2=X2, nboot=100, 
                        order=list(5,6,7,8,c(5,6),c(5,7),c(5,8),c(6,7),c(6,8),c(7,8),5:8))
yy <- predict.km(krig, sa$X, type = "UK", se.compute = FALSE)
tell(sa, yy$mean)
print(sa$V[-1,]/sa$V[1,1])
print(sa12T$T)

# interactions d'ordres 1,2,3
sa3 = sensitivity::sobol(model=NULL, X1=X1, X2=X2, order=3, nboot=100)
yy <- predict.km(krig, sa3$X, type = "UK", se.compute = FALSE)
tell(sa3, yy$mean)
print(sa3)
x11()
plot(sa3)


