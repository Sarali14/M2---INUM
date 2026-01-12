
#####################################################################
# TD 4 - Analyse de sensibilite, Sobol
# Bertrand Iooss
# Polytech Nice Sophia

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

x1 = ???
x2 = ???
x = cbind(x1,x2)

x1p = ???
x2p = ???
xp = cbind(x1p,x2p)

# calculer le denominateur des indices de Sobol
# en faisant tourner le modele sur x
y = modele(x)
V = ???

# calculez S1 et S2 avec l'estimation pick-freeze
y1 <- modele(cbind(x1,x2p))
y2 <- modele(cbind(x1p,x2))

S1 <- ?? / V
S2 <- ?? / V

print(c(S1,S2))

# 1b) Calculer les indices de Sobol avec la fonction sobol() 
# du package "sensitivity"

?sobol

sa <- sensitivity::sobol(??)
print(sa)
plot(sa)

# 1c)

n = 10000

x1 = ???
x2 = ???
x = cbind(??,??)

x1p = ???
x2p = ???
xp = cbind(??,??)

sa <- sensitivity::sobol(???)
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
x = matrix(???,nr=n)
colnames(x) = names
y = ???

src = sensitivity::???(x,y)
print(src)
print(src$SRC^2)
sum(src$SRC^2)

# b) sobol2002()
n = 1000

X1 = matrix(???,nr=n)
colnames(X1) = names
X2 = matrix(???,nr=n)
colnames(X2) = names

sa = sensitivity::sobol2002(???)
print(sa)
x11()
plot(sa)

n = 10000

X1 = matrix(???,nr=n)
colnames(X1) = names
X2 = matrix(???,nr=n)
colnames(X2) = names

sa = sensitivity::sobol2002(???)
print(sa)
x11()
plot(sa)

# c) sobolmartinez()

n = 1000

X1 = matrix(???,nr=n)
colnames(X1) = names
X2 = matrix(???,nr=n)
colnames(X2) = names

sa = sensitivity::sobolmartinez(???)
print(sa)
x11()
plot(sa)

n = 10000

X1 = matrix(???,nr=n)
colnames(X1) = names
X2 = matrix(???,nr=n)
colnames(X2) = names

sa = sensitivity::sobolmartinez(???)
print(sa)
x11()
plot(sa)

# d) echantillons quasi-Monte carlo

library(randtoolbox)

n = 10000

X = randtoolbox::sobol(???)
X1 = ???
colnames(X1) = names
X2 = ???
colnames(X2) = names

sa = sensitivity::sobol2002(???)
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

