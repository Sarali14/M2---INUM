
#####################################################################
# TP2 - Quantification et propagation d'incertitudes
# Bertrand Iooss
# Polytech Nice Sophia

# S O L U T I O N S

#####################################################################

rm(list=ls())
graphics.off()

library(MASS) # pour fitdistr
library(triangle)

#########################
#2.1
library(triangle)

# deux manieres de visualiser les lois

# avec plot et density
x11()
n <- 1e6
par(mfcol=c(2,2))
F1 <- rnorm(n,3e4,9e3)
plot(density(F1))
E1 <- rtriangle(n,2.8e7,4.8e7,3e7)
plot(density(E1))
L1 <- runif(n,250,260)
plot(density(L1))
I1 <- rtriangle(n,310,450,400)
plot(density(I1))

# ou avec curve
x11()
par(mfcol=c(2,2))
curve(dnorm(x,3e4,9e3),from = -10000 , to = 72000, col='blue',xlab='F',ylab='pdf',lwd=2)
curve(dtriangle(x,2.8e7,4.8e7,3e7),from = 2.7e7 , to = 4.9e7,col='red',xlab='E',ylab='pdf',lwd=2)
curve(dunif(x,250,260),from = 248 , to = 262,col=rgb(0.3,0.7,0.3),xlab='L',ylab='pdf',lwd=2)
curve(dtriangle(x,310,450,400),from = 305 , to = 455,col='orange',xlab='I',ylab='pdf',lwd=2)


#2.2

fleche <- function(x){ 
  # parametres dans l'ordre suivant : F, E, L, I
  # la fonction prend en entree une matrice de taille n x d et renvoie un vecteur de taille n
  if (is.vector(x)){x <- matrix(x,nrow=1)}  # passage d'un vecteur a une matrice
  result <- (x[,1]*x[,3]^3)/(3*x[,2]*x[,4])
  return(result)
}

#2.3

mu =c(3e4,((2.8e7+4.8e7+3e7)/3),255,((310+450+400)/3))  # point nominal pour calcul fleche en cm

# calul des variances des 4 parametres avec formule analytique
varF = 9000^2
varE = (2.8e7^2 + 4.8e7^2 + 3e7^2 - 2.8e7*4.8e7 - 2.8e7*3e7 - 4.8e7*3e7)/18 
varL = (260 - 250)^2/12
varI = ( 310^2 + 450^2 + 400^2 -310*450 - 310*400 - 450*400)/18

m_hat_cum <- fleche(mu)    # moyenne par cumul quadratique ordre 1
print(m_hat_cum)

help(deriv) # Differentiation formelle dans R
DE <- deriv(y ~ ( F * L^3 ) / (3 * E * I), c("F","E","L","I"))
print(DE)

# evaluation du gradient au point nominal
F = mu[1]
E = mu[2]
L = mu[3]
I = mu[4]

grad <- eval(DE)
grad <- attributes(grad)$gradient[1,]
print("Derivees exactes :")
print(grad)

#Cumul quadratique
var_hat_cum <- grad[1]^2*varF+grad[2]^2*varE+grad[3]^2*varL+grad[4]^2*varI  # variance par cumul quadratique ordre 1
print(as.numeric(var_hat_cum))

#2.4

# Methode de MC
n <- 200

# generation des echantillons aleatoires des parametres d'entrees
F <- rnorm(n,3e4,9e3)
E <- rtriangle(n,2.8e7,4.8e7,3e7)
L <- runif(n,250,260)
I <- rtriangle(n,310,450,400)

data <- cbind(F,E,L,I)    # concatenation des parametres d'entrees (en colonne)

Y <- fleche(data)  # calcul des reponses du modele

#---------------- a)

m_hat_MC = mean(Y) # moyenne de Y
print(m_hat_MC )
print(m_hat_cum)

var_hat_MC  = var(Y)   # variance de Y
print(var_hat_MC)
print(var_hat_cum)

summary(Y)   # resume de l'echantillon Y 

x11()  
hist(Y) # histogramme de Y

?hist
par(mfcol=c(1,1))
hist(Y,probability=TRUE,col='lightblue',main='histogramme de la fleche')
lines(density(Y),col='blue',lwd=2)

# comparaison cumul quadratique et Monte Carlo
print(as.numeric(var_hat_cum))
print(var_hat_MC)

#---------------- b)

# Intervalle de confiance sous la forme d'un vecteur c( min_IC, max_IC )
IC95_MC  <- c(m_hat_MC -1.96*sqrt(var_hat_MC )/sqrt(n),m_hat_MC +1.96*sqrt(var_hat_MC )/sqrt(n))
print(IC95_MC )
print(m_hat_MC )


#----------------- c)

m_hat_cv <- c()           # initialisation
var_hat_cv <- c()
IC95_cv <- c()

seqn = c(seq(10,5000,by=50))# taille de l'echantillon dans la boucle

for (k in seqn){   # boucle
  Fk <- rnorm(k,3e4,9e3)
  Ek <- rtriangle(k,2.8e7,4.8e7,3e7)
  Lk <- runif(k,250,260)
  Ik <- rtriangle(k,310,450,400)
  datak = cbind(Fk,Ek,Lk,Ik)    
  Yk = fleche(datak)
  m_hat_cv <- c(m_hat_cv,mean(Yk))
  var_hat_cv <- c(var_hat_cv,var(Yk))
  IC95_cv <- rbind(IC95_cv,c(mean(Yk)-1.96*sqrt(var(Yk))/sqrt(k),mean(Yk)+1.96*sqrt(var(Yk))/sqrt(k)))
}

# graphe illustratif
x11()
par(mfrow = c(2, 1))
plot(seqn, m_hat_cv,col='blue',type='l',xlab='n',ylab='estimation de m',main='estimation de m en fonction de n')
lines(seqn,IC95_cv[,1],col='orange')
lines(seqn,IC95_cv[,2],col='orange')
plot(seqn ,sqrt(var_hat_cv)/m_hat_cv,col='red',type='l',xlab='n',ylab='coefficient de variation',main='estimation du coefficient de variation fonction de n')

################### 2.5

nn <- 20
B <- 1000

Z <- sample(Y,nn)
mZ <- mean(Z)

mzz <- NULL
for (b in 1:B){
  zz <- sample(Z,nn,replace=T)
  mzz <- c(mzz,mean(zz))
}

print(quantile(mzz,probs=c(0.025,0.975)))
print(IC95_MC )

nn <- 200
B <- 1000

Z <- sample(Y,nn)
mZ <- mean(Z)

mzz <- NULL
for (b in 1:B){
  zz <- sample(Z,nn,replace=T)
  mzz <- c(mzz,mean(zz))
}

print(quantile(mzz,probs=c(0.025,0.975)))
print(IC95_MC )

#########################
# 3

defaill = function(x){ 
  if (is.vector(x)){x = matrix(x,nrow=1)} 
  res = 30 - (x[,1]*x[,3]^3)/(3*x[,2]*x[,4])
  return(res)
}

simul_def = function(n) {
  F = rnorm(n,3e4,9e3)
  E = rtriangle(n,2.8e7,4.8e7,3e7)
  L = runif(n,250,260)
  I = rtriangle(n,310,450,400)
  data = cbind(F,E,L,I)
  return(defaill(data))
}

# 3.1

n = 100000

def = simul_def(n)

pf =  sum(def < 0) / n
print(pf)

cv <- sqrt((1-pf)/(n*pf))
print(cv)

IC95_pf =  c(pf -1.96*sqrt(pf*(1-pf))/sqrt(n),pf +1.96*sqrt(pf*(1-pf))/sqrt(n))
print(IC95_pf)

# 3.2

cvlim <- 0.01 # critere d'arret coefficient de variation
Nlim <- 1e8   # critere d'arret nombre de tirage

N = 2e6
Nrun = N
cv = 1
def = NULL
res = NULL

# Si trop long a tourner -> reduire Nlim et augmenter cvlim
while ((Nrun <= Nlim) & (cv > cvlim)){
  Y = simul_def(N)
  def = c(def,Y)
  Nrun = length(def)
  pf = sum(def < 0) / Nrun
  cv = sqrt((1-pf)/(Nrun*pf))
  res = rbind(res,c(Nrun,pf,cv))
  print(c(Nrun,pf,cv))
}

x11() 
plot(res[,1],res[,2],xlab='N',ylab='pf')
