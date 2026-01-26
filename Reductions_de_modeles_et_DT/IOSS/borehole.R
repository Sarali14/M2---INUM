
##################################################################################

borehole <- function(x)
{
  ##########################################################################
  #
  # BOREHOLE FUNCTION
  #
  ##########################################################################
  #
  # OUTPUT AND INPUT:
  #
  # y  = water flow rate
  # x = c(rw, riw, r, Tu, Hu, Tum, Hum, Tlm, Hlm, Tl, Hl, L, Kw)
  #
  ##########################################################################
  
  xx <- matrix(x, ncol=13)
  
  rw  <- xx[,1]
  riw <- xx[,2]
  r   <- xx[,3]
  Tu  <- xx[,4]
  Hu  <- xx[,5]
  Tum <- xx[,6]
  Hum <- xx[,7]
  Tlm <- xx[,8]
  Hlm <- xx[,9]
  Tl  <- xx[,10]
  Hl  <- xx[,11]
  L   <- xx[,12]
  Kw  <- xx[,13]
  
  frac1 <- 2 * pi * Tu * (Hu-Hl)
  frac11 <- (Tum - Tlm) * (Hum - Hlm)
  frac2 <- frac1 / frac11
  
  frac2a <- 2*L*Tu / (log(r/rw)*rw^2*Kw)
  frac2b <- Tu / Tl
  frac2 <- log(r/rw) * (1+frac2a+frac2b)
  
  y <- frac1 / frac2
  return(y)
}

##################################################################################

EchantBorehole <- function(N){
  
  # Description de la fonction :
  # Cette fonction genere un échantillon Monte Carlo de taille N,
  # pour le modele borhole
  #
  # Entrées de la fonction :
  # - taille = taille de l'échantillon
  #
  # Description de la sortie de la fonctio : 
  # - X = matrice de taille N x 13 qui contient des valeurs des 13 entrées
  # (statistiquement independantes) du modele borehole
  
  X = matrix(NA, N, 13)
  
  X[,1] <- rnorm(N, 0.1, 0.015)
  X[,2] <- rnorm(N, 0.05, 0.01)
  X[,3] <- rlnorm(N, 7.71, 1.0056)
  X[,4] <- runif(N, 63100, 116000)
  X[,5] <- runif(N, 1000, 1100)
  X[,6] <- runif(N, 6310, 11600)
  X[,7] <- runif(N, 900, 1000)
  X[,8] <- runif(N, 631, 1160)
  X[,9] <- runif(N, 800, 900)
  X[,10] <- runif(N, 63.1, 116)
  X[,11] <- runif(N, 700, 800)
  X[,12] <- runif(N, 1120, 1680)
  X[,13] <- runif(N, 3000, 12000)
  
  colnames(X) <- c("rw","riw","r","Tu","Hu","Tum","Hum","Tlm","Hlm","Tl","Hl","L","Kw") # noms de variables aleatoires
  
  return(X)
  
}#end function


d <- 13 # nombre de variables incertaines
names <- c("rw","riw","r","Tu","Hu","Tum","Hum","Tlm","Hlm","Tl","Hl","L","Kw") # noms de variables aleatoires

########################################################

             # SARAH ALI - Réponses # 

########################################################
set.seed(123)

#1------------------------------------------------------------------------------
# evaulation % min
x_min <- c(0.05,0.02,100,63100,1000,6310,900,631,800,63.1,700,1120,3000)
Y_min <- borehole(x_min)

print(Y_min)

#evaluation % max
x_max <- c(0.15,0.08,50000,116000,1100,11600,1000,1160,900,116,800,1680,12000)
Y_max <- borehole(x_max)

print(Y_max)

#evaluation % moyennes
mean_r <- exp(7.71+0.5*(1.0056)^2)
x_mean <- c(0.10,0.05,mean_r,89550,1050,8955,950,895.5,850,89.55,750,1400,7500)
Y_mean <- borehole(x_mean)

print(Y_mean)

#2------------------------------------------------------------------------------
#2-a
N <- 1000
X_MC_1 <- EchantBorehole(N)
Y_MC_1 <- borehole(X_MC_1)

mean_Y_MC_1 <- mean(Y_MC_1)
var_Y_MC_1 <- var(Y_MC_1)

print(mean_Y_MC_1)
print(var_Y_MC_1)

hist(Y_MC_1,breaks = 30,probability = TRUE,main = "Histogram of the borehole flow rate",xlab = "Flow rate")
#2-b
#rapport

#2-c
m = 10000
#quantile 95%
q95_MC <- quantile(Y_MC_1 ,probs = 0.95)
print(q95_MC)

#Intervale de confiance
q95_valeurs <- numeric(m)
for (i in 1:m){
  x <- EchantBorehole(N)
  y <- borehole(x)
  q95_valeurs [i] = quantile(y, probs = 0.95)
}

IC_q95 = quantile(q95_valeurs, probs=c(0.025,0.975))
print(IC_q95)

#2-d
p <- 250
cv <- 0.1

N_seq <- seq (2*10^6, 4*10^7, by=2*10^6)

Ns_used <- numeric(length(N_seq))
p_hats <- numeric(length(N_seq))
cvs <- numeric(length(N_seq))
k <- 0
for (Ns in N_seq) {
  k = k + 1
  X <- EchantBorehole(Ns)
  Y <- borehole(X)
  
  p_hat <- mean(Y > p)
  cv_hat <- sqrt((1 - p_hat) / (Ns * p_hat))   # relative error estimate
  
  Ns_used[k] <- Ns
  p_hats[k] <- p_hat
  cvs[k] <- cv_hat
  
  if (is.finite(cv_hat) && cv_hat <= cv) break
}
p_hats <- p_hats[1:k]
cvs <- cvs[1:k]
Ns_used <- Ns_used[1:k]

plot(Ns_used, p_hats, type = "b",
     xlab = "Sample size N",
     ylab = "Estimated P(Y > 250)",
     main = "Monte Carlo convergence for P(Y > 250)")

print(Ns_used[length(Ns_used)])

#3------------------------------------------------------------------------------
#3-a-i

L <- 500
X_MC_2 <- EchantBorehole(L)
Y_MC_2 <- borehole(X_MC_2)

x11()
pairs(cbind(debit = Y_MC_2, X_MC_2),
      main = "Scatterplots entre le débit et les entrées")

#3-a-ii 
library(sensitivity)

res_src <- sensitivity::src(X_MC_2, Y_MC_2)
print(res_src)
print(res_src$SRC)
print(res_src$SRC^2)   

#3-b

M <- 1000
X1 <- EchantBorehole(M)
X2 <- EchantBorehole(M)

sa <- sensitivity::sobolmartinez(model = borehole, X1 = X1, X2 = X2, nboot = 100)

print(sa)
plot(sa) 
#4------------------------------------------------------------------------------
#4-a
library(DiceKriging)

X_train <- as.data.frame(X_MC_2)
Y_train <- Y_MC_2

krig <- km(formula  = ~ 1,design   = X_train,response = Y_train,covtype  = "matern5_2")

print(krig)

N_test <- 5000
X_test <- as.data.frame(EchantBorehole(N_test))
Y_test <- borehole(as.matrix(X_test))

pred_test <- predict(krig, newdata = X_test, type = "UK", checkNames = FALSE)

Q2_test <- 1 - mean((Y_test - pred_test$mean)^2) / var(Y_test)
cat("Q2 (kriging, test set) =", Q2_test, "\n")


loo <- leaveOneOut.km(krig, type = "UK")
Q2_loo <- 1 - mean((Y_train - loo$mean)^2) / var(Y_train)
cat("Q2 (kriging, LOO) =", Q2_loo, "\n")


x11()
plot(Y_test, pred_test$mean,
     xlab = "True borehole output (test)",
     ylab = "Kriging prediction",
     main = paste("Kriging validation (Q2 =", round(Q2_test, 3), ")"))
abline(0, 1, lty = 2)

#4-b
M <- 1000
X1 <- EchantBorehole(M)
X2 <- EchantBorehole(M)

krig_model <- function(X){
  Xdf <- as.data.frame(X)
  colnames(Xdf) <- colnames(X_train)  # ensure same variable names/order as training
  predict(krig, newdata = Xdf, type = "UK", checkNames = FALSE)$mean
}

# 3) Sobol sur le métamodèle (bootstrap inclus comme en 3-b)
sa_krig <- sobolmartinez(model = krig_model, X1 = X1, X2 = X2, nboot = 100)

print(sa_krig)
plot(sa_krig)

# 4) Comparaison simple (colonne "original")
S1_borehole <- sa$S$original
T_borehole  <- sa$T$original

S1_krig <- sa_krig$S$original
T_krig  <- sa_krig$T$original

compar <- data.frame(
  S1_borehole = S1_borehole,
  S1_krig     = S1_krig,
  T_borehole  = T_borehole,
  T_krig      = T_krig
)
print(compar)
