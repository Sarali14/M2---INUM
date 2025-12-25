
#####################################################################
# TD - Initiation a l'analyse statistique dans l'environment R

# Bertrand Iooss
# Polytech Nice Sophia

# S O L U T I O N S

#####################################################################

rm(list=ls())
graphics.off()

##################################
# 2

data()
?airquality
a = airquality
names(a)
dim(a)

# 2.1
summary(a)
# Question

# 2.2
boxplot(a)
x11()
par(mfrow=c(2,3))
sapply(a,boxplot)
names = dimnames(a)[[2]]
x11()
par(mfrow=c(2,3))
for (i in 1:dim(a)[2]) boxplot(a[,i],main=names[i],col=8)

# 2.3
x11()
hist(a$Wind)

# Exercice *****
x11()
hist(a$Wind,breaks=20) 
# End **********
x11()
hist(a$Wind,prob=TRUE)
lines(density(a$Wind))

# Exercice *****
x11()
par(mfrow=c(2,3))
for (i in 1:dim(a)[2]) hist(a[,i],main=names[i],col=8)
# End **********

# 2.4
quantile(a$Wind)
quantile(a$Wind, probs=c(0.05, 0.95))

# Exercice *****
x11()
par(mfrow=c(1,3))
qqnorm(a$Wind)
qqline(a$Wind)
qqnorm(a$Ozone)
qqline(a$Ozone)
qqnorm(log(a$Ozone))
qqline(log(a$Ozone))
# End ***********

# 2.5
# Exercice *****
par_MM = c(mean(a$Wind),sd(a$Wind))
library(MASS)
par_ML <- fitdistr(a$Wind,"normal")$estimate
ks.test(a$Wind, "pnorm",mean=par_ML[1], sd=par_ML[2])
# End ***********

##################################
# 3

x11()
pairs(a,panel=panel.smooth)
# Question

#var(a)
var(a,na.rm=TRUE)
cor(a,use="complete.obs")
# Question

##################################
# 4

#??? 4.1
# Exercice *****
names = dimnames(a)[[2]]
n = 1000
x1=matrix(0,nrow=n,ncol=3,dimnames=list(1:n,names[2:4]))
x1[,1]=runif(1000,min=min(a$Solar.R,na.rm=T),
             max=max(a$Solar.R,na.rm=T))
x1[,2]=rnorm(1000,mean=mean(a$Wind),sd=sd(a$Wind))
x1[,3]=rnorm(1000,mean=mean(a$Temp),sd=sd(a$Temp))
summary(x1)

x2 = x1[x1[,2]>0,]
summary(x2)

x11()
par(mfrow=c(2,3))
for (i in 1:3) hist(x2[,i],main="Simulated",xlab=names[i+1])
for (i in 2:4) hist(a[,i],main="Data",xlab=names[i])
x11()
pairs(x1,panel=panel.smooth,main="Simulated")
x11()
pairs(a[,2:4],panel=panel.smooth,main="Real")
# End **********

# 4.2
library(MASS)
help(mvrnorm)
var(a,na.rm=TRUE)
covar=matrix(c(12.657324,-16.857166,-16.857166,90.820311),nrow=2,ncol=2)
x2=matrix(0,nrow=1000,ncol=3,dimnames=list(1:1000,names[2:4]))
x2[,1]=runif(1000,min=7,max=334)
x2[,2:3]=mvrnorm(1000,mu=c(mean(a$Wind),mean(a$Temp)),Sigma=covar)
summary(x2)
x3 = x2[ x2[,2]>0, ]
summary(x3)
x11()
pairs(x3,panel=panel.smooth,main="Simulated")
x11()
pairs(a[,2:4],panel=panel.smooth,main="Real")
cor(x3)
cor(a[,2:4],use="complete.obs")

##################################
# 5

b = a[ !is.na(a[,1]) & !is.na(a[,2]),]

# 5.1

formule = as.formula( b$Ozone ~ b$Solar.R + b$Wind + b$Temp + b$Month + b$Day)
m1 = lm(formule,data=data.frame(x=b[,2:6],y=b$Ozone))
m1
names(m1)
x11()
y = predict(m1,as.data.frame(b[,2:6]))
plot(y,b$Ozone,xlab="prediction",ylab="observation")
lines(y,y)
x11()
plot(m1)
summary(m1)
anova(m1)
c = m1$coefficients
src = rep(0,5)
src = matrix(c(src),ncol=5,dimnames=list(c("SRC"),names[2:6]))
for (i in 1:5) src[i] <- c[i+1]*sd(b[,i+1])/sd(b$Ozone)
print(src)
print(src^2)

# 5.2

formule = as.formula( b$Ozone ~ (b$Solar.R + b$Wind + b$Temp + b$Month)^2+I(b$Solar.R^2)+I(b$Wind^2)+I(b$Temp^2)+I(b$Month^2))
m2 = lm(formule,data=data.frame(x=b[,2:6],y=b$Ozone))
summary(m2)
m3=step(m2,trace=0)
summary(m3)
x11()
y = predict(m3,as.data.frame(b[,2:6]))
plot(y,b$Ozone,xlab="prediction",ylab="observation")
lines(y,y)

##################################
# 6

# Exercice *****
n=dim(x3)[[1]]
month=sample(5:9,n,replace=TRUE)
b=matrix(0,nrow=n,ncol=4,dimnames=list(1:n,names[2:5]))
b[,1:3]=x3
b[,4]=month
b=data.frame(b)
y = predict(m3,b)
x11()
par(mfrow=c(1,2))
hist(y,main="Simulated- poynomial model")
hist(a$Ozone,main="Real")
x11()
pairs(b)
quantile(y, probs=c(0.05, 0.95))
quantile(a$Ozone, na.rm=TRUE, probs=c(0.05, 0.95))
# End ***********

