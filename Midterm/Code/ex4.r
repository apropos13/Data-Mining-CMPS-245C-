library(glmnet)
data <-read.csv("../Data/filtered_no_missing_no_norm.csv")


#define attributes
current_satisfaction_score<-data$current_satisfaction_score



#Determine Split
n<-nrow(data)
split_point<-round(n*0.66)

#take random sample
data<-data[sample(n),]

train<-data[1:split_point,]
test<-data[split_point:n,]


fit.lm <-lm(current_satisfaction_score~. , data=train)
coef(fit.lm)


#make predictions from all the data (careful order above)
predictions.lm <- predict(fit.lm, newdata=test)
mse.lm<-mean((test$current_satisfaction_score-predictions.lm)^2)
mse.lm


mae.lm<-mean( abs(test$current_satisfaction_score-predictions.lm ))
mae.lm

rae.lm<-mean(abs (test$current_satisfaction_score-predictions.lm )/test$current_satisfaction_score)
rae.lm

bias.lm<-sum(test$current_satisfaction_score-predictions.lm)
bias.lm

variance.lm<- sum(predictions.lm^2)- sum(predictions.lm)^2
variance.lm

summary(predictions.lm)


#ADVANCED START
m<-ncol(data)
x.train<-as.matrix(data[1:split_point,1:m-1])
x.test<-as.matrix(data[split_point:n,1:m-1])

y.train<-as.matrix(data[1:split_point,m])
y.test<-as.matrix(data[split_point:n,m])

lambda_parameter<-0.05

fit.lasso<-glmnet(x.train, y.train, alpha=1, lambda=lambda_parameter)
#coef(fit.lasso)

fit.ridge<-glmnet(x.train, y.train, alpha=0, lambda=lambda_parameter)
#coef(fit.ridge)

fit.elnet<-glmnet(x.train, y.train, alpha=0.5, lambda=lambda_parameter)
coef(fit.elnet)

fit.c1<-glmnet(x.train, y.train, alpha=0.0005, lambda=lambda_parameter)
#coef(fit.c1)

fit.c1<-glmnet(x.train, y.train, alpha=0.005, lambda=lambda_parameter)
#coef(fit.c1)

fit.c1<-glmnet(x.train, y.train, alpha=0.5, lambda=lambda_parameter)
#coef(fit.c1)

fit.c1<-glmnet(x.train, y.train, alpha=5, lambda=lambda_parameter)
#coef(fit.c1)



#make predictions:
predictions.lasso <- predict(fit.lasso, x.test, type="link")
predictions.ridge <- predict(fit.ridge, x.test, type="link")
predictions.elnet <- predict(fit.elnet, x.test, type="link")

#summarize accuracy
mse.lasso <-mean( (y.test-predictions.lasso)^2)
mse.lasso
max(y.test-predictions.lasso )
min(y.test-predictions.lasso )

mse.ridge <-mean( (y.test-predictions.ridge)^2)
mse.ridge
max(y.test-predictions.ridge )
min(y.test-predictions.ridge )


mse.elnet <-mean( (y.test-predictions.elnet)^2)
mse.elnet
max(y.test-predictions.elnet )
min(y.test-predictions.elnet )


max(y.test-predictions.lm )
min(y.test-predictions.lm )

