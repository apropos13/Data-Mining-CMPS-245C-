data <-read.csv("/Users/panos/Desktop/UCSC/Spring_2017/Data Mining/Hw2/world_happiness_survey_data.csv")
data <- data[,!(colnames(data)%in% c("country"))]


#define attributes
econ<-data$economy
family<-data$family
health<-data$health
freedom<-data$freedom
govern<-data$government_corruption
happy<-data$happiness_score

gimme_predictions <- function(model){

	predictions.lm <- predict(model, newdata=test)
	mse.lm<-mean((test$happiness_score-predictions.lm)^2)
	print(mse.lm)
	"
	mae.lm<-mean( abs(test$happiness_score-predictions.lm ))
	mae.lm

	rae.lm<-mean(abs (test$happiness_score-predictions.lm )/test$happiness_score)
	rae.lm

	bias.lm<-sum(test$happiness_score-predictions.lm)
	bias.lm

	variance.lm<- sum(predictions.lm^2)- sum(predictions.lm)^2
	variance.lm
	"


}

#General
cor(data, method="pearson")

#Economy
mean(econ)
median(econ)
sd(econ)
mad(econ)
sd(econ)
pdf("econ_hist.pdf")
h<-hist(econ, xlab="Economy")
dev.off()

#Family 
mean(family)
median(family)
sd(family)
mad(family)
sd(family)
pdf("fam_hist.pdf")
h<-hist(family, xlab="Family")
dev.off()

#Health 
mean(health)
median(health)
sd(health)
mad(health)
sd(health)
pdf("health_hist.pdf")
h<-hist(health, xlab="Health")
dev.off()

#Freedom 
mean(freedom)
median(freedom)
sd(freedom)
mad(freedom)
sd(freedom)
pdf("free_hist.pdf")
h<-hist(freedom, xlab="Freedom")
dev.off()


#Gov 
mean(govern)
median(govern)
sd(govern)
mad(govern)
sd(govern)
pdf("gov_hist.pdf")
h<-hist(govern, xlab="Government")
dev.off()


#Happiness 
mean(happy)
median(happy)
sd(happy)
mad(happy)
sd(happy)
pdf("happy_hist.pdf")
h<-hist(happy, xlab="Happiness Score")
dev.off()

#PLOTS
pdf("econ_plot.pdf")
p<-plot(econ, happy, xlab="Economy", ylab="Happiness")
dev.off()


pdf("fam_plot.pdf")
p<-plot(family, happy, xlab="Family", ylab="Happiness")
dev.off()


pdf("health_plot.pdf")
p<-plot(health, happy, xlab="Health", ylab="Happiness")
dev.off()

pdf("free_plot.pdf")
p<-plot(freedom, happy, xlab="Freedom", ylab="Happiness")
dev.off()

pdf("gov_plot.pdf")
p<-plot(govern, happy, xlab="Government", ylab="Happiness")
dev.off()

#Determine Split
n<-nrow(data)
n
split_point<-round(n*0.66)

#take random sample
data<-data[sample(n),]

train<-data[1:split_point,]
test<-data[split_point:n,]

#PREDICT
econ.lm <-lm(happiness_score~economy , data=train)
coef(econ.lm)
gimme_predictions(econ.lm)

fam.lm<-lm(happiness_score~family, data=train)
coef(fam.lm)
gimme_predictions(fam.lm)

h.lm<-lm(happiness_score~health, data=train)
coef(h.lm)
gimme_predictions(h.lm)

free.lm<-lm(happiness_score~freedom, data=train)
coef(free.lm)
gimme_predictions(free.lm)


suma.lm<-lm(happiness_score~economy+family+health, data=train)
coef(suma.lm)
gimme_predictions(suma.lm)

all.lm<-lm(happiness_score~., data=train)
coef(all.lm)



X<-data.matrix(data[3:5])
Y<-data.matrix(data[6])
X_T<-t(X)
 b_hat<- solve( X_T %*% X) %*% X_T  %*% Y 
 b_hat
