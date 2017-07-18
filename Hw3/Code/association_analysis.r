library(arules)
library(arulesViz)

#Load
dataset <- read.csv("../Data/survey_dataset.csv")
# dataset <- read.csv("/Users/panos/Desktop/UCSC/Spring_2017/Data\ Mining/Hw3/Data/survey_dataset.csv")

#Convert dataset to nominal attributes 
dataset <- data.frame(sapply(dataset, function(x) as.factor(as.character(x))))

#set number of rules to examine 
n = 200 
support <- 0.1
confidence <- 0.1

#generate rules
rules <- apriori(dataset, parameter= list(sup=support, conf=confidence))

#prune redundant rules: i.e. rules for which anothe more general rule exists with the same or higher confidence 
rules.pruned= rules[!is.redundant(rules)]


#sort by lift and take the n top
rules.pruned.sorted <- sort(rules.pruned, by="lift")
top_n_rules <- head(rules.pruned.sorted, n=n)

#print top rules
inspect(top_n_rules)

#visualize
plot(top_n_rules, method="graph", control=list(type="items"))
