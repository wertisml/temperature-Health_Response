# reset
rm(list=ls(all=TRUE))
# library
library(data.table)
library(dplyr)
library(psych)
library(caret)
library(gam)
library(RSNNS)
library(e1071)
library(ranger)
library(randomForest)
library(ggpubr)
library(xgboost)
library(tidymodels)

#==============================================================================#
# Input data
#==============================================================================#

Data <- fread("C:\\Users\\owner\\Documents\\Thesis_Documents\\Population_Data\\North_Carolina_sheps_temp_1.csv")

Data <- Data %>%
  filter(RUCA1 == 1) %>%
  mutate(Total_Pop = (exp(log_Total_Pop_per1000))*1000)

# Remove the NA rows
Data <- Data[complete.cases(Data),]

Data$Mental_Health <- as.numeric(Data$Mental_Health)

#==============================================================================#
# Split data for training
#==============================================================================#

set.seed(24)                                                                                                  
split <- rsample::initial_split(Data,
                                prop = 0.8, 
                                strata = Mental_Health)

Train <- training(split)
Test <- testing(split)

#==============================================================================#
# define functions
#==============================================================================#

RMSE <- function(Obs, Pred){
	Dif <- Pred - Obs
	RMSE <- round(sqrt(mean(Dif**2)), 2)
	return(RMSE)
}

MAE <- function(Obs, Pred){
	Dif <- Pred - Obs
	MAE <- Dif %>% abs() %>% mean()	%>% round(., 2)
	return(MAE)
}

MAPE <- function(Obs, Pred){
	Dif <- Pred - Obs
	MAPE <- mean(abs(Dif/Obs)*100) %>% round(., 2)
	return(MAPE)
}

#==============================================================================#
# Performance All Mental_Health in Train and Test
#==============================================================================#

setwd("~/Thesis_Documents/performance_outputs/Cities")
Models <- c("GLM_1.rds",
            "GAM_1.rds"#,
            #"RF_ranger_1.rds",
            #"XGBoost_1.rds"
            )

# get model list
Model_list <- Models

# Save performance
Hoge <- c()

# loop
for(iii in Model_list){

# load models
Model <- readRDS(iii)

# get predicted values
Train$Pred <- if(iii == "glm_uni.rds"){Train$Pred <- predict(Model, Train, type = "response")}else{Train$Pred <- predict(Model, Train)}
Train$Pred <- ifelse(Train$Pred < 0, 0, Train$Pred)

# get performance
RMSE_train <- RMSE(Obs = Train$Mental_Health, Pred = Train$Pred)

MAE_train <- MAE(Obs = Train$Mental_Health, Pred = Train$Pred)

MAPE_train <- MAPE(Obs = Train$Mental_Health, Pred = Train$Pred)

Cor_train <- paste(
	cor.test(Train$Mental_Health, Train$Pred)$estimate %>% round(., 2),
	" (",
	cor.test(Train$Mental_Health, Train$Pred)$conf.int[1] %>% round(., 2),
	" to ",
	cor.test(Train$Mental_Health, Train$Pred)$conf.int[2] %>% round(., 2),
	")",
	sep = ""
	)

# get predicted values
Test$Pred <- if(iii == "glm_uni.rds"){Test$Pred <- predict(Model, Test, type = "response")}else{Test$Pred <- predict(Model, Test)}
Test$Pred <- ifelse(Test$Pred < 0, 0, Test$Pred)

# get performance
RMSE_test <- RMSE(Obs = Test$Mental_Health, Pred = Test$Pred)

nRMSE_test <- round(RMSE(Obs = Test$Mental_Health, Pred = Test$Pred) / mean(Test$Mental_Health), 3)

MAE_test <- MAE(Obs = Test$Mental_Health, Pred = Test$Pred)

nMAE_test <- round(MAE(Obs = Test$Mental_Health, Pred = Test$Pred) / mean(Test$Mental_Health), 3)

Cor_test <- paste(
	cor.test(Test$Mental_Health, Test$Pred)$estimate %>% round(., 2),
	" (",
	cor.test(Test$Mental_Health, Test$Pred)$conf.int[1] %>% round(., 2),
	" to ",
	cor.test(Test$Mental_Health, Test$Pred)$conf.int[2] %>% round(., 2),
	")",
	sep = ""
	)

# Summarize
Performance <- c(iii,
                 RMSE_train,
                 RMSE_test,
                 MAE_train,
                 MAE_test)

Hoge <- cbind(Hoge, Performance)	

}

Hoge <- data.table(Hoge)
Hoge

fwrite(Hoge, "RMSE.csv")

#==============================================================================#
# Variable Importance
#==============================================================================#

# get model list
Model_list <- Models

# loop
for(iii in Model_list){
  
  # load models
  Model <- readRDS(iii)
  
  varimp_data <- data.frame(feature = row.names(varImp(Model))[1:5],
                            importance = varImp(Model)[1:5, 1])
  
  varimp_data$feature <- factor(varimp_data$feature, 
                                levels = varimp_data$feature[order(varimp_data$importance, 
                                                                   decreasing = FALSE)])
  
  ggplot(data = varimp_data, aes(x = feature, y = importance)) +
    labs(x = "Features", y = "Variable Importance") + 
    geom_point(stat="identity") +
    coord_flip() + 
    theme_bw() + theme(legend.position = "none")
  ggsave(paste("", iii, "_variable_importance.png", sep = ""), width = 4.2, height = 3.2) #*action
  
}

#==============================================================================#
# Figure time series by 24h
#==============================================================================#

# get model list
Model_list <- Models

# Save performance
Hoge <- c()

# Roop
for(iii in Model_list){

  
# iii <- Models[1]
# load models
Model <- readRDS(iii)

# get predicted values
Train$Pred <- if(iii == "glm_uni.rds"){Train$Pred <- predict(Model, Train, type = "response")}else{Train$Pred <- predict(Model, Train)}
Train$Predicted <- ifelse(Train$Pred < 0, 0, Train$Pred)

Train$Obserbved <- Train$Mental_Health
Train$Predicted <- round(Train$Predicted) #This might need to be removed
Train$Dif <- Train$Predicted - Train$Obserbved


# get predicted values
Test$Pred <- if(iii == "glm_uni.rds"){Test$Pred <- predict(Model, Test, type = "response")}else{Test$Pred <- predict(Model, Test)}
Test$Predicted <- ifelse(Test$Pred < 0, 0, Test$Pred)

Test$Obserbved <- Test$Mental_Health
Test$Predicted <- round(Test$Predicted) #This might need to be removed
Test$Dif <- Test$Predicted - Test$Obserbved


#==============================================================================#
# Make figures
#==============================================================================#

# # Make figures train
# ggplot(data = Train, aes(x = Predicted, y = Obserbved)) +
# 	geom_point() + 
# 	geom_smooth(method = lm) + 
# 	scale_x_continuous(limits = c(0, 10), breaks = seq(0, 10, 1)) + 
# 	scale_y_continuous(limits = c(0, 20), breaks = seq(0, 10, 1)) + 
#   xlab("Predicted") + 
#   ylab("Actual") +
#   stat_cor(label.y = 10, 
#            aes(label = paste(..rr.label.., sep = "~`,`~"))) +
#   stat_regline_equation(label.y = 9.5) +
# 	theme_classic() 
# ggsave(paste("Out_plot_", iii, "_train.png", sep = ""), width = 4.2, height = 3.2) #*action
# 
# # Make figures test
# ggplot(data = Test, aes(x = Predicted, y = Obserbved)) +
# 	geom_point() + 
# 	geom_smooth(method = lm) + 
# 	scale_x_continuous(limits = c(0, 15), breaks = seq(0, 15, 5)) + 
# 	scale_y_continuous(limits = c(0, 25), breaks = seq(0, 25, 5)) + 
#   xlab("Predicted") + 
#   ylab("Actual") +
#   stat_cor(label.y = 20, 
#            aes(label = paste(..rr.label.., sep = "~`,`~"))) +
#   stat_regline_equation(label.y = 18.5) +
# 	theme_classic() 
# ggsave(paste("Out_plot_", iii, "_test.png", sep = ""), width = 4.2, height = 3.2) #*action
# 
# # Make Fig train
# DataSum_train <- Train %>%
#   filter(Date > "2016-01-01") %>%
# 	group_by(Date) %>%
# 	summarise(Obserbved = sum(Obserbved),
# 		Predicted = sum(Predicted)) %>%
# 	data.table() %>%
# 	print()


DataSum_test <- Test %>%
  filter(Date > "2016-01-01") %>%
	group_by(Date) %>%
	summarise(
		Obserbved = sum(Obserbved),
		Predicted = sum(Predicted)
		) %>%
	data.table() %>%
	print()
# 
# DataSum_train[, Date:=as.Date(Date), ]
# DataSum_train[, YearUse:=year(Date), ]
# DataSum_train <- DataSum_train[order(Date), , ]
# DataSum_train[, Day:=1:length(Obserbved), by = YearUse]
# DataSum_train
# 
# 
# DataSum_train[Date == as.Date("2016-06-01"), , ]
# DataSum_train[Date == as.Date("2016-07-01"), , ]
# DataSum_train[Date == as.Date("2016-08-01"), , ]
# 
# 
# ggplot(data = DataSum_train, aes(x = Day), group=factor(YearUse)) +
#   geom_line(aes(y = Obserbved, x=Day), colour = "Black", size = 0.4) + 
#   geom_line(aes(y = Predicted, x=Day), colour = "Red", size = 0.4) + 
#   xlab("") + 
#   ylab("") +
#   scale_y_continuous(limits = c(0, 140), 
#                      breaks = seq(0, 140, 10)) + 
#   scale_x_continuous(label = c("Jun.", "Jul.", "Aug."),    
#                      breaks = c(1, 31, 62)) + 
#   theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
#   facet_grid(~YearUse) +
#   theme_classic()

#==============================================================================#
#
#==============================================================================#

DataSum_test[, Date:=as.Date(Date), ]
DataSum_test[, YearUse:=year(Date), ]
DataSum_test <- DataSum_test[order(Date), , ]
DataSum_test[, Day:=1:length(Obserbved), by = YearUse]
DataSum_test

DataSum_test[Date == as.Date("2016-06-01"), , ]
DataSum_test[Date == as.Date("2016-07-01"), , ]
DataSum_test[Date == as.Date("2016-08-01"), , ]

#GLM <- 
ggplot(data = DataSum_test, aes(x = Day), group=factor(YearUse)) +
  geom_line(aes(y = Obserbved, x=Day, colour = "Obserbved"), size = 0.75) + 
  geom_line(aes(y = Predicted, x=Day, colour = "Predicted"), size = 0.75) + 
  scale_colour_manual("",
                      breaks = c("Obserbved", "Predicted"),
                      values = c("black", "red")) +
  xlab("") + 
  ylab("")+
  scale_y_continuous(limits = c(0, 100),
                     breaks = seq(0, 100, 20)) + 
  scale_x_continuous(label = c("Jun.", "Jul.", "Aug."),
                     breaks = c(1, 31, 62)) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  facet_grid(~YearUse) +
  theme_classic() 

#ggsave(paste("",iii,  ".png", sep = ""), width = 6.7 * 1.5, height = 3.5 * 0.66) #*action

# setwd("~/Thesis_Documents/performance_outputs/Cities")
# png(file = 'Actual_vs_Predicted.png', width = 10, height = 6, units = 'in', res = 600)
# grid.arrange(GLM, GAM, RF, XGBoost, ncol = 1, padding = 20)
# dev.off()

#fwrite(DataSum, paste("Out_data_figure1_", iii, ".csv", sep = ""))

}

Hoge <- data.table(Hoge)
Hoge

#==============================================================================#
# Bonus Stuff
#==============================================================================#

# formula
Form_temp <- formula( 	
  Mental_Health ~ 
    offset(log_Total_Pop_per1000)  
  #+ Median_Age  
  #+ Pop_5_24_per1000 
  + Male_to_Female_Ratio 
  + RUCA1 
  + Region
  + Income 
  + Race 
  
  + Day 
  + month 
  + NDVI 
  
  #+ TAVG  
  + TMIN 
  + TMAX 
  #+ TAVGLag1 
  #+ TMINLag1 
  #+ TMAXLag1 
  #+ TAVG_24hr_diff 
  + TMIN_24hr_diff 
  + TMAX_24hr_diff 
  #+ Daily_Difference 
  + EHF 
  + RH 
  #+ Heat_Index 
  #+ Discomfort_Index
)

#==============================================================================#
# Variance Inflation Factors
#==============================================================================#

Model_vif <- glm(family = "poisson",
                 Form_temp,
                 data = Train)

summary(Model_vif)
vif(Model_vif)

#==============================================================================#
# Correlation Matrix
#==============================================================================#

my_data <- Data[, c("Male_to_Female_Ratio", "RUCA1", "Region", "Income", "Race",
                    "Day", "month", "NDVI", "TMIN", "TMAX", "TMAX_24hr_diff", 
                    "EHF", "RH")]
res <- cor(my_data)
corrplot(res, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)

#==============================================================================#
# Plotting Cool Stuff
#==============================================================================#

Cases_data <- Data %>%
  mutate(Year = as.factor(year(Date))) %>%
  select(Mental_Health, Year)

# Box plot of cases per year
Tree_intensity_long <- melt(Cases_data, id = "Year") 

ggplot(Tree_intensity_long, aes(x = variable, y = value, fill = Year)) +  # ggplot function
  geom_boxplot(outlier.alpha = 0.5)+
  coord_flip() +
  theme_bw() +
  labs(fill = NULL, x = NULL,
       y = "Case breakdown by year",
       title = "Frequency of any mental health ICD code") +
  theme(plot.title = element_text(hjust = 0.5))

# Region
Cases_data <- Data %>%
  mutate(Year = as.factor(year(Date))) %>%
  select(Mental_Health, Region) 

Cases_data$Region <- as.factor(Cases_data$Region)

# Box plot of cases per year
Tree_intensity_long <- melt(Cases_data, id = "Region") 

ggplot(Tree_intensity_long, aes(x = variable, y = value, fill = Region)) +  # ggplot function
  geom_boxplot(outlier.alpha = 0.5)+
  coord_flip() +
  theme_bw() +
  labs(fill = NULL, x = NULL,
       y = "Case breakdown by Region",
       title = "Frequency of any mental health ICD code") +
  theme(plot.title = element_text(hjust = 0.5))

#==============================================================================#

Datar <- Data %>%
  group_by(Date) %>%
  summarize(Mental_Health = sum(Mental_Health)) %>%
  mutate(Day = 1:length(Mental_Health),
         Year = as.character(year(Date)),
         month = as.character(month(Date))) %>% 
  select(Date, Mental_Health, Day, Year, month)

# Mental Health
ggplot(data = Datar, aes(x = Mental_Health, color = Year)) +
  geom_line(aes(y = Mental_Health, x = Day), size = 0.4) +
  theme_classic() 

#==============================================================================#

Temp <- Data %>%
  group_by(Region, RUCA1) %>%
  summarize(mean_TAVG = mean(TAVG),
            meadian_TAVG = median(TAVG))
