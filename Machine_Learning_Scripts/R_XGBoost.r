rm(list=ls(all=TRUE))

library(data.table)
library(dplyr)
library(ggplot2)
library(car)
library(psych)
library(pROC)
library(mgcv)
library(epitools)
library(tableone)
library(caret)
library(tictoc)
library(Boruta)
library(doParallel)

#==============================================================================#
# Input data
#==============================================================================#

Data <- fread("/home/students/wertisml/Thesis/Mental_Health_data/North_Carolina_sheps_temp.csv")

# Data <- fread("/home/students/wertisml/Thesis/Mental_Health_data/Sheps_temp_ML.csv")
# Population <- fread("/home/students/wertisml/Thesis/Mental_Health_data/NC_sociodemographic.csv")
# NDVI <- fread("/home/students/wertisml/Thesis/Mental_Health_data/NDVI_Combined.csv")

#==============================================================================#
# Set up parallel
#==============================================================================#

core <- detectCores(all.tests = FALSE, logical = TRUE)
cl <- makePSOCKcluster(core - 1)
registerDoParallel(cl)

set.seed(1234)

#==============================================================================#
# Clean Data
#==============================================================================#

# Data <- Data %>%
#   mutate(month = month(admitdt)) %>%
#   filter(month >= 6, month <= 8, admitdt > "2016-01-01", admitdt < "2020-01-01") %>% #only looking at the months June thru September
#   group_by(admitdt, ZCTA, TAVG, TMIN, TMAX, RH, EHF,
#            Heat_Index, Discomfort_Index, Respiratory) %>%
#   summarize(Mental_Health = sum(Mental_Health), .groups = "keep") %>%
#   dplyr::select(admitdt, ZCTA, TAVG, TMIN, TMAX, EHF, RH,
#                 Heat_Index, Discomfort_Index, Mental_Health, Respiratory) %>%
#   as_tibble()
# 
# Data <- Population %>%
#   left_join(Data, by = c("ZCTA" = "ZCTA"))
# 
# NDVI$Zip <- as.numeric(NDVI$Zip)
# 
# Data <- Data %>%
#   left_join(NDVI, by = c("admitdt" = "Date", "ZIP"="Zip"))
# 
# Data <- Data %>%
#   group_by(ZIP) %>%
#   rename(Date = admitdt) %>%
#   mutate(Year = year(Date),
#          month = month(Date),
#          Day = wday(Date),
#          Daily_Difference = TMAX - TMIN,
#          TAVGLag1 = lag(TAVG, order_by = Date),
#          TMAXLag1 = lag(TMAX, order_by = Date),
#          TMINLag1 = lag(TMIN, order_by = Date),
#          TAVG_24hr_diff = round(TAVG - TAVGLag1, 2),
#          TMAX_24hr_diff = round(TMAX - TMAXLag1, 2),
#          TMIN_24hr_diff = round(TMIN - TMINLag1, 2),
#          cases_per = Mental_Health / Pop_5_24_per1000) %>%
#   filter(Pop_5_24 >= 1) %>%
#   filter(RUCA1 <= 3) %>%
#   select(-ZCTA)

# Remove the NA rows
Data <- Data[complete.cases(Data),]

colnames(Data)
dim(Data)

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
#Prediction settings
#==============================================================================#

# CV methods
CV_method <- "cv"
CV_number <- 5
Repeated_number <- 1
Tune_number <- 1

# formula
Form_temp <- formula( 	
  Mental_Health ~ 
    log_Total_Pop_per1000
    + Median_Age  
  + Pop_5_24_per1000 
  + Male_to_Female_Ratio 
  + RUCA1 
  + Income 
  + Race 
  + Region
  
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
# XGBTree Tune Round 1
#==============================================================================#

Size_n <- 72  ### set seed
Seeds_fix <- vector(mode = "list", length = CV_number * Repeated_number + 1)
for(i in 1:(CV_number * Repeated_number)) Seeds_fix[[i]] <- sample.int(n = 10000, size = Size_n )
Seeds_fix[[(CV_number * Repeated_number + 1)]] <- sample.int(10000, 1)
Seeds_fix
					 
# construct rfeControl object
rfe_control = rfeControl(functions = caretFuncs, 
                         allowParallel = TRUE,
                         method = CV_method,
                         number = CV_number,
                         returnResamp = "final",
                         seeds = Seeds_fix
	                      )


# construct trainControl object for train method 
fit_control = trainControl(allowParallel = TRUE,
                           method = CV_method,
                           number = CV_number,
                           seeds = Seeds_fix
	                        )

# Hyper parameter grid

tune_params = expand.grid(nrounds = seq(1, 151, 50), 
                          max_depth = seq(3, 9, 3),
                          min_child_weight = seq(1, 5, 2),
                          gamma = seq(0, 0.4, 0.4),
                          colsample_bytree = seq(0.6, 1, 0.4),
                          subsample = seq(0.6, 1, 0.4),
                          eta = 0.1
                         ) %>% as.data.frame

# xgbTree
tic()
xgbTree_temp <- train(form = Form_temp,
                      data = Train,
                      method = "xgbTree",
                      objective="reg:squarederror",
                      trControl = fit_control,
                      metric = "RMSE", 
                      preProcess = c("center", "scale"),
                      tuneGrid = tune_params
	                   )

summary(xgbTree_temp)
xgbTree_temp
toc()

plot(xgbTree_temp)

#==============================================================================#
# XGBTree Tune Round 2
#==============================================================================#

Size_n <- 324  ### set seed
Seeds_fix <- vector(mode = "list", length = CV_number * Repeated_number + 1)
for(i in 1:(CV_number * Repeated_number)) Seeds_fix[[i]] <- sample.int(n = 10000, size = Size_n )
Seeds_fix[[(CV_number * Repeated_number + 1)]] <- sample.int(10000, 1)
Seeds_fix

# construct rfeControl object
rfe_control = rfeControl(functions = caretFuncs, 
                         allowParallel = TRUE,
                         method = CV_method,
                         number = CV_number,
                         returnResamp = "final",
                         seeds = Seeds_fix
	                      )

# construct trainControl object for train method 
fit_control = trainControl(allowParallel = TRUE,
                           method = CV_method,	
                           number = CV_number,
                           seeds = Seeds_fix
	                        )

# Tune Hyper parameters
tune_params = expand.grid(nrounds = seq(76, 76, 25), 
                          max_depth = seq(7, 8, 1),
                          min_child_weight = seq(5, 5, 1),
                          gamma = seq(0.3, 0.5, 0.1),
                          colsample_bytree = seq(0.6, 0.6, 0.1),
                          subsample = seq(0.8, 1.0, 0.1),
                          eta = seq(0.1, 0.1, 1)
                         ) %>% as.data.frame

# xgbTree
tic()
xgbTree_temp <- train(form = Form_temp,
                      data = Train,
                      method = "xgbTree",
                      objective="reg:squarederror",
                      trControl = fit_control,
                      metric = "RMSE", 
                      preProcess = c("center", "scale"),
                      tuneGrid = tune_params
                     )

summary(xgbTree_temp)
xgbTree_temp
toc()

plot(xgbTree_temp)

#==============================================================================#
# Run final XGBoost Model
#==============================================================================#

### set seed
Size_n <- 41
Seeds_fix <- vector(mode = "list", length = CV_number * Repeated_number + 1)
for(i in 1:(CV_number * Repeated_number)) Seeds_fix[[i]] <- sample.int(n = 10000, size = Size_n )
Seeds_fix[[(CV_number * Repeated_number + 1)]] <- sample.int(10000, 1)
Seeds_fix

# construct rfeControl object
rfe_control = rfeControl(functions = caretFuncs, 
                         allowParallel = TRUE,
                         method = CV_method,
                         number = CV_number,
                         returnResamp = "final",
                         seeds = Seeds_fix
	                      )

# construct trainControl object for train method 
fit_control = trainControl(allowParallel = TRUE,
                           method = CV_method,
                           number = CV_number,
                           seeds = Seeds_fix
	                       )
	
# Create grid of optimal parameters
tune_params = expand.grid(nrounds = xgbTree_temp$bestTune$nrounds, 
                          max_depth = xgbTree_temp$bestTune$max_depth,
                          min_child_weight = xgbTree_temp$bestTune$min_child_weight,
                          gamma = xgbTree_temp$bestTune$gamma,
                          colsample_bytree = xgbTree_temp$bestTune$colsample_bytree,
                          subsample = xgbTree_temp$bestTune$subsample,
                          eta = xgbTree_temp$bestTune$eta
                         ) %>% as.data.frame

# Run final model
tic()
xgbTree_temp <- rfe(form = Form_temp,
                    data = Train,
                    method = "xgbTree",
                    objective="reg:squarederror",
                    trControl = fit_control,
                    metric = "RMSE", 
                    preProcess = c("center", "scale"),
                    tuneGrid = tune_params,
                    sizes = 1:20, 
                    rfeControl = rfe_control
                   )

summary(xgbTree_temp)
xgbTree_temp
toc()

xgbTree_temp$optVariables

saveRDS(xgbTree_temp, "xgboost_Model.rds")





