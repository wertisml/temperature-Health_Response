library(tidymodels)
library(vip)
library(data.table)
library(dplyr)
library(ggpubr)

#==============================================================================#
# Input data
#==============================================================================#

Data <- fread("/home/students/wertisml/Thesis/Mental_Health_data/North_Carolina_sheps_temp.csv")

# Data <- fread("/home/students/wertisml/Thesis/Mental_Health_data/Sheps_temp_ML.csv")
# Population <- fread("/home/students/wertisml/Thesis/Mental_Health_data/Mountain_sociodemographic.csv")
# NDVI <- fread("/home/students/wertisml/Thesis/Mental_Health_data/NDVI_Combined.csv")

#==============================================================================#
# Set up parallel
#==============================================================================#

n.cores <- parallel::detectCores() - 1

unregister <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}

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
#    filter(Pop_5_24 >= 1) %>%
#    filter(RUCA1 <= 3) %>%
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
# Preprocessing "recipe" 
#==============================================================================#

preprocessing_recipe <- recipes::recipe(Form_temp, data = Train) %>%
  # convert categorical variables to factors (One hot encoding)
  recipes::step_string2factor(all_nominal()) %>%
  # combine low frequency factor levels
  recipes::step_other(all_numeric_predictors(), threshold = 0.01) %>%
  # remove no variance predictors which provide no predictive information 
  recipes::step_nzv(all_numeric_predictors())

tree_prep <- prep(preprocessing_recipe)
juiced <- juice(tree_prep)

#==============================================================================#
# Build the Random Forest model (Round 1)
#==============================================================================#

# Cross fold validation
set.seed(24)
trees_folds <- vfold_cv(Train, v = 5)

# Build the model frame
rf_spec <- rand_forest(mtry = tune(),
                       trees = tune(),
                       min_n = tune()) %>%
  set_mode("regression") %>%
  set_engine("ranger")

# Set up the workflow for the model to follow
tune_wf <- workflow() %>%
  add_recipe(preprocessing_recipe) %>%
  add_model(rf_spec)

#create and register cluster
my.cluster <- parallel::makeCluster(n.cores)
doParallel::registerDoParallel(cl = my.cluster)

# Tune the hyperparamters
set.seed(24)
tune_res <- tune_grid(tune_wf,
                      resamples = trees_folds,
                      grid = 50)

# This turns off the parallel
unregister()

# Visually determine the trend in the data to identify mtry and min_n to focus on
tune_res %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  pivot_longer(mtry:min_n,
               values_to = "value",
               names_to = "parameter") %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "RMSE")

tune_res %>%
  show_best("rmse")

#==============================================================================#
# Fine tune the Random Forest model (Round 2)
#==============================================================================#

# Create grid that is more centered around the best hyperparameters
rf_grid <- expand.grid(min_n = seq(30, 35, 2),
                       mtry = seq(6,6,1),
                       trees = seq(251, 1501, 250))
#mtry = 2, trees = 501, min_n = 32

#create and register cluster
my.cluster <- parallel::makeCluster(n.cores)
doParallel::registerDoParallel(cl = my.cluster)

# Run the model again with the new grid
set.seed(24)
regular_res <- tune_grid(tune_wf,
                         resamples = trees_folds,
                         grid = rf_grid)

# This turns off the parallel
unregister()

regular_res %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  pivot_longer(mtry:min_n,
               values_to = "value",
               names_to = "parameter") %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "RMSE")

#==============================================================================#
# Set up final model (Round 3)
#==============================================================================#

best_auc <- select_best(regular_res, "rmse")

final_rf <- finalize_model(rf_spec, best_auc)

#==============================================================================#
# Run model for RDS
#==============================================================================#

# Set-up parallel
core <- detectCores(all.tests = FALSE, logical = TRUE)
cl <- makePSOCKcluster(core - 1)
registerDoParallel(cl)

set.seed(1234)

# CV methods
CV_method <- "cv"
CV_number <- 5
Repeated_number <- 1
Tune_number <- 1

# Set seed
Size_n <- 19
Seeds_fix <- vector(mode = "list", length = CV_number * Repeated_number + 1)
for(i in 1:(CV_number * Repeated_number)) Seeds_fix[[i]] <- sample.int(n = 10000, size = Size_n )
Seeds_fix[[(CV_number * Repeated_number + 1)]] <- sample.int(10000, 1)
Seeds_fix

# Construct rfeControl object
rfe_control = rfeControl(
  functions = rfFuncs, 
  allowParallel = TRUE,
  method = CV_method,
  number = CV_number,
  returnResamp = "final",
  seeds = Seeds_fix)

# construct trainControl object for train method 
fit_control = trainControl(
  allowParallel = TRUE,
  method = CV_method,
  number = CV_number,
  seeds = Seeds_fix)

rf_grid <- expand.grid(min_n = best_auc$min_n, #34
                       mtry = best_auc$mtry,  #6
                       trees = best_auc$trees) #1251

RF_temp <- rfe(form = Form_temp,
               data = Train,
               trControl = fit_control,
               metric = "RMSE", 
               preProcess = c("center", "scale"),
               tuneGrid = rf_grid,
               sizes = 1:13, 
               rfeControl = rfe_control)

saveRDS(RF_temp, "RF_ranger.rds")
