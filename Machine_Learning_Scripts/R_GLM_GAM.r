library(data.table)
library(dplyr)
library(ggplot2)
library(car)
library(psych)
library(pROC)
library(epitools)
library(tableone)
library(caret)
library(tictoc)
library(Boruta)
library(doParallel)
library(tidymodels)
library(corrplot)
library(gam)

#==============================================================================#
# Input data
#==============================================================================#

# Data <- fread("C:\\Users\\owner\\Documents\\Thesis_Documents\\Sheps_temp_ML.csv")
# Population <- fread("C:\\Users\\owner\\Documents\\Thesis_Documents\\Population_Data\\NC_sociodemographic.csv")
# NDVI <- fread("C:\\Users\\owner\\Documents\\Thesis_Documents\\Landcover\\NDVI_Combined.csv")
Data <- fread("C:\\Users\\owner\\Documents\\Thesis_Documents\\Population_Data\\North_Carolina_sheps_temp.csv")

#==============================================================================#
# Set up parallel
#==============================================================================#

detectCores(all.tests = FALSE, logical = TRUE)
cl <- makePSOCKcluster(7)
registerDoParallel(cl)

set.seed(1234)

#==============================================================================#
# Clean Data
#==============================================================================#

# Data <- Data %>%
#   mutate(month = month(admitdt)) %>%
#   filter(month >= 6, month <= 8, admitdt > "2016-01-01", admitdt < "2020-01-01") %>% #only looking at the months June thru September
#   group_by(admitdt, ZCTA, TAVG, TMIN, TMAX, RH, EHF,
#            Heat_Index, Discomfort_Index, Respiratory, month) %>%
#   summarize(Mental_Health = sum(Mental_Health), .groups = "keep") %>%
#   dplyr::select(admitdt, ZCTA, TAVG, TMIN, TMAX, EHF, RH,
#                 Heat_Index, Discomfort_Index, Mental_Health, Respiratory, month) %>%
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

# CV methods
CV_method <- "cv"
CV_number <- 5
Repeated_number <- 1
Tune_number <- 1

# formula
Form_temp <- formula( 	
  Mental_Health ~ 
    offset(log_Total_Pop_per1000)  
    + Median_Age  
    + Male_to_Female_Ratio 
    + Pop_5_24_per1000 
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

my_data <- Data[, c("Male_to_Female_Ratio", "RUCA1", "Region", "Income", 
                    "Race", "Day", "month", "NDVI", "TMIN", "TMAX", "TMIN_24hr_diff",
                    "TMAX_24hr_diff", "EHF", "RH")]

res <- cor(my_data)
corrplot(res, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)

#==============================================================================#
# GLM
#==============================================================================#

modelLookup("glm")

### set seed
Size_n <- 14 # The number of variables in the formula + 1
Seeds_fix <- vector(mode = "list", length = CV_number * Repeated_number + 1)
for(i in 1:(CV_number * Repeated_number)) Seeds_fix[[i]] <- sample.int(n = 10000, size = Size_n )
Seeds_fix[[(CV_number * Repeated_number + 1)]] <- sample.int(10000, 1)
Seeds_fix

# Construct rfeControl object
rfe_control = rfeControl(functions = caretFuncs,
                         allowParallel = TRUE,
                         method = CV_method,
                         number = CV_number,
                         returnResamp = "final",
                         seeds = Seeds_fix)

# Construct trainControl object for train method 
fit_control = trainControl(allowParallel = TRUE,
                           method = CV_method,
                           number = CV_number,
                           seeds = Seeds_fix)

# GLM
tic()
GLM_temp <- rfe(form = Form_temp,
                data = Train,
                method = "glm",
                family = "poisson",
                trControl = fit_control,
                metric = "RMSE", 
                preProcess = c("center", "scale"),
                tuneLength = 10,
                sizes = 1:13, 
                rfeControl = rfe_control)

summary(GLM_temp)
GLM_temp
toc()

GLM_temp$optVariables

setwd("~/Thesis_Documents/performance_outputs/Metro_ZIPs")
saveRDS(GLM_temp, "GLM_Model.rds")

#==============================================================================#
# GAM
#==============================================================================#

# Might have to reload R if you want to run this after running other models.
modelLookup("gamSpline")

# Set seed
Size_n <- 18
Seeds_fix <- vector(mode = "list", length = CV_number * Repeated_number + 1)
for(i in 1:(CV_number * Repeated_number)) Seeds_fix[[i]] <- sample.int(n = 10000, size = Size_n )
Seeds_fix[[(CV_number * Repeated_number + 1)]] <- sample.int(10000, 1)
Seeds_fix

# Construct rfeControl object
rfe_control = rfeControl(functions = caretFuncs, 
                         allowParallel = TRUE,
                         method = CV_method,
                         number = CV_number,
                         returnResamp = "final",
                         seeds = Seeds_fix)

# Construct trainControl object for train method 
fit_control = trainControl(allowParallel = TRUE,
                           method = CV_method,
                           number = CV_number,
                           seeds = Seeds_fix)

# GAMSpline
tic()
gamSpline_temp <- rfe(form = Form_temp,
                      data = Train,
                      method = "gamSpline",
                      family = "poisson",
                      trControl = fit_control,
                      metric = "RMSE", 
                      preProcess = c("center", "scale"),
                      tuneLength = 10,
                      sizes = 1:19, 
                      rfeControl = rfe_control)

summary(gamSpline_temp)
gamSpline_temp
toc()

gamSpline_temp$optVariables

setwd("~/Thesis_Documents/performance_outputs/Metro_ZIPs")
saveRDS(gamSpline_temp, "GAM_Model.rds")
