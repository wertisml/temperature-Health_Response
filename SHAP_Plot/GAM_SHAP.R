rm(list=ls(all=TRUE))

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
library(tidymodels)
library(DALEX)
library(iBreakDown)
library(doParallel)
library(forcats)
library(gam)

#==============================================================================#
# Input data
#==============================================================================#

Data <- fread("C:\\Users\\owner\\Documents\\Thesis_Documents\\Population_Data\\North_Carolina_sheps_temp_1.csv")

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

Data <- Data %>%
  mutate(Total_Pop = (exp(log_Total_Pop_per1000))*1000) 

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
# SHAP
#==============================================================================#

Train <- Train[order(Date), , ]
Train[, ID_SHAP := 1:nrow(Train), ]
Train <- data.frame(Train)
Test <- data.frame(Test)

setwd("~/Thesis_Documents/performance_outputs/Cities")
Model <- readRDS("GAM_1.rds")

optimal_vars <- Model[6]
optimal_vars <- c(optimal_vars$optVariables)

Select_predictor <- Model[19]
Select_predictor <- c("log_Total_Pop_per1000", Select_predictor$coefnames)
Select_predictor

### object for SHAP
library(gam)

explain_Model <- DALEX::explain(Model,
                                data = Train[, Select_predictor],
                                y = Train$Mental_Health,
                                label = "gam")

Obs <- Train$Mental_Health
Pred <- explain_Model$y_hat

RMSE <- function(Obs, Pred){
  Dif <- Pred - Obs
  RMSE <- round(sqrt(mean(Dif**2)), 6)
  return(RMSE)
}

RMSE(Obs, Pred)

### loop

tic()

Hoge <- c()

Select_ID_SHAP <- sample(Train$ID_SHAP, size = 293)

Hoge <- foreach(iii = Select_ID_SHAP, .combine = "rbind") %do% {
  SHAP_model <- shap(explain_Model, subset(Train, ID_SHAP == iii), B = 5)
  
  Kari <- SHAP_model %>% 
    data.table()
  
  Kari[, ID := iii, ]
  Kari
  
}

toc()
stopCluster(cl)

fwrite(Hoge, "SHAP_gam.csv")

#==============================================================================#
# Summarise SHAP
#==============================================================================#

SHAP_data <- Hoge %>% 
  group_by(ID, variable_name) %>%
  summarize(Variable_value = first(variable_value),
            Contribution = mean(contribution)) %>%
  data.table() %>% 
  print()

SHAP_data[, Contribution := as.numeric(Contribution), ]
SHAP_data[, Variable_value := as.numeric(Variable_value), ]

SHAP_data[, Variable_value_scale := scale(Variable_value), by = variable_name]

SHAP_data %>% 
  group_by(variable_name) %>%
  summarize(mean(Variable_value_scale),
            sd(Variable_value_scale)) 

SHAP <- SHAP_data %>%
  filter(variable_name %in% optimal_vars) %>%
  group_by(variable_name) %>%
  mutate(mean_value = mean(abs(Contribution)))

#==============================================================================#
# Plot SHAP
#==============================================================================#

Plot_data <- SHAP[ , , ]

myfuns <- list(Low = min, High = max)
ls_val <- unlist(lapply(myfuns, function(f) f(Plot_data$Contribution)))

ggplot(data = Plot_data) +
  coord_flip() +
  ggforce::geom_sina(aes(x = fct_reorder(variable_name, mean_value), y = Contribution, color = Variable_value_scale),
                     method = "counts", alpha = 3) + 
  scale_color_gradient(low = "#FFCC33", high = "#6600CC") +
  geom_text(data = unique(Plot_data[, c("variable_name", "mean_value")]),
            aes(x = variable_name, y=-Inf, label = round(mean_value,3)),
            size = 3, alpha = 0.7,
            hjust = -0.2,
            fontface = "bold",
            check_overlap = TRUE) +
  theme_bw() +
  ylim(1.15*(ls_val)) +
  scale_color_gradient(low="#FFCC33", high="#6600CC",
                       breaks=ls_val, 
                       guide = guide_colorbar(barwidth = 12, barheight = 0.3)) +
  theme(axis.line.y = element_blank(),
        axis.ticks.y = element_blank(), # remove axis line
        legend.position="bottom",
        legend.title=element_text(size=10),
        legend.text=element_text(size=8),
        axis.title.x= element_text(size = 10)) +
  labs(y = "SHAP value (impact on model output)", x = "", color = "Feature value")

ggsave(paste("SHAP_values_", "Raleigh", ".png", sep = ""), width = 4.2, height = 3.2)

#==============================================================================#
# Plot individual SHAP Plots
#==============================================================================#

features <- Plot_data %>%
  filter(variable_name %in% optimal_vars) %>%
  rename(variable = variable_name,
         rfvalue = Variable_value,
         value = Variable_value_scale) %>%
  group_by(variable) %>%
  arrange(rfvalue) %>%
  mutate(stdfvalue = cumsum(c(0,as.numeric(diff(rfvalue))!=0))/length(unique(rfvalue)),
         mean_value = mean(abs(Contribution))) %>% #This is the SHAP value
  select(ID, variable, value, rfvalue, stdfvalue, mean_value) %>%
  data.table()

for(i in 1:length(optimal_vars)) {
  
  shap.plot.dependence(features, optimal_vars[i], color_feature = "auto", 
                       alpha = 0.5, jitter_width = 0.1)
  
  ggsave(paste("shap_plot_", optimal_vars[i], ".png", sep = ""), width = 4.2, height = 3.2)
  
}

#==============================================================================#
#
#==============================================================================#

new_observation <- Train %>%
  summarise(Median_Age = mean(Median_Age),
            Pop_5_24_per1000 = mean(Pop_5_24_per1000),
            Male_to_Female_Ratio = mean(Male_to_Female_Ratio),
            loc = mean(loc),
            RH = mean(RH),
            Day = mean(Day),
            TMIN_24hr_diff = mean(TMIN_24hr_diff),
            TMAX_24hr_diff = mean(TMAX_24hr_diff),
            NDVI = mean(NDVI),
            TMAX = mean(TMAX),
            log_Total_Pop_per1000 = mean(log_Total_Pop_per1000),
            TMIN = mean(TMIN),
            EHF = mean(EHF),
            Above_95th = mean(Above_95th),
            month = mean(month))


train <- Train[, Select_predictor]
test <- Test[, Select_predictor]

exp_gam <- DALEX::explain(Model, data = train)
ive_gam <- iBreakDown::shap(exp_gam, new_observation = test)
plot(ive_gam)

bd_gam <- break_down(exp_gam, test, keep_distributions = TRUE)
plot(bd_gam)
description <- iBreakDown::describe(bd_gam,
                                    label = "the total mental or behaviroal disorder a city will experience is",
                                    short_description = FALSE,
                                    display_values = TRUE,
                                    display_numbers = TRUE,
                                    display_distribution_details = FALSE)
description 

  