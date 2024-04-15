
library(dplyr)library(geepack)
library(readxl)
dat = read.csv("~/Desktop/Predicted-context/HeartStepV1/HS1_cleaned_for_simulation.csv")
file_location = "~/Desktop/Predicted-context/HeartStepV1/gee_fit"
  
dat$home.location = as.numeric(dat$work.location == 0 & dat$other.location == 0)
dat$other.location = as.factor(dat$other.location)
dat$work.location = as.factor(dat$work.location)
dat$home.location = as.factor(dat$home.location)
dat$send = as.factor(dat$send)
dat$sd_steps60 = as.factor(dat$sd_steps60)
dat = dat[dat$dec.temperature > -100, ]

dat$sum_step.log = scale(dat$sum_step.log)
dat$jbsteps30pre.log = scale(dat$jbsteps30pre.log)
dat$dec.temperature = scale(dat$dec.temperature)



############################
## GEE
############################

# formula_string = "jbsteps30.log ~ 1 + sum_step.log + jbsteps30pre.log + other.location + work.location + dec.temperature + sd_steps60 + dosage + other.location * send + work.location * send + dosage * send + sd_steps60 * send"


dat = na.omit(dat)
## analyze data with gee
for(u in unique(dat$user)){
  formula_string = "jbsteps30.log ~ 1 + sum_step.log + jbsteps30pre.log  + dec.temperature + sd_steps60 + dosage + dosage * send + sd_steps60 * send"
  dat_tmp = dat%>% filter(user == u)
#  if (length(unique(dat_tmp$other.location)) > 1){
#    formula_string = paste(formula_string, "+ other.location + other.location * send")
#  }
#  if (length(unique(dat_tmp$work.location)) > 1){
#    formula_string = paste(formula_string, "+ work.location + work.location * send")
#  }
  if (sum((dat_tmp$home.location == 1)) > 0.05 * length(dat_tmp$home.location) & sum((dat_tmp$home.location == 0)) > 0.05 * length(dat_tmp$home.location)){
    formula_string = paste(formula_string, "+ home.location + home.location * send")
  }
  formula_touse = formula(formula_string)
  fit_gee <- geeglm(formula_touse,
                  data= dat_tmp,
                  id = user,
                  family = gaussian,
                  corstr = "independence")
  print(paste0("-------------", str(u), "---------"))
  print(summary(fit_gee))
  filename = sprintf("%s/residual_pair%s.csv",file_location, u)
  write.csv(data.frame(fit_gee$residuals), filename, row.names = FALSE)
  filename = sprintf("%s/coeffi_pair%s.csv",file_location, u)
  coef = data.frame(fit_gee$coefficients)
  if(dim(coef)[1] < 11){
     coef = rbind(coef, c(0))
     coef = rbind(coef, c(0))
     rownames(coef) = c("(Intercept)","sum_step.log", "jbsteps30pre.log", "dec.temperature", "sd_steps601", "dosage", "send1",
                         "dosage:send1", "sd_steps601:send1", "send1:home.location1", "home.location1")
     coef$names = c("(Intercept)","sum_step.log", "jbsteps30pre.log", "dec.temperature", "sd_steps601", "dosage", "send1",
                     "dosage:send1", "sd_steps601:send1", "send1:home.location1", "home.location1")
     
  }
  else{
    coef$names = c("(Intercept)","sum_step.log", "jbsteps30pre.log", "dec.temperature", "sd_steps601", "dosage", "send1",
                   "home.location1", "dosage:send1", "sd_steps601:send1", "send1:home.location1")
  }
  coef = coef[c("(Intercept)","sum_step.log", "jbsteps30pre.log", "dec.temperature", "sd_steps601", "home.location1", "send1",
              "sd_steps601:send1", "send1:home.location1", "dosage:send1", "dosage"), ]
  
  write.csv(coef, filename, row.names = FALSE)
  
  dat_tmp = dat_tmp[, c('sum_step.log', 'jbsteps30pre.log', 'dec.temperature', 'sd_steps60', 'dosage', 'send', 'home.location')]
  write.csv(dat_tmp, sprintf("%s/context_pair%s.csv",file_location, u), row.names = FALSE)
}


formula_string = "jbsteps30.log ~ 1 + sum_step.log + jbsteps30pre.log  + dec.temperature + sd_steps60 + dosage + dosage * send + sd_steps60 * send"
formula_string = paste(formula_string, "+ home.location + home.location * send")
formula_touse = formula(formula_string)
fit_gee <- geeglm(formula_touse,
                  data= dat,
                  id = user,
                  family = gaussian,
                  corstr = "independence")
print(paste0("-------------", str(u), "---------"))
print(summary(fit_gee))
filename = sprintf("%s/residual_pair%s.csv",file_location, "population")
write.csv(data.frame(fit_gee$residuals), filename)
filename = sprintf("%s/coeffi_pair%s.csv",file_location, "population")
write.csv(data.frame(fit_gee$coefficients), filename)


# -------------- density plot -------------#
library(ggplot2)
data = data.frame(fit_gee$residuals)
print(sd(fit_gee$residuals))
ggplot(data = data, aes(x = fit_gee.residuals)) +
  geom_density()


data = data.frame(dat$dosage)
print(sd(dat$dosage))
ggplot(data = data, aes(x = dat.dosage)) +
  geom_density()
