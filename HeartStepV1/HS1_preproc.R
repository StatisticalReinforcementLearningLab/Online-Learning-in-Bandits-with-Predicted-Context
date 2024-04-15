# setwd("HeartStepsV1-main/data_files")
library(dplyr)
library(tidyr)
## suggestion data
suggestions = read.csv("HeartStepsV1-main/data_files/suggestions.csv")
## remove points during travel days (390)
suggestions = subset(suggestions, !is.na(suggestions$decision.index.nogap))
## reorder the data
suggestions = suggestions[order(suggestions$user.index, suggestions$decision.index),]
row.names(suggestions) = NULL
## one point does not have decision time
idx.null = which(suggestions$sugg.select.utime == "")
suggestions$sugg.select.utime[idx.null] = suggestions$sugg.select.utime[idx.null + 1]
## to adjust for time zone difference
suggestions$study.date = sapply(suggestions$sugg.select.utime, function(t) substring(t, 1, 10))
suggestions$study.hour = sapply(suggestions$sugg.select.utime, function(t) as.numeric(substring(t, 12, 13)))
date.minus.one = function(x) {
  as.character(format(as.Date(x) - 1, "%02Y-%m-%d"))
}
suggestions$study.date[suggestions$study.hour <= 3] = sapply(
  suggestions$study.date[suggestions$study.hour <= 3], date.minus.one
)


## merge with the study day data
dailyema = read.csv("HeartStepsV1-main/data_files/dailyema.csv")
dat = merge(
  suggestions, dailyema, 
  by = c("user.index", "study.date"), all.x = TRUE, sort = FALSE
)
## merge with previous day step count data

jbsteps = read.csv("HeartStepsV1-main/data_files/jbsteps.csv")
jbsteps$study.date = sapply(jbsteps$steps.utime.local, function(t) substring(t, 1, 10))
pre_sum = jbsteps %>% group_by(study.date, user.index) %>% summarize(sum_step = sum(steps))
next_date_string <- function(input_date_string) {
  # Parse the input date string into a Date object
  input_date <- as.Date(input_date_string)
  
  # Calculate the next date by adding 1 day to the input date
  next_date <- input_date + 1
  
  # Format the next date as a string in the desired format (e.g., "YYYY-MM-DD")
  next_date_string <- format(next_date, "%Y-%m-%d")
  
  return(next_date_string)
}
pre_sum$study.date = sapply(pre_sum$study.date, next_date_string)

dat = merge(
  dat, pre_sum,
  by = c('user.index', 'study.date')
)

dat = dat[order(dat$user.index, dat$decision.index),]

dat$work.location = dat$dec.location.exact == 'work'
dat$other.location = !(dat$dec.location.exact == 'work' | dat$dec.location.exact == 'home')

n = dim(dat)[1]
sd_by_user = dat %>% group_by(user.index) %>% summarise(sd_time = sd(jbsteps60, na.rm=TRUE))
dat$sd_steps60 = NA
for(i in seq(2, n, 1)){
  cur_time = dat$sugg.select.utime[i]
  cur_user = dat$user.index[i]
  date = as.Date(cur_time)
  hour = strptime(substring(cur_time, 12, 19), format = "%H:%M:%S")$hour
  i0 = max(i-40, 1)
  ll = list()
  for(j in i0:(i-1)){
    if(cur_user != dat$user.index[j]){
      next
    }
    time_j = dat$sugg.select.utime[i]
    date_j = as.Date(time_j)
    hour_j = strptime(substring(time_j, 12, 19), format = "%H:%M:%S")$hour
    if(date - date_j <= 7 & hour_j == hour){
      ll = append(ll, dat$jbsteps60[j])
    }
  }
  dat$sd_steps60[i] = (sd(unlist(ll), na.rm = TRUE) > sd_by_user[cur_user, 2])
}

dat$dosage = NA
gamma = 0.95
for(i in seq(2, n, 1)){
  cur_time = dat$sugg.select.utime[i]
  cur_user = dat$user.index[i]
  date = as.Date(cur_time)
  hour = strptime(substring(cur_time, 12, 19), format = "%H:%M:%S")$hour
  i0 = max(i-40, 1)
  ll = list()
  for(j in i0:(i-1)){
    if(cur_user != dat$user.index[j]){
      next
    }
    ll = append(ll, as.numeric(dat$send[j] == "True") * (gamma^(i-j)))
  }
  dat$dosage[i] = sum(unlist(ll), na.rm = TRUE)
}


# dat1 = dat
# dat = dat1




## exclude decision points past 42 days (340) and the anomalous decision points (4)
dat = subset(dat, study.day.nogap <= 41)
dat = dat[- which(dat$avail == "False" & dat$send == "True"), ]
dat = dat[-7193,] ## an anomalous point from the original paper

## impute missing data and take logarithm
dat$avail = as.logical(dat$avail)
dat$send = as.logical(dat$send)
dat$send[is.na(dat$send)] = FALSE
dat$jbsteps30pre[is.na(dat$jbsteps30pre)] = 0
dat$jbsteps30[is.na(dat$jbsteps30)] = 0
dat$jbsteps30pre = dat$jbsteps30pre + 0.5
dat$jbsteps30 = dat$jbsteps30 + 0.5
dat$jbsteps30pre.log = log(dat$jbsteps30pre)
dat$jbsteps30.log = log(dat$jbsteps30)

dat$sum_step.log = log(dat$sum_step)

colnames(dat)[1] = "user"

dat = subset(dat, select = c(
  user,
  sum_step.log, # Yesterday’s step count
  jbsteps30pre.log, # Prior 30-minute step count
  other.location, # location other than home and work
  work.location, # work location
  dec.temperature, # current temperature
  sd_steps60, # Step variation level
  dosage,
  send, # binary action
  jbsteps30.log # y
))
dat$send = as.numeric(dat$send)
dat$sd_steps60 = as.numeric(dat$sd_steps60)
dat$other.location = as.numeric(dat$other.location)
dat$work.location = as.numeric(dat$work.location)

write.csv(dat, "HS1_cleaned_for_simulation.csv", row.names = FALSE)
