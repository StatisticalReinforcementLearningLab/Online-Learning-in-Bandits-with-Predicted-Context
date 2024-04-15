# daily step counts
library(dplyr)
library(tidyr)

jbsteps = read.csv("jbsteps.csv")
jbsteps$study.date = sapply(jbsteps$steps.utime.local, function(t) substring(t, 1, 10))
results = jbsteps %>% group_by(study.date, user.index) %>% summarize(sum_step = sum(steps))
results$sum_step
