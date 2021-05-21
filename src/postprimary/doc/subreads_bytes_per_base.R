library(ggplot2)

x <- read.csv("c:/lakata/subreads_bytes_per_base.csv",col.names = c('bytes', 'bases','readout','minutes','path'))
x$minutesn = as.numeric(x$minutes)

x$bytes_per_base   = x$bytes / x$bases
x$bytes_per_minute = x$bytes / x$minutesn
xx <- subset(x, bases > 0 & bytes > 0 & bytes <1000000000 & readout =="Pulses")

m = mean(xx$bytes_per_base)
s = sd(xx$bytes_per_base)

ggplot(xx,aes(x=bytes_per_base)) + geom_histogram(aes(y = ..density..),bins=100) + 
  stat_function (fun=dnorm,args=list(mean=m, sd=s), colour = "red")

print( m)
print (s)


# ggplot(x,aes(x=bytes)) + geom_histogram(aes(y = ..density..),bins=100)

