library(ggplot2)

x <- read.csv("c:/lakata/subreads_bytes_per_base.csv",col.names = c('bytes', 'bases','readout','minutes','path'))
x$minutesn = as.numeric(x$minutes)

x$bytes_per_base   = x$bytes / x$bases
x$bytes_per_minute = x$bytes / x$minutesn
x$bases_per_second = x$bases / x$minutesn / 60 / 1e6
xx <- subset(x, bases > 0 & bytes > 0 & bytes <1000000000 & readout =="Pulses")

m = mean(xx$bytes_per_base)
s = sd(xx$bytes_per_base)

ggplot(xx,aes(x=bases_per_second)) + geom_histogram(aes(y = ..density..),bins=100) + 
  ggplot(xx,aes(x=bases_per_second)) + geom_histogram(aes(y = ..density..),bins=100) 
print( m)
print (s)


# ggplot(x,aes(x=bytes)) + geom_histogram(aes(y = ..density..),bins=100)



###########
# bytes per pulse, without overhead/header/metrics/padding
# to get this data, uncomment line 422 of StitchedZmw.cpp (#if 1), run 'bazviewer -d .... > sum1.txt'

library(ggplot2)
t <- read.table("x:git/primaryanalysis/Sequel/ppa/build/x86_64/Release/sum1.txt")
ggplot(t,aes(x=V5)) + geom_histogram()
print (mean(t$V5))

save as bytes_per_pulse.png

