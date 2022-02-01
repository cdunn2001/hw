#!/usr/bin/env Rscript

suppressMessages(require(pbgalaxy))
suppressMessages(require(assertthat))

args <- commandArgs(trailingOnly=TRUE)

stsh5fn <- args[1]
stsxmlfn <- args[2]

stsh5 <- H5File(stsh5fn)

doc <- xmlTreeParse(stsxmlfn, useInternal=T)
top <- xmlRoot(doc)

# Check Productivity
p0 <- as.numeric(xmlValue(top[["ProdDist"]][["BinCounts"]][[1]]))
p1 <- as.numeric(xmlValue(top[["ProdDist"]][["BinCounts"]][[2]]))
p2 <- as.numeric(xmlValue(top[["ProdDist"]][["BinCounts"]][[3]]))
prod_dt <- getH5Dataset(stsh5, "/ZMWMetrics/Productivity")[]
p1s <- which(prod_dt == 1, arr.ind=T)
prod <- table(prod_dt)
validate_that(p0 == prod[1])
validate_that(p1 == prod[2])
validate_that(p2 == prod[3])

# Check some stats.
hqRegionSnr <- getH5Dataset(stsh5, "/ZMWMetrics/HQRegionSnrMean")[]
p1Snr <- hqRegionSnr[p1s, ]
hqSnrA <- as.numeric(xmlValue(top[["HqRegionSnrDist"]][["SampleMean"]]))
validate_that(min(p1Snr) >= 4)
validate_that(abs(mean(p1Snr[,1])-hqSnrA) < 0.001)

hqBaselineStd <- getH5Dataset(stsh5, "/ZMWMetrics/HQBaselineStd")[]
p1BaselineStd <- hqBaselineStd[p1s, 2]
hqBaselineStd <- as.numeric(xmlValue(top[["BaselineStdDist"]][["SampleMean"]]))
validate_that(abs(mean(p1BaselineStd)-hqBaselineStd) < 0.001)


rl <- getH5Dataset(stsh5, "/ZMWMetrics/ReadLength")[]
validate_that(min(rl[p1s]) >= 50)





