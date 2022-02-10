module use /etc/modulefiles
module load pacbio-pa-smrt-basecaller/0.1.4
ROI=[[0,0,64,256],[4000,3072,64,256]] TRACE_OUTPUT=/data/pa/mytracefile.trc.h5 NOP=2 FRAMES=5120 ./start_pa_mongo.sh
