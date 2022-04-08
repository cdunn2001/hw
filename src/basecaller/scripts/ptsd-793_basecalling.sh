module use /etc/modulefiles
module load pacbio-pa-smrt-basecaller/0.1.5
BAZ_OUTPUT=/data/nrta/0/mybazfile.baz FRAMES=51200 ./start_pa_mongo.sh
