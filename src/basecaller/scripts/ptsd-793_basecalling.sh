module use /etc/modulefiles
module load pacbio-pa-smrt-basecaller/0.1.4
BAZ_OUTPUT=/data/pa/mybazfile.baz FRAMES=5120 ./start_pa_mongo.sh
