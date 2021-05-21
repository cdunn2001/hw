export basecaller_console_app_exec=basecaller-console-app
export baz=rerun.baz

. /etc/profile.d/modules.sh
module load basecaller/mainline
module load smrtanalysis/mainline

./qsub.sh martin_runs.csv

