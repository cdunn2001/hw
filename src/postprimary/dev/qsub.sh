#!/bin/bash

export rerundir=${rerundir-Sequel_3.0.17}
export reportdir=${reportdir-$HOME/hqrm}

mkdir -p $reportdir
chmod 777 $reportdir

for i in "$@"
do
  case $i in
    -v|--verbose)
      set -v
      ;;
    -d|--dryrun)
      DRYRUN=1
      ;;
    --rerundir=*)
      rerundir="${i#*=}"
      ;;
    *)
      RUNS_CSV="${i#*=}"
      ;;
  esac
done


[ ! -f $RUNS_CSV ] && { echo "$RUNS_CSV file not found"; exit 99; }

OLDIFS=$IFS
IFS=", " # allow either spaces or commas
while read inputdir movie reference
do
  lims="$inputdir/lims.yml"
  if [ -e "$lims" ]
  then
    moviex=`perl -ne 'if (/tracefile: .(.*).trc.h5./) {print $1;}' $lims `
    if [ "$movie" == "x" ]
    then
      movie=$moviex
    else
     if [ "$moviex" != "$movie" ]
     then
      echo BAD movie agreement, $moviex ne $movie
      exit 1
     fi
    fi
  fi

  export inputdir
  export movie
  export reference
  echo GOT: $inputdir movie:$movie rerundir:$rerundir reference:$reference
  OUTDIR=`./setup.sh | gawk '/OUTDIR/{print $2}'`
  echo OUTDIR ${OUTDIR}
  if [ ! $DRYRUN ]
  then
    ( cd ${OUTDIR} && make submit )
  fi
done < $RUNS_CSV


IFS=$OLDIFS
