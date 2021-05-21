#!/bin/bash

if [ "$inputdir" == "" ] ; then
  echo Please specify inputdir=
  exit 1
fi
if [ "${movie}" == "" ] ; then
  echo Please specify mpovie=
  exit 1
fi
if [ "$rerundir" == "" ] ; then
  echo Warning rerun directory not set, running from start
  rerundir=.
fi

if [ "$reportdir" == "" ] ; then
  reportdir=${HOME}/hqrm
fi

OUTDIR=$(mktemp -d ${reportdir}/${movie}__$(basename $inputdir)__${rerundir}_XXXXXX)

echo OUTDIR ${OUTDIR}
echo HOSTNAME ${HOSTNAME}

settings=${OUTDIR}/Makefile.settings
echo export inputdir=${inputdir}    >  ${settings}
echo export rerundir=${rerundir}    >> ${settings}
echo export movie=${movie}              >> ${settings}
echo export outputdir=${OUTDIR}     >> ${settings}
echo export srccode=`pwd`           >> ${settings}
echo export reference=${reference}  >> ${settings}
echo export trc=/dev/null           >> ${settings}
echo export baz=/dev/null           >> ${settings}
echo export inputs=${inputdir}/${movie}.subreads.bam ${inputdir}/${movie}.scraps.bam >> ${settings}

saveif() {
  eval v=\$$1
  d=$2
  if [ "$v" == "" ]
  then 
    echo "var $1 is unset"
    if [ "$d" != "" ]
    then
      echo "will use default $d" 
      echo export $1=$d >> ${settings}
    fi
  else 
    echo "var is set to '$v'" 
    echo export $1=$v >> ${settings} 
  fi
}

saveif bam2bam_exec
saveif baz2bam_exec
saveif basecaller_console_app_exec
saveif baz
saveif reportdir

cp Makefile.pbbamr ${OUTDIR}/Makefile
pushd ${OUTDIR}
chmod 644 Makefile
make update
popd

