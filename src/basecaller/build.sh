#!/bin/bash
set -euo pipefail

case $1 in
  basecaller)
    ;;
  *)
    echo "[INFO] usage: $0 basecaller"
    exit
    ;;
esac

build_basecaller(){
  (
  source module_setup.sh
  ./cmake_setup.sh
  cd build/x86_64/Release_gcc
  make -j
  ldd applications/mongo-basecaller
  cd -
  cd build/x86_64/Release
  make -j
  ldd applications/mongo-basecaller
  )
}

deploy_basecaller(){
  case "$bamboo_planRepository_branchName" in
    #master|release/*)
    #  moduleVersion="${bamboo_release_version}-${bamboo_globalBuildNumber}"
    #  ;;
    develop)
      moduleVersion="develop-${bamboo_globalBuildNumber}"
      ;;
    *)
      echo "Branch is ${bamboo_planRepository_branchName}."
      echo "[INFO] skip publishing ${bamboo_planRepository_branchName} branch."
      exit 0
      ;;
  esac
  set -x
  mkdir -p /mnt/software/m/mongo-basecaller/$moduleVersion/bin-{gcc,intel}
  cp -a build/x86_64/Release_gcc/applications/mongo-basecaller \
    /mnt/software/m/mongo-basecaller/$moduleVersion/bin-gcc
  cp -a build/x86_64/Release/applications/mongo-basecaller \
    /mnt/software/m/mongo-basecaller/$moduleVersion/bin-intel
  cat > /mnt/software/modulefiles/mongo-basecaller/${moduleVersion} << EOF
#%Module
module load ppa/$bamboo_planRepository_branchName

prepend-path LD_LIBRARY_PATH /mnt/software/m/mongo-basecaller/lib
set-alias mongo-basecaller-gcc   "/mnt/software/m/mongo-basecaller/${moduleVersion}/bin-gcc/mongo-basecaller"
set-alias mongo-basecaller-intel "/mnt/software/m/mongo-basecaller/${moduleVersion}/bin-intel/mongo-basecaller"
setenv smoke_cmd "mongo-basecaller-gcc --version"

if {[file executable /mnt/software/log/log_usage]} {
    exec /mnt/software/log/log_usage bascaller ${moduleVersion}
}
EOF
  if [[ $bamboo_planRepository_branchName == develop ]]; then
    ln -sfn ${moduleVersion} /mnt/software/modulefiles/mongo-basecaller/develop
  fi
}

build_basecaller
deploy_basecaller
