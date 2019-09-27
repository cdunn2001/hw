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
  mkdir -p /mnt/software/m/mongo-basecaller/develop-$moduleVersion/bin-{gcc,intel}
  cp -a build/x86_64/Release_gcc/applications/mongo-basecaller \
    /mnt/software/m/mongo-basecaller/develop-$moduleVersion/bin-gcc
  cp -a build/x86_64/Release/applications/mongo-basecaller \
    /mnt/software/m/mongo-basecaller/develop-$moduleVersion/bin-intel
  cat > /mnt/software/modulefiles/mongo-basecaller/${moduleVersion} << EOF
#%Module
module load ppa/$bamboo_planRepository_branchName

prepend-path PATH /mnt/software/m/mongo-basecaller/${moduleVersion}/
prepend-path LD_LIBRARY_PATH /mnt/software/m/mongo-basecaller/lib

if {[file executable /mnt/software/log/log_usage]} {
    exec /mnt/software/log/log_usage bascaller ${moduleVersion}
}
setenv smoke_cmd "mongo-basecaller --version"
#setenv BASECALLER_VERSION ${moduleVersion}
EOF
}

build_basecaller
deploy_basecaller
