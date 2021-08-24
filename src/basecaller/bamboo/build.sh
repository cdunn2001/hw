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
  cd ..
  source module_setup.sh
  ./cmake_setup.sh
  cd build/x86_64/Release_gcc
  cmake --build . -j
  ldd applications/smrt-basecaller
# There no longer is an Intel build, but this should be
# revived/replaced in the near term hopefully
#  cd -
#  cd build/x86_64/Release
#  ninja
#  ldd applications/smrt-basecaller
  )
}

deploy_basecaller(){
  case "$bamboo_planRepository_branchName" in
    master|release/*)
      moduleVersion="${bamboo_release_version}-${bamboo_globalBuildNumber}"
      ;;
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
  mkdir -p /mnt/software/m/mongo-basecaller/$moduleVersion/bin-gcc
#  mkdir -p /mnt/software/m/mongo-basecaller/$moduleVersion/bin-{gcc,intel}
  cp -a build/x86_64/Release_gcc/mongo/applications/mongo-basecaller \
    /mnt/software/m/mongo-basecaller/$moduleVersion/bin-gcc
#  cp -a build/x86_64/Release/mongo/applications/mongo-basecaller \
#    /mnt/software/m/mongo-basecaller/$moduleVersion/bin-intel
  cat > /mnt/software/modulefiles/mongo-basecaller/${moduleVersion} << EOF
#%Module
module load ppa/$bamboo_planRepository_branchName

prepend-path LD_LIBRARY_PATH /mnt/software/m/mongo-basecaller/lib
set-alias mongo-basecaller-gcc   "/mnt/software/m/mongo-basecaller/${moduleVersion}/bin-gcc/mongo-basecaller"
#set-alias mongo-basecaller-intel "/mnt/software/m/mongo-basecaller/${moduleVersion}/bin-intel/mongo-basecaller"
setenv smoke_cmd "mongo-basecaller-gcc --version"

if {[file executable /mnt/software/log/log_usage]} {
    exec /mnt/software/log/log_usage bascaller ${moduleVersion}
}
EOF
  if [[ $bamboo_planRepository_branchName == develop ]] \
  || [[ $bamboo_planRepository_branchName == master ]]; then
    ln -sfn ${moduleVersion} /mnt/software/modulefiles/mongo-basecaller/${bamboo_planRepository_branchName}
  fi
}

build_basecaller
echo "Deploying disabled!!!"
#deploy_basecaller
