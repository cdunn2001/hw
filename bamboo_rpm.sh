#!/bin/bash
set -euo pipefail

source module_setup.sh
set -vx

main(){
  tool=$1
  if [[ ! -e build/$tool ]]; then
    ./cmake_setup.sh
  fi
  cd build/$tool/gcc/x86_64/Release
  ninja -v package
}

main $1
