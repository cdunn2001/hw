#!/bin/bash
set -euo pipefail

source module_setup.sh

main(){
  tool=$1
  if [[ ! -e build/$tool ]]; then
    ./cmake_setup.sh
  fi
  cd build/$tool/gcc/x86_64/Release
  ninja package
}

main $1
