Installation
============

Build
-----

1. On a PA server::

    cd Sequel/ppa &&\
    mkdir build &&\
    cd build &&\
    cmake -DCMAKE_TOOLCHAIN_FILE=../../../common/TC-icc-x86_64.cmake .. &&\
    make -j

or::

    ./cmake_setup.sh

Deployment
----------

1. Primary::

    http://jenkins:8080/view/PostPrimary/view/Deployment/job/ppa-deployment-package/

2. Secondary::

    http://jenkins:8080/view/PostPrimary/view/Deployment/job/ppa-secondary-deployment/
