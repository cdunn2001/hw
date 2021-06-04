Overview
========

Primary Analysis applications for kestrel and beyond, including smrt-basecaller, pa-ws, pa-calib, pa-transfer

Directory structure:
====================

Directories committed to git:

    src/
    doc/

Directories generated:

    build/
    depcache/



Build instructions:
===================

Example for pa-ws. Replace "pa-ws" with "pa-cal" or "basecaller".

1. . module_setup.sh
2. ./cmake_setup.sh
3. cd build/pa-ws/gcc/x86_64/Release
4. ninja
5. ./pa-ws --showconfig
6. ./test/testPaWs
   
