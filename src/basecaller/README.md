smrt-basecaller

Commandline utility to read from a DataSource (typically trace file or WX hardware) and write a BAZ file and optional trc.h5 file.


Release History

0.1.6            Release testing on instrument (84003)
0.1.8            First automation release.
0.1.9            Contains logger fix so that it doesn't rotate or cleanup log directory
0.1.10 2022-4-8  Fixes for darkframe and crosstalk and PSF support
0.1.11 2022-4-11 More fixes for darkframe and crosstalk support. Needed to build with newer hw-mongo artifact.
0.1.12 2022-4-14 baz2bam progress fixes, PTSD-1462 Nan/Zeros fixes
0.1.13 2022-4-21 upgrade to hw-mongo to get photoelectronSensitivity, Analogs and refSnr working.
0.1.14 2022-4-22 Passing of crosstalk and PSF kernels to DataSource was fixed. Otherwise PSF is never used.
0.1.15 2022-4-27 Added pedestal support via WXIPCDataSource rebuild
0.1.31 2022-5-3  Plumbing of chipLayoutName added
0.1.34 2022-6-02 Added additional basecaller status reports when waiting for chunks from datasource, to prevent a timeout