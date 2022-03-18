# This script starts a dark capture on instrument, with the wx-daemon in loopback mode.
# when the capture is done, the wx-daemon is restarted to get ready for the next capture.

sudo systemctl start pacbio-pa-wx-daemon-0.0.8@HARDLOOP

./pa-cal --numFrames=128 --outputFile=/data/nrta/0/dark1.h5 --cal=Dark --sra=0 --config source.WXIPCDataSourceConfig.loopback=true

curl -X POST localhost:23602/restart
