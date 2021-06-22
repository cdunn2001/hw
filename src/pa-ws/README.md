pa-ws
-----

pa-ws is the REST interface between Primary Analysis applications and the rest of the world.

Demo
====

    ./pa-ws &

Open http://$machine:23632/dashboard in a Chrome browser.

Or run

    src/pa-ws/test/rest_test.sh


Debug Tools
===========

Make sure that pa-ws is listening.

    sudo netstat -tulpn | grep LISTEN

    curl http://localhost:23632/sockets/1

