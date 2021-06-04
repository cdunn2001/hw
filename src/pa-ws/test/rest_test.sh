echo "<pre>"

function wraparrays {
    perl -pe 'if (/^(\s+)(\d+,)$/) {$_ = ($x ? "" : $1 ) . $2; $x = 1; } else { if ($x) { s/^\s+//;} $x = 0; }'
}

function pretty {
    python -m json.tool | wraparrays
}


echo "#########################################################################"
echo
echo "GET /sockets"
echo


curl -s -X GET localhost:23632/sockets         | pretty

echo
echo "GET /sockets/1"
echo

curl -s -X GET localhost:23632/sockets/1       | pretty

echo
echo "GET /sockets/1/basecaller/process_status"
echo

curl -s -X GET localhost:23632/sockets/1/basecaller/process_status       | pretty

echo
echo "#########################################################################"
echo
echo "GET /postprimaries"
echo

curl -s -X GET localhost:23632/postprimaries   | pretty

# curl -s -X GET localhost:23632/postprimaries/m123456_000001

echo
echo "GET /postprimaries/m123456_000001"
echo

curl -s -X GET localhost:23632/postprimaries/m123456_000001   | pretty

echo
echo "#########################################################################"
echo
echo "GET /storages"
echo

curl -s -X GET localhost:23632/storages        | pretty

echo
echo "GET /storages/m123456_000001"
echo

curl -s -X GET localhost:23632/storages/m123456_000001       | pretty

echo
echo "#########################################################################"
echo
echo "GET /transfers"
echo

curl -s -X GET localhost:23632/transfers       | pretty

echo
echo "GET /transfers/m123456_000001"
echo

curl -s -X GET localhost:23632/transfers/m123456_000001       | pretty

echo "</pre>"
