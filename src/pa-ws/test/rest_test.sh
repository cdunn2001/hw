host=${host:-pa-dev01.lab.nanofluidics.com}
port=${port:-23632}
baseurl="http://${host}:${port}"
curlopts="--silent --show-error --write-out :CODE(%{http_code})"
#curlopts="--silent --show-error --write-out '<<<%{http_code} %{content_type}>>>'"

echo "Using $baseurl"

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


curl $curlopts -X GET $baseurl/sockets         | pretty

echo
echo "GET /sockets/1"
echo

curl $curlopts -X GET $baseurl/sockets/1       | pretty

echo
echo "GET /sockets/1/basecaller/process_status"
echo

curl $curlopts -X GET $baseurl/sockets/1/basecaller/process_status       | pretty

echo
echo "#########################################################################"
echo
echo "GET /postprimaries"
echo

curl $curlopts -X GET $baseurl/postprimaries   | pretty

# curl $curlopts -X GET $baseurl/postprimaries/m123456_000001

echo
echo "GET /postprimaries/m123456_000001"
echo

curl $curlopts -X GET $baseurl/postprimaries/m123456_000001   | pretty

echo
echo "#########################################################################"
echo
echo "GET /storages"
echo

curl $curlopts -X GET $baseurl/storages        | pretty

echo
echo "GET /storages/m123456_000001"
echo

curl $curlopts -X GET $baseurl/storages/m123456_000001       | pretty

echo
echo "#########################################################################"
echo
echo "GET /transfers"
echo

curl $curlopts -X GET $baseurl/transfers       | pretty

echo
echo "GET /transfers/m123456_000001"
echo

curl $curlopts -X GET $baseurl/transfers/m123456_000001       | pretty

echo "</pre>"

failures=0

echo "<ul>"
# postprimaries
ret=$(curl $curlopts --data '{"mid":"m123456_987654_s0","uuid":"123"}' -X POST $baseurl/postprimaries )
echo "<li> postprimaries post returned $ret"
if [[ ! $ret == "\"m123456_987654_s0\"CODE(201)" ]] ; then echo FAIL POST $baseurl/postprimaries; failures=$(($failures+1)); fi
echo -e "</li>\n"

ret=$(curl $curlopts -X POST $baseurl/postprimaries/m123456_987654_s0/stop )
echo "<li> postprimaries/stop post returned $ret"
echo -e "</li>\n"

ret=$(curl $curlopts -X DELETE $baseurl/postprimaries/m123456_987654_s0 )
echo "<li> postprimaries DELETE returned $ret"
echo -e "</li>\n"

# sockets
ret=$(curl $curlopts -X POST $baseurl/sockets/reset )
echo "<li> sockets/reset returned $ret"
echo -e "</li>\n"

ret=$(curl $curlopts -X POST $baseurl/sockets/0/reset )
echo "<li> sockets/0/reset returned $ret"
echo -e "</li>\n"

ret=$(curl $curlopts --data '{"movieNumber":0,"calibFileUrl":"/storages/0/darkcal.h5"}' -X POST $baseurl/sockets/0/darkcal/start )
echo "<li> socket darkcal/start created $ret"
if [[ $ret =~ CODE(201) ]] ; then echo FAIL POST $baseurl/sockets/0/darkcal/start; failures=$(($failures+1)); fi
echo -e "</li>\n"

ret=$(curl $curlopts --data '{"movieNumber":2,"calibFileUrl":"/storages/0/loadingcal.h5"}' -X POST $baseurl/sockets/0/loadingcal/start )
echo "<li> socket loadingcal/start created $ret"
if [[ $ret =~ CODE(201) ]] ; then echo FAIL POST $baseurl/sockets/0/loadingcal/start; failures=$(($failures+1)); fi
echo -e "</li>\n"

ret=$(curl $curlopts --data '{"uuid":"123"}' -X POST $baseurl/sockets/0/basecaller/start )
echo "<li> socket basecaller/start POST created $ret"
if [[ $ret =~ CODE(201) ]] ; then echo FAIL POST $baseurl/sockets/0/basecaller/start; failures=$(($failures+1)); fi
echo -e "</li>\n"

# storages
ret=$(curl $curlopts --data '{"mid":"m123456_987654_s0"}' -X POST $baseurl/storages/0 ) 
echo "<li> storages POST returned $ret"
if [[ $ret =~ CODE(201) ]] ; then echo FAIL POST $baseurl/storages/0; failures=$(($failures+1)); fi
echo -e "</li>\n"

# transfers
ret=$(curl $curlopts --data '{"mid":"m123456_987654_s0","uuid":"123"}' -X POST $baseurl/transfers )
echo "<li> transfers POST returned $ret"
if [[ $ret =~ CODE(201) ]] ; then echo FAIL POST $baseurl/transfers; failures=$(($failures+1)); fi
echo -e "</li>\n"

ret=$(curl $curlopts -X POST $baseurl/transfers/m123456_987654_s0/stop )
echo "<li> transfers/stop POST returned $ret"
if [[ $ret =~ CODE(200) ]] ; then echo FAIL POST $baseurl/transfers/m123456_987654_s0/stop; failures=$(($failures+1)); fi
echo -e "</li>\n"

ret=$(curl $curlopts -X DELETE $baseurl/transfers/m123456_987654_s0 )
echo "<li> transfers DELETE returned $ret"
echo -e "</li>\n"

echo "</ul>"

echo "failures: $failures"

