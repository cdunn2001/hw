host=${host:-localhost}
port=${port:-23632}
baseurl="http://${host}:${port}"
curlopts1="--silent"
curlopts2="--silent --show-error --write-out :CODE(%{http_code})"

echo "<h1>Testing pa-ws on $baseurl</h1>"

function wraparrays {
    perl -pe 'if (/^(\s+)(\d+,)$/) {$_ = ($x ? "" : $1 ) . $2; $x = 1; } else { if ($x) { s/^\s+//;} $x = 0; }'
}

function pretty {
    python -m json.tool | wraparrays
}


testGet() {
    echo "<hr>#########################################################################"
    echo "<li> GET $baseurl/$1 returned"
    response=$(curl $curlopts2 -X GET $baseurl/$1 )
    echo "<pre>"
    echo $response | perl -pe 's/:CODE\(\d+\)//;' | pretty
    echo "</pre>"
    code=$( echo $response | perl -ne 'if(/:CODE\((\d+)\)/) {print $1;};' )
    if [[ $code == 200 ]] ; then 
      echo SUCCESS
    else
      echo FAIL $code is not $2
      failures=$(($failures+1))
    fi

}

testPost() {
    echo "<hr>#########################################################################"
    echo "<li> POST $baseurl/$1 returned"
    response=$(curl $curlopts2 --data "$3" -X POST $baseurl/$1 )
    echo "<pre>"
    echo $response | perl -pe 's/:CODE\(\d+\)//;' | pretty
    echo "</pre>"
    code=$( echo $response | perl -ne 'if(/:CODE\((\d+)\)/) {print $1;};' )
    if [[ $code == $2 ]] ; then 
      echo SUCCESS
    else
      echo FAIL $code is not $2
      failures=$(($failures+1))
    fi

    echo -e "</li>\n"
}

testDelete() {
    echo "<hr>#########################################################################"
    echo "<li> DELETE $baseurl/$1 returned"
    response=$(curl $curlopts2 -X DELETE $baseurl/$1 )
    echo "<pre>"
    echo $response | perl -pe 's/:CODE\(\d+\)//;' | pretty
    echo "</pre>"
    code=$( echo $response | perl -ne 'if(/:CODE\((\d+)\)/) {print $1;};' )
    if [[ $code == $2 ]] ; then 
      echo SUCCESS
    else
      echo FAIL $code is not $2
      failures=$(($failures+1))
    fi

    echo -e "</li>\n"
}
failures=0


echo "<ul>"
testGet status
testGet sockets
testGet sockets/1
testGet sockets/1/basecaller/processStatus
testGet postprimaries
testGet postprimaries/m123456_000001
testGet storages
testGet storages/m123456_000001


testPost postprimaries 201 '{"mid":"m123456_987654_s0","uuid":"123"}'
testPost postprimaries/m123456_987654_s0/stop 200 ''
testDelete postprimaries/m123456_987654_s0 200

testPost sockets/reset   200 ''
testPost sockets/1/reset 200 ''
testPost sockets/1/darkcal/start    201 '{"movieNumber":0,"calibFileUrl":"/storages/0/darkcal.h5"}'
testPost sockets/1/loadingcal/start 201 '{"movieNumber":2,"calibFileUrl":"/storages/0/loadingcal.h5"}'
testPost sockets/1/basecaller/start 201 '{"uuid":"123"}'
 
testPost storages 201 '{"mid":"m123456_987654_s0"}'
testGet  storages/m123456_987654_s0
echo "</ul>"

echo "failures: $failures"

