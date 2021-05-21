#! perl

$errors=0;
while(<>)
{
  ($pd,$px,$sf) = /(pd:\S+).*(px:\S+).*(sf:\S+)/;

  @pd = split/,/,$pd;
  @px = split/,/,$px;
  @sf = split/,/,$sf ;

  $running2= 0;

  print "pd pw px   sf   pd+pw pd+px\n";
  for $i (0..$#pd)
  {
    $running2 += $pd[$i];
    $test2="";
    if ($running2 != $sf[$i]){
      $errors++;
      $test2 = "FAIL2";
    }
    print "$pd[$i] $px[$i]   $sf[$i]   $running $running2 $test $test2\n";
    $running2 += $px[$i];
  }

}
print "total errors $errors\n";
exit($errors);

