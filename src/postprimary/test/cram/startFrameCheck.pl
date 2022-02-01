#! perl

$errors=0;
print "zm pd px sf pd+px\n";
while(<>)
{
  ($pd,$px,$sf,$zm) = /(pd:\S+).*(px:\S+).*(sf:\S+).*zm:i:(\S+)/;

  @pd = split/,/,$pd;
  @px = split/,/,$px;
  @sf = split/,/,$sf;

  $running = 0;

  for $i (1..$#pd)
  {
    $running += $pd[$i];
    $test = "";
    if ($running != $sf[$i]){
      $errors++;
      $test = "FAIL2";
    }
    print "$zm $pd[$i] $px[$i] $sf[$i] $running $test\n";
    $running += $px[$i];
  }

}
print "total errors $errors\n";
exit($errors);

