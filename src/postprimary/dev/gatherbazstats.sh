
# all=/pbi/collections/320/3200036/r54133_20161028_151237/29_F01/m54133_161029_053924.baz

movielength(){
  x=$1
  xmllint --xpath "string(//*[local-name()='AutomationParameter'][@Name='MovieLength']/@SimpleValue)" $x
}
readout(){
  x=$1
  path=$(dirname $1)/
 # xmllint --xpath "//*[local-name()='Readout']/text()" $x
  xmllint --xpath "//*[local-name()='CollectionPathUri' and text()='$path']/../*[local-name()='Readout']/text()"  $x
}
verbosity(){
  x=$1
  path=$(dirname $1)/
  #xmllint --xpath "//*[local-name()='MetricsVerbosity']/text()" $x
  xmllint --xpath "//*[local-name()='CollectionPathUri' and text()='$path']/../*[local-name()='MetricsVerbosity']/text()"  $x
}


all=/pbi/collections/*/*/*/*/*.baz

process(){
  baz=$1
  xml=${baz%.baz}.run.metadata.xml

  echo $baz,$(stat -c%s $baz),$(movielength $xml),$(readout $xml),$(verbosity $xml)
}

#find /pbi/collections/320 -name "*.baz" -exec process {} \;

echo '"baz","size","movielength","readout","verbosity"'
find /pbi/collections -name "*.baz"  | while read file; do process $file; done


# xmllint --xpath "//*[local-name()='CollectionPathUri' and text()='/pbi/collections/320/3200036/r54133_20161028_151237/29_F01/']/../*[local-name()='Readout']/text()"  $x

#          <pbmeta:OutputOptions>
#.            <pbmeta:CollectionPathUri>/pbi/collections/320/3200036/r54133_20161028_151237/1_A01/</pbmeta:CollectionPathUri>
#.            <pbmeta:Readout>Bases_Without_QVs</pbmeta:Readout>
#.            <pbmeta:MetricsVerbosity>High</pbmeta:MetricsVerbosity>


