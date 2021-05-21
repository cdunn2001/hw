Test to see if it runs on 1M holes.

  $ D=/pbi/collections/315/3150283/r54008_20160701_000712/1_A01
  $ M=m54008_160701_001124
  $ baz2bam -o out -m ${D}/${M}.metadata.xml.sequel --adapters ${D}/${M}.adapters.fasta -j 44 -b 4 ${D}/${M}.baz


Alternate (not used)
baz2bam -o out -m ${D}/${M}.metadata.xml.sequel --adapters ${D}/${M}.adapters.fasta -j 44 -b 4 ${D}/${M}.baz --noBam --noScraps --noPbi
