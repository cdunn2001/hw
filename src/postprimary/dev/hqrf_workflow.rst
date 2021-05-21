Generate reference High Quality Region
--------------------------------------

Input: standard BAM output (subtreads.bam)

1. use bam2bam to reproduce large BAM file that is not broken into subreads
2. develope tool or use existing feature to set HQR to "everything"
3. align against reference genome with blasr
4. Mark regions of alignment as "A1", and mark other regions as An.



bam2bam
=======

::

    % bam2bam  --polymerase                              \
          -o movieName.stitched  \
          in.subreads.bam in.scraps.bam -o out


    Took 37 minutes


blasr
=====
Use blasr 3.0.0 which has ability to load BAM files
(module load blasr/3.0.0)


::

    % blasr -m 0 /data/pa/compare0.fasta groundtruth.fasta


     -m t
               If not printing SAM, modify the output of the alignment.
                t=0 Print blast like output with |'s connecting matched nucleotides.
                  1 Print only a summary: score and pos.
                  2 Print in Compare.xml format.
                  3 Print in vulgar format (DEPRECATED).
                  4 Print a longer tabular version of the alignment.
                  5 Print in a machine-parsable format that is read by compareSequences.py.

::

    pa-dev01$ pwd
    /pbi/collections/315/3150113/r54007_20160226_225820/1_A01

    pa-dev01$ less /lims.yml
    expcode: 3150113
    runcode: '3150113-0002'
    path: 'file:///pbi/collections/315/3150113/r54007_20160226_225820/1_A01'
    user: 'MilhouseUser'
    uid: '6680947a1e6d16ca35a9182c98cc4413'
    tracefile: 'm54007_160226_225834.trc.h5'
    description: '10kEcoli'
    wellname: 'A01'
    cellbarcode: '00000000635921243149065820'
    seqkitbarcode: '170216100620000021717'
    cellindex: 0
    colnum: 0
    samplename: '10kEcoli'
    instid: 90



references
==========

Lance suggested this reference:

::

  /mnt/secondary/iSmrtanalysis/current/common/references/ecoliK12_pbi_March2013/sequence/ecoliK12_pbi_March2013.fasta

  test case walk through:

  the first ZMW to have a sequence that is alignable is 4194377 (0x00400049).
  pa-dev01$ blasr -m 0 /data/pa/m54007_160226_225834.polymerase.bam /mnt/secondary/iSmrtanalysis/current/common/references/ecoliK12_pbi_March2013/sequence/ecoliK12_pbi_March2013.fasta -holeNumbers 4194367-4194377
  [INFO] 2016-02-29T16:39:04 [blasr] started.
      nMatch: 3541
   nMisMatch: 168
        nIns: 1064
        nDel: 105
        %sim: 72.5912
       Score: -10852
           Query: m54007_160226_225834/4194377/0_9106/0_9106
          Target: ecoliK12_pbi_March2013
           Model: a hybrid of global/local non-affine alignment
       Raw score: -10852
          Map QV: 254
    Query strand: 0
   Target strand: 0
     QueryRange: 974 -> 5747 of 9106
    TargetRange: 725798 -> 729612 of 4642522
     974  GGCTAACCAAAGCAGAATGGAAGGCTAATCAACCAGAACAAAGGGTGGGG
          |||||||||||||||||||||||||||||   ||   || ||
  725798  GGCTAACCAAAGCAGAATGGAAGGCTAAT---CC---AC-AA--------

    1024  GGAGAGGACGGGCCAGATCGGGAGCGAGAACAAAGTAACGCAGC-CGACG
           *| | | |  |||   ||  |*|| |||||  ||| ||| | | || ||
  725833  -CA-A-G-C--GCC---TC--GTGC-AGAAC--AGT-ACG-A-CACG-CG

    1073  AAAAGACCATCAAGCAAAACTCAGGCTGGGTAATGCGCTTATGAGGCGTA
          |||| |||||||||| |||||||||||||||  ||||| ||||| |||||
  725865  AAAA-ACCATCAAGC-AAACTCAGGCTGGGT--TGCGC-TATGA-GCGTA

    1123  CGCCTGAGTTCGCGGGATTGATATGAG-GTAGAGGAATGTA-GCCGGGAG
           |||||||||||||||| ||||||||| || ||*| ||||| ||| ||||
  725909  -GCCTGAGTTCGCGGGA-TGATATGAGTGT-GATG-ATGTATGCC-GGAG

    1171  AGCGAGAAAACGAACCCCAGGACCAGGTGCAATACCGACATGGCACCCAC
          |||||| ||||| | |||||*|||| || |||||||  ||||||| ||||
  725954  AGCGAG-AAACG-A-CCCAGTACCA-GT-CAATACC--CATGGCA-CCAC

    1221  AACTTAAAACCCGCCACATGCGGGCGGCGTGATTACCCTGCAACGCCATT
          |||||  ||||||*|||||||||||||||||||||||||||||||||| |
  725996  AACTT--AACCCGTCACATGCGGGCGGCGTGATTACCCTGCAACGCCA-T

    1271  TAACCAAGACGAAGA-AC-GCCGCTGTATGGACAAACCGTGGGGGAAACT
          | ||| || *|||*| || |||||    ||||||*||||| ||*| ||||
  726043  T-ACC-AG-GGAATAGACAGCCGC----TGGACACACCGT-GGTG-AACT

    1319  GGGCGTTCGCGGTCACGTTTAACGTACC-GCCGTAAAACTACGAAAACAT
          |||||||| |||||||| ||*|| |||| ||| | *||||||||||||||
  726084  GGGCGTTC-CGGTCACG-TTCAC-TACCGGCC-T-GAACTACGAAAACAT

    1368  GAGATTGAAAACCGCAAGGGCGACAATAACTTCCGCCTGAATA-CGGCAA
          |||  ||||||||||||||||*||||||||||||||||||||| |||| |
  726129  GAG--TGAAAACCGCAAGGGCTACAATAACTTCCGCCTGAATAGCGGC-A

    1417  TGCCGGAGTACGGGCAAAAGACAAAAGGTGAAGTTAGGGCGGGAGAAAGT
          |||||||||||||||     | |||||||| ||||   ||        ||
  726176  TGCCGGAGTACGGGC-----A-AAAAGGTG-AGTT---GC--------GT

