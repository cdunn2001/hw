PPA - Preamble
==============

Every single customer-facing byte that gets produced by Sequel is processed by
PPA, thus every tiny change may lead to downstream analysis incompatibilities 
or wrong results.

Software development process
----------------------------

We follow a thorough process to minimize the introduction of defects and 
guarantee interface stability with primary and secondary analysis.

1) Every change has to go through swarm code review. 
   Direct p4 submissions will get reverted.

2) Before issuing a review, changes have to pass
    - validation cram tests to assure SA compatibility,
    - all existing unit tests, if not, provide reasons for the test change,
    - compilation of PA basewriter and basecaller.

3) New functionality has to be tested via unit-tests or cram validation tests.

4) BazWriter API changes have to be approved by PA.

5) Stats changes have to be coordinated with ICS.

6) IPC changes have to be coordinated with PAWS.

7) BazWriter implementation changes have to be benchmarked via simulations to
   guarantee that each superchunks gets written in less than 200 seconds.

8) Algorthmic changes to AdapterFinding (AF), BarcodeCalling (BC), 
   HQ-RegionFinding (HQRF), and ControlFiltering (CF) have to meet 
   existing sensitivity and accuracy standards.

9) Baz2bam has to process a 1M chip with mean read-length of 65kb with 
   AF, BF, CF, and HQRF in less than three hours with -j 12 -b 4.

Versioning scheme
-----------------
Software package scheme is: x.y.z.w  

RPM scheme: x.y.z-w::

  x = BAZ format or other MAJOR change; 
      if this changes, no backward compatibility at all
  y = New official release version
  z = Changes that are due to additional functionality
  w = Hourly changes (typically no change in functionality)

Branching
---------
PPA gets branched at the same time as Primary. After that, PPA y-version 
gets incremented. Important fixes will be performed in the branch and if 
necessary, the z-version will be incremented to reflect that. Secondary
will use the branched version if they intend to ship a version for that
particular release.
