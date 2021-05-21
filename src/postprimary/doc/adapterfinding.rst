Algorithm - Adapter Finding
---------------------------

In theory, AdapterFinder is a C# port of the RSII code base. In practice,
the workflow has been adapted, but the code has been heavily modified to 
achieve a 5x performance boost. Following implementation tweaks
have been performed:

* Memory access in the SpacedSelector has been tuned to get a 50% run-time 
  reduction in this step.
* Minimalistic Smith-Waterman implementation including the knight-move.
  Doesn't save the additional backtracking matrix.
* New backtracking algorithm, as the old backtracking matrix has been removed.

Symmetric vs. asymmetric
^^^^^^^^^^^^^^^^^^^^^^^^

The AdapterFinder can run in a symmetric and asymmetric mode. The switch is 
implicite, determined by the number of sequences in the adapters.fasta file,
provided to baz2bam/bam2bam. The maximal number of allowed sequences is two.

In the **symmetric mode**, the adapter sequence is being align against
the full-length ZMW read. The last row of the dynamic programming matrix is being 
used for the SpacedSelector that searches for adapter hits that are at least 
40 bp apart; current optimal number to separate partial from full adapter hits.
This list, sorted by begin position, undergoes a clipping process to avoid
overlapping adapters.

In the **asymmetric mode**, each adapter is treated as an individual symmetric
adapter, with the exception that the distance for the SpacedSelector is being
raised to the double distance, 80 bp. Both adapter hit lists are merged
and sorted. Overlapping hits of different adapters sequences are not allowed;
the hit with the lower score gets removed. Overlapping hits of the same adapter
sequence gets clipped as in the symmetric case. In this mode
the orientation of the subreads can be determined by the leading and trailing
adapter hits. This orientation can be read from the cx tag in the BAM file.
You can use this handy website to `decode the cx tag <http://web/~atoepfer/cx.html>`_.
Asymmetric adapter calling takes twice as long as the symmetric mode. 
Vectorization, aligning both adapters at the same time, does not provide a
run-time advantage.