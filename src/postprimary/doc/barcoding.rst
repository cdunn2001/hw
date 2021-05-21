Algorithm - Barcode Calling
---------------------------

The new barcoding algorithm differs from the original RSII version, to provide
better estimates. Following the new workflow.

Selection of the asymmetric barcode pair:

1. Select the first N adapters, which flanking barcode regions are in the HQ-region.
2. Align all possible X barcodes against the flanking adapter regions 
   individually, using a Smith-Waterman. We receive 2*N score vectors, each of length X.
3. Normalize each score vector by its sum.
4. Get the argmax for each score vector.
5. If we have two or more argmax values from the 2*N score vector, 
   get two score vectors that correspond to those argmax with the the highest 
   counts and use them as initial clusters of a kmeans Lloyd algorithm.
6. Create horizontal score vector sums for each cluster.
7. Get the argmax of each cluster score vector sum.