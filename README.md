This directory contains three MATLAB scripts: MPRI\_HSI\_pseudoRGB_SSL.m, TensorSSA\_HSI\_pseudoRGB_SSL.m and Grassmann\_HSI.m. The first two scripts combine self-trained semi-supervised learning (SSL) algorithm, implemented through MATLAB function fitsemiself, with 3D spectral-spatial features extracted by:

(1) Multiscale Priniciple of Relevant Information (MPRI) approach, Y. Wei, S. YU, L. S. Giraldo, J. C. Principe, "Multiscale principle of relevant information for hyperspectral image classification", Machine Learning (2023) 112:1227-1252, https://doi.org/10.1007/s10994-021-06011-9. 

Corresponding code is placed in sub-directory .\MPRI\_code.

(2) Tensor Singular Spectrum  Analysis (TensorSSA) approach,  H. Fu, G. Sun, A. Zhang, B. Shao, J. Ren, and X. Jia, "Tensor Singular Spectrum Analysis for 3-D Feature Extraction in Hyperspectral Images," IEEE Transactions of Geoscience and Remote Sensing", vol. 61, article no. 5403914, 2023. 

Corresponding code is placed in sub-directory .\TensorSSA\_code.

The third script, Grassmann\_HSI.m, implements Grassmann-manifold and nearest subspace classifier with TensorSSA spectral-spatial features.

The scripts read either 27 hyperspectral images of 27 corresponding co-registered RGB color images assumed to be placed, together with ground truth files, in the same working directory. Consequently, prior to running the scripts data have to be downloaded from the repository: https://data.fulir.irb.hr/islandora/object/irb:538

This work is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
