Random Tilings
==============

This project provides the Python package `RandomTilings`, which makes it possible to generate random tilings of the Aztec diamond and the hexagon for a given doubly periodic weighting.
It is a translation of the Matlab program `MatlabTilings` by Christophe Charlier, which is based on the shuffling algorithm as described in  [arXiv:0111034](https://arxiv.org/abs/math/0111034).
The original Matlab implementation can be found on his [hompage](https://sites.google.com/view/cchar/home). We are grateful to Christophe Charlier for
allowing us to make this program publicly available.

<p align="center">
<img width="383" height="389" alt="image" src="https://github.com/user-attachments/assets/b3d983e2-561e-461c-8fda-cfd6c933a24d" />
<img width="340" height="389" alt="image" src="https://github.com/user-attachments/assets/50255924-afd4-46ee-9942-39400836810f" />
</p>


# Important Notes
A documentation for the the rotuines provided by `RandomTilings` can be found in the pdf `Documentation for Random Tilings`. 
In the current version, not all options from `MatlabTilings` have been implemented in the Python
program. Most notably, the program only produces images of tilings, and not of the corresponding
non-intersecting path systems. Therefore, `Documentation for Random Tilings` mainly discusses the necessary changes important for
the Python program. Some parts of the [`Help file`](https://sites.google.com/view/cchar/random-tilings)  for `MatlabTilings` are copied for context. For a
more complete description, the reader should consult the `Help file`; please keep in mind that only some
features are available in the Python program.

For the example usage of the routines provided by `RandomTilings` have a look in the Jupyter notebook `Examples - How to use` or in the `How to use` section below.

**The Power of Numba**      
The creation of these Tilings is numerically expensive. In order to drastically improve the performance this package makes heavy use of the `numba` compiler `jit`.
Therefore, the preinstallation of `numba` is required. Also importing `RandomTilings` might take some time, as all subroutines have to be compiled.


# Acknowledgments

# How to get started

# Examples
