Random Tilings
==============

This project provides the Python package `RandomTilings`, which makes it possible to create random Hexagon and Aztec diamond tilings.
It is a translation of the Matlab program `MatlabTilings` by Christophe Charlier, which is based on the shuffling algorithm as described in  [arXiv:0111034](https://arxiv.org/abs/math/0111034).
The original Matlab implementation can be found on his [hompage](https://sites.google.com/view/cchar/home). We are grateful to Christophe Charlier for
allowing us to make this program publicly available.

<p align="center">
<img width="383" height="389" alt="image" src="https://github.com/user-attachments/assets/b3d983e2-561e-461c-8fda-cfd6c933a24d" />
<img width="340" height="389" alt="image" src="https://github.com/user-attachments/assets/50255924-afd4-46ee-9942-39400836810f" />
</p>




Let us begin with an important note. The Python program uses the Numba package to compile certain
routines, resulting in a drastic performance improvement. This package needs to be installed to be
able to run the program. The disadvantage of this package is that the type-handling is more delicate
than is usually the case in Python; please be aware of this fact. In addition, the packages NumPy and
Matplotlib are also required.
In the current version, not all options from MatlabTilings have been implemented in the Python
program. Most notably, the program only produces images of tilings, and not of the corresponding
non-intersecting path systems. This document mainly discusses the necessary changes important for
the Python program. Some parts of the Help file for MatlabTilings are copied for context. For a
more complete description, the reader should consult the Help file; please keep in mind that only some
features are available in the Python program.
