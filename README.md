Random Tilings
==============

This project provides the Python package `RandomTilings`, which makes it possible to generate random tilings of the Aztec diamond and the hexagon for doubly periodic weightings. This package is a Python adaptation of the MATLAB program `MatlabTilings` by Christophe Charlier, which is based on the [domino shuffling algorithm](https://arxiv.org/abs/math/0111034). The original MATLAB implementation can be found on his [webpage](https://sites.google.com/view/cchar/random-tilings). We are grateful to Christophe Charlier for allowing us to make this package publicly available.

<p align="center">
<img width="383" height="389" alt="image" src="https://github.com/user-attachments/assets/b3d983e2-561e-461c-8fda-cfd6c933a24d" />
<img width="340" height="389" alt="image" src="https://github.com/user-attachments/assets/50255924-afd4-46ee-9942-39400836810f" />
</p>

## The New Version
The new version introduces an optimized implementation of the core algorithms. The main improvement is a reduction in memory complexity from cubic to quadratic, resulting in significantly better memory efficiency, and improved runtime performance. In addition, the structure of the main object has changed: instead of using a single function call to generate a plot, the new version follows an object-oriented design. This makes it possible to adjust plot settings or create new instances of a random tiling without recomputing everything from scratch. The previous implementation remains available in the GitHub branch [`version-0`](https://github.com/MaxvanHorssen/random-tilings/tree/version-0). Future development and support will focus on the main branch.

## Documentation
The documentation for the routines provided by `RandomTilings` is available in [`Documentation`](https://github.com/MaxvanHorssen/random-tilings/blob/main/Documentation.pdf). Not every feature of the original MATLAB implementation is currently available in the Python package. For a more comprehensive description of the original MATLAB implementation, consult the [`MatlabTilings help file`](https://drive.google.com/file/d/119W_nJ6amyrWDoBiC_xm9X1kyvuhz4L2/view). Additional examples are provided in the Jupyter notebooks [`Basic examples`](https://github.com/MaxvanHorssen/random-tilings/blob/main/Basic%20examples.ipynb) and [`Advanced examples`](https://github.com/MaxvanHorssen/random-tilings/blob/main/Advanced%20examples.ipynb).

## Acknowledgments
We would like to express our sincere gratitude to Christophe Charlier for providing the original Matlab implementation and for allowing us to release this Python adaptation of his program.

## How to get started
The package `RandomTilings` requires to install the following packages:
 - NumPy
 - Matplotlib
 - ipympl
 - ipywidgets
 - Numba
 - tqdm

Make sure that these libraries are installed beforehand. Once the installation is successful, you only have to copy the folder `RandomTilings` in the same folder as your Python script or Jupyter notebook. Then, you can import the routines by simply calling:
```python
 import RandomTilings as RT
```
Note that by importing all necessary subroutines will be compiled by `Numba.njit`, therefore importing the package for the first time might take some time.

## How to use the package
Using `RandomTilings`, we can generate random tilings of both the Aztec diamond and the hexagon. The workflow is the same in both cases.
First, we create an `RT` object, short for “random tiling”. This object represents the model from which individual random tilings can be sampled. Once the model, or equivalently the `RT` object, has been constructed, we can sample random tilings from it and then plot them.

The typical workflow is:
1. Create the model.
2. Repeat:
   - Shuffle the model to generate a random tiling.
   - Plot the resulting tiling.
3. Close the model.

## Minimal examples

**Random Tilings of the Aztec Diamond**     
```python
import RandomTilings as RT
A = RT.Aztec(100)
A.shuffle()
A.plot()
```
The steps `shuffle` and `plot` can then be repeated. The `matplotlib.figure` object corresponding to the plot can be accessed via `A.fig`. Once finished, the object can be deleted by calling `A.close()`, which frees the memory used by the `RT` object.

**Random Tilings of the Hexagon**
```python
import RandomTilings as RT
H = RT.Hexagon(100)
H.shuffle()
H.plot()
H.close()
```
