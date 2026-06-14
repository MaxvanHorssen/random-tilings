Random tilings
==============

This project provides the Python package `RandomTilings`, which makes it possible to generate random tilings of the Aztec diamond and the hexagon for doubly periodic weightings. This package is a Python adaptation of the MATLAB program `MatlabTilings` by Christophe Charlier, which is based on the shuffling algorithm as described in [arXiv:0111034](https://arxiv.org/abs/math/0111034). The original MATLAB implementation can be found on his [homepage](https://sites.google.com/view/cchar/home). We are grateful to Christophe Charlier for allowing us to make this package publicly available.

<p align="center">
<img width="383" height="389" alt="image" src="https://github.com/user-attachments/assets/b3d983e2-561e-461c-8fda-cfd6c933a24d" />
<img width="340" height="389" alt="image" src="https://github.com/user-attachments/assets/50255924-afd4-46ee-9942-39400836810f" />
</p>

# Important Notes
The documentation for the routines provided by `RandomTilings` can be found in the PDF `Documentation for RandomTilings`.  In the current version, not all options from `MatlabTilings` have been implemented in the Python package. Therefore, `Documentation for RandomTilings` mainly discusses the necessary changes important to the Python package. Some parts of the [`Help file`](https://sites.google.com/view/cchar/random-tilings)  for `MatlabTilings` are copied for context. For a more complete description, the reader should consult the [`Help file`](https://sites.google.com/view/cchar/random-tilings); please keep in mind that only some features are available in the Python package.

Some examples using the routines from the Python package can be found in the Jupyter notebook [`Examples - How to use`](https://github.com/MaxvanHorssen/random-tilings/blob/main/Examples%20-%20How%20to%20use.ipynb) or in the `How to use` section below.

**The Power of Numba**      
The creation of these random tilings is numerically expensive. In order to drastically improve the performance, this package makes heavy use of the `Numba` compiler. Therefore, installation of `Numba` is required to be able to run the package. Note that importing `RandomTilings` might take some time, as all subroutines have to be compiled.

**The New Version**
Version 2 introduces a new optimized implementation of the core algorithms. The main improvement is a reduction in memory complexity from cubic to quadratic, resulting in significantly better memory efficiency and improved runtime performance. Overall, Version 2 is roughly 25 times faster and it can handle bigger tilings. In addition, the structure of the main object has changed: instead of using a single function call to generate a plot, Version 2 follows an object-oriented design. This makes it possible to adjust plot settings or create new instances of a random tiling without recomputing everything from scratch. The previous implementation remains available in the GitHub branch `version-1`, but future development and support will focus on Version 2.
<p align="center">
<img width="789" height="290" alt="image" src="https://github.com/user-attachments/assets/6c0b6376-e3da-4ed5-a7f2-2d4d8699efbc" />
</p>


# Acknowledgments
We would like to express our sincere gratitude to Christophe Charlier for providing the original Matlab implementation and for giving us permission to release this Python adaptation of his program.


# How to get started
The package `RandomTilings` makes heavy use of:
 - NumPy
 - Matplotlib
 - Numba

Additionally, the `RandomTilings` requires the following modules:
 - tqdm
 - os
 - re

Therefore, make sure that these libraries are installed beforehand. Once the installation is successful, you only have to copy the folder `RandomTilings` in the same folder as your Python script or Jupyter notebook, and you can import the routines by simply calling:
```python
 import RandomTilings as RT
```
Note that by importing all necessary subroutines will be compiled by `Numba.njit`, therefore it might take some time.


# Example - How to use
Using `RandomTilings`, we can generate random tilings of both the Aztec diamond and the hexagon. The workflow is the same in both cases.
First, we create an `RT` object, short for “random tiling.” This object represents the random model from which individual tilings can be sampled. Once the model, or equivalently the `RT` object, has been constructed, we can draw random tilings from it and then plot them.
In the literature, the procedure of sampling a random tiling from the model is usually called *shuffling*, which is why the corresponding method uses this name.
The typical workflow is:

1. Create the model.
2. Repeat:
   - Shuffle the model to generate a random tiling.
   - Plot the resulting tiling.
3. Close the model.

As example:
```python
import numpy as np
import RandomTilings as RT
n = 100
w = np.array([[1]])

A = RT.Aztec(100,w)
A.shuffle()
A.plot()
```
The steps `shuffle` and `plot` can then be repeated. The `matplotlib.Figure` object corresponding to the plot can be accessed via `A.fig`. Once finished you can use `A.close()` to close the `RT` object and free memory.


## Random Tilings of a Hexagon
To create a random tiling of the hexagon call:
```python
H = RT.Hexagon(n,w,a=1,b=1,c=1,gap=False)
H.shuffle()
H.plot()
```
The inputs correspond to the following:
 - `n`           : positive integer, denoting the size of the tiling.
 - `w`           : 2d - numpy.ndarray, giving the weighting  used for the hexagon tiling.
 - `a,b,c`       : positive integer, denoting the side length multiplicator.
                   They can be used to create tilings of a special shape.
 - `gap`         : 2d - numpy.ndarray of shape (N,4), indicating the location and size of the gaps.
   
Optional plotting parameters can be found by calling `H.plot?` or `help(H.plot)`. For more information see `Documentation for RandomTilings`.

### Uniform Weighting 
```python
import numpy as np
import RandomTilings as RT

n     = 2
a,b,c = 1,2,3
w     = np.array([[1,1]]).T

H = RT.Hexagon(n,w,a,b,c)
H.shuffle()
H.plot(edge=1)
```
<p align="center">
<img width="460" height="389" alt="image" src="https://github.com/user-attachments/assets/cbe3e328-93b0-4cce-9e0a-6d78c5c2c4d5" />
</p>
 

### Doubly Periodic Weighting 
Here we will use non-trivial weightings `w`.

```python
import numpy as np
import RandomTilings

n     = 120                                     # Side length

# Constructing the periodic weight
alpha1 = 0.3
alpha2 = 0.3
w=np.array([[1,1,1],[1,1,1],[1,1,1],[1/alpha1,alpha2,
            alpha1/alpha2],[1,1,1],[alpha1,1/alpha2,alpha2/alpha1]])

# Creating the random tiling of the hexagon
H = RT.Hexagon(n,w)
H.shuffle()
H.plot()
```
<p align="center">
<img width="340" height="389" alt="image" src="https://github.com/user-attachments/assets/f868f0bf-3530-434e-9c41-5a105b908b78" />
</p>

### Random Tilings of the Hexagon with Gaps

```python
import numpy as np
import RandomTilings as RT

n     = 1                                 # Side length
a,b,c = 120,120,80                        # Side length multiplicator
w     = np.array([[1,1]]).T               # Weighting
gap   = np.array([[130,100.5,140.5]])     # Gap

# Creating the random tiling of the hexagon with a gap
H = RT.Hexagon(n,w,a,b,c,gap)
H.shuffle()
H.plot()
```
<p align="center">
<img width="311" height="389" alt="image" src="https://github.com/user-attachments/assets/e163dbb5-70c3-4de8-94cb-6d6dbf041aaa" />
</p>


## Random Tilings of the Aztec Diamond
To create a random tiling of the Aztec diamond, call the routine:
```python
A = RT.Aztec(n,w,gap=False)
```
The inputs correspond to the following:
 - `n`           : positive integer, denoting the size of the tiling.
 - `w`           : 2d - numpy.ndarray, giving the weighting  used for the hexagon tiling.
 - `gap`         : 2d - numpy.ndarray of shape (N,4), indicating the location and size of the gaps.

Optional plotting parameters can be found by calling `A.plot?` or `help(A.plot)`. For more information see `Documentation for RandomTilings`.


### Uniform Weighting
Here in the more common way to plot it.
```python
import numpy as np
import RandomTilings as RT

n = 30                                # Side length
w = np.array([[1]]).T                 # Weighting matrix

# Creating the random tiling of the Aztec diamond, with rotating the image
A = RT.Aztec(n,w)
A.shuffle()
A.plot(edge=1)
```
<p align="center">
<img width="383" height="389" alt="image" src="https://github.com/user-attachments/assets/9be46e59-0748-4d63-a01c-94242e9f0d27" />
</p>

In case one wants it non-rotated:
```python
import numpy as np
import RandomTilings as RT

n = 30                                # Side length
w = np.array([[1]]).T                 # Weighting matrix

# Creating the random tiling of the Aztec diamond, without rotating the image
A = RT.Aztec(n,w)
A.shuffle()
A.plot(edge=1,rotated=False)
```

<p align="center">
<img width="389" height="389" alt="image" src="https://github.com/user-attachments/assets/4b4a17da-9ab8-4c33-8a26-43d660a00f5f" />
</p>

### Random Tilings of the Aztec Diamond with Gaps
To create a random tiling of the Aztec diamond with gaps, call the routine:
```python
import numpy as np
import RandomTilings as RT

n   = 300                               # Side length
w   = np.array([[1]]).T                 # Weighting matrix
gap = np.array([[2*150-1,80,150]])      # Gap

# Creating the random tiling of the Aztec diamond with a gap
A = RT.Aztec(n,w,gap)
A.shuffle()
A.plot()
```
<p align="center">
<img width="389" height="389" alt="image" src="https://github.com/user-attachments/assets/bbda479c-e3b1-4a94-a48b-b3cad4a8d0c8" />
</p>
