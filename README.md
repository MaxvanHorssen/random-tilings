Random Tilings
==============

This project provides the Python package `RandomTilings`, which makes it possible to generate random tilings of the Aztec diamond and the hexagon for a given doubly periodic weighting.
It is a translation of the Matlab program `MatlabTilings` by Christophe Charlier, which is based on the shuffling algorithm as described in  [arXiv:0111034](https://arxiv.org/abs/math/0111034).
The original Matlab implementation can be found on his [homepage](https://sites.google.com/view/cchar/home). We are grateful to Christophe Charlier for
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
We would like to express our sincere gratitude to Christophe Charlier for providing the original MATLAB implementation and for granting us permission to release this Python adaptation of his program on random tilings.

# How to get started
The package `RandomTilings` makes heavy use of:
 - numpy
 - matplotlib
 - numba

Therefore, make sure that these libraries are preinstalled. Once the installation is successful you only have to copy the folder `RandomTilings` in the same folder as your Python script or Jupyter notebook and you can import the routines by simply calling:
```python
 from RandomTilings import *
```
Note that by importing all necessary subroutines will be compiled by `numba.jit`, therefore it might take some time.

If you are working in a Jupyter notebook it is recommended to use the following import:
```python
from RandomTilings import *
import matplotlib.pyplot as plt
plt.figure()
plt.close()
_ = plt.ioff()
```
Doing so it will prevent the issue that each image is plotted twice.

# Example - How to Use
## Random Hexagon Tilings
To create Hexagon tilings we use the function: 
```python
draw_hexagon(n,w,a=1,b=1,c=1,edge=0,dpi=200,show_figure=True)
```
The inputs correspond to the following:
 - `n`           : positive integer, denoting the size of the tiling.
 - `w`           : 2d - numpy.ndarray, giving the weight used for the Hexagon tiling.
 - `a,b,c`       : positive integer, denotinge the side length multiplicator.
                   They can be used to create tilings of a special shape.
 - `edge`        : float, giving the edge width of the tilings. If `edge=0`, then no
                   edges will be displayed. 
 - `dpi`         : integer, defining the resolution used for the resulting figure.
                   Note, while using Jupyter notebook it will also increase the plot.
 - `show_figure` : If true, the plot will be shown atomatically.

Output:
 - `fig`         : `matplotlib.pyplot.figure` containing the plot. 

For more information see `Documentation for Random Tilings`.

### Uniform Weighting
```python
import numpy as np
import matplotlib.pyplot as plt
from RandomTilings import *

n     = 2
a,b,c = 1,2,3
w     = np.array([[1,1]]).T

fig = draw_hexagon(n,w,a,b,c,edge=1,dpi=100,show_figure=False)
plt.show()
```

<img width="460" height="389" alt="image" src="https://github.com/user-attachments/assets/cbe3e328-93b0-4cce-9e0a-6d78c5c2c4d5" />

Alternatively, you can also make use of the `show_figure` option:

```python
import numpy as np
import matplotlib.pyplot as plt
from RandomTilings import *

n     = 2                                   # Side length
a,b,c = 1,1,1                               # Side length multiplicator
w     = np.array([[1,1]]).T                 # initializing uniform weight

fig = draw_hexagon(n,w,a,b,c,edge=1,dpi=100)
```
<img width="460" height="389" alt="image" src="https://github.com/user-attachments/assets/b40cdc8c-5e12-4aca-a698-aea479cabe60" />

### Periodically Weighting
Here we will use non-trivial weights `w`.

```python
import numpy as np
import matplotlib.pyplot as plt
from RandomTilings import *

n     = 120                                     # Side length
a,b,c = 1,1,1                                   # Side length multiplicator

# Constructing the periodic weight
alpha1 = 0.3
alpha2 = 0.3
w=np.array([[1,1,1],[1,1,1],[1,1,1],[1/alpha1,alpha2,
            alpha1/alpha2],[1,1,1],[alpha1,1/alpha2,alpha2/alpha1]])

# Creating the random Hexagon tiling
fig = draw_hexagon(n,w,1,1,1,dpi=100)
```
<img width="340" height="389" alt="image" src="https://github.com/user-attachments/assets/f868f0bf-3530-434e-9c41-5a105b908b78" />

## Random Hexagon Tilings with Gaps
In order to create a Hexagon tiling with a gap we call:
```python
draw_hexagon_gap(n,w,gap,a=1,b=1,c=1,edge=0,dpi=200,show_figure=True)
```
The inputs correspond to the following:
 - `n`           : positive integer, denoting the size of the tiling.
 - `w`           : 2d - numpy.ndarray, giving the weight used for the Hexagon tiling.
 - `gap`         : 2d - numpy.ndarray of shape (N,4), indicating the location and size of the gaps.
 - `a,b,c`       : positive integer, denotinge the side length multiplicator.
                   They can be used to create tilings of a special shape.
 - `edge`        : float, giving the edge width of the tilings. If `edge=0`, then no
                   edges will be displayed. 
 - `dpi`         : integer, defining the resolution used for the resulting figure.
                   Note, while using Jupyter notebook it will also increase the plot.
 - `show_figure` : If true, the plot will be shown atomatically.


Output:
 - `vec`         : a triple containing : `logZnNum` - the log of the partition function with gaps, `logZnDen` - the log of the partition function without gaps,
                  `logPn` - equals `logZnNum-logZnDen` corresponding to the log of the probability density to obeserve a tiling with specified gaps. 
 - `fig`         : `matplotlib.pyplot.figure` containing the plot.

For more information see `Documentation for Random Tilings`.

Example:
```python
import numpy as np
import matplotlib.pyplot as plt
from RandomTilings import *

n     = 1                                 # Side length
a,b,c = 120,120,80                        # Side length multiplicator
w     = np.array([[1,1]]).T               # Weight
gap   = np.array([[130,100.5,140.5]])     # Gap


# Creating the random Hexagon tiling with an gap
_,fig = draw_hexagon_gap(n,w,gap,a,b,c,dpi=100)
```
<img width="311" height="389" alt="image" src="https://github.com/user-attachments/assets/e163dbb5-70c3-4de8-94cb-6d6dbf041aaa" />

## Random Aztec Diamond
To create a random Aztec diamond we call the routine:
```python
draw_aztec(n, w, edge=0, dpi=200, rotated=True, show_figure=True)
```
The inputs correspond to the following:
 - `n`           : positive integer, denoting the size of the tiling.
 - `w`           : 2d - numpy.ndarray, giving the weight used for the Hexagon tiling.
 - `edge`        : float, giving the edge width of the tilings. If `edge=0`, then no
                   edges will be displayed. 
 - `dpi`         : integer, defining the resolution used for the resulting figure.
                   Note, while using Jupyter notebook it will also increase the plot.
 - `rotated`     : bool, if set to `False` the Aztec diamond will not be rotated and will appear as a 'regular square'.
 - `show_figure` : If true, the plot will be shown atomatically.


Output:
 - `fig`         : `matplotlib.pyplot.figure` containing the plot.

For more information see `Documentation for Random Tilings`.

### Uniform Weighting
Here in the more common way to plot it.
```python
import numpy as np
import matplotlib.pyplot as plt
from RandomTilings import *

n = 30                                # Size
w = np.array([[1]]).T                 # Weight

# reating the random tiling for the Aztec diamond, without rotating the image.
fig = draw_aztec(n,w,edge=1,dpi=100) 
```
<img width="383" height="389" alt="image" src="https://github.com/user-attachments/assets/9be46e59-0748-4d63-a01c-94242e9f0d27" />


In case one wants it non-rotated:
```python
import numpy as np
import matplotlib.pyplot as plt
from RandomTilings import *

n = 30                                # Size
w = np.array([[1]]).T                 # Weight

# reating the random tiling for the Aztec diamond, without rotating the image.
fig = draw_aztec(n,w,edge=1,dpi=100,rotated=False) 
```
<img width="389" height="389" alt="image" src="https://github.com/user-attachments/assets/4b4a17da-9ab8-4c33-8a26-43d660a00f5f" />

## Random Aztec Diamond with Gaps
To create a random Aztec diamond with gaps we call:
```python
draw_aztec_gap(n, w, gap, edge=0, dpi=200, rotated=True, show_figure=True)
```
The inputs correspond to the following:
 - `n`           : positive integer, denoting the size of the tiling.
 - `w`           : 2d - numpy.ndarray, giving the weight used for the Hexagon tiling.
 - `gap`         : 2d - numpy.ndarray of shape (N,4), indicating the location and size of the gaps.
 - `edge`        : float, giving the edge width of the tilings. If `edge=0`, then no
                   edges will be displayed. 
 - `dpi`         : integer, defining the resolution used for the resulting figure.
                   Note, while using Jupyter notebook it will also increase the plot.
 - `rotated`     : bool, if set to `False` the Aztec diamond will not be rotated and will appear as a 'regular square'.
 - `show_figure` : If true, the plot will be shown atomatically.

Output:
 - `vec`         : a triple containing : `logZnNum` - the log of the partition function with gaps, `logZnDen` - the log of the partition function without gaps,
                  `logPn` - equals `logZnNum-logZnDen` corresponding to the log of the probability density to obeserve a tiling with specified gaps. 
 - `fig`         : `matplotlib.pyplot.figure` containing the plot.

   
For more information see `Documentation for Random Tilings`.

Example:
```python
import numpy as np
import matplotlib.pyplot as plt
from RandomTilings import *

n   = 300                               # Size
w   = np.array([[1]]).T                 # Weight
gap = np.array([[2*150-1,80,150]])      # Gap

# reating the random tiling for the Aztec diamond with gap.
_,fig = draw_aztec_gap(n,w,gap,dpi=100) 
```
<img width="389" height="389" alt="image" src="https://github.com/user-attachments/assets/bbda479c-e3b1-4a94-a48b-b3cad4a8d0c8" />


