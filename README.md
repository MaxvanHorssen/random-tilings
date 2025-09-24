Random tilings
==============

This project provides the Python package `Randomtilings`, which makes it possible to generate random tilings of the Aztec diamond and the hexagon for doubly periodic weighting s.
It is a translation of the Matlab program `Matlabtilings` by Christophe Charlier, which is based on the shuffling algorithm as described in  [arXiv:0111034](https://arxiv.org/abs/math/0111034).
The original Matlab implementation can be found on his [homepage](https://sites.google.com/view/cchar/home). We are grateful to Christophe Charlier for
allowing us to make this package publicly available.

<p align="center">
<img width="383" height="389" alt="image" src="https://github.com/user-attachments/assets/b3d983e2-561e-461c-8fda-cfd6c933a24d" />
<img width="340" height="389" alt="image" src="https://github.com/user-attachments/assets/50255924-afd4-46ee-9942-39400836810f" />
</p>

# Important Notes
A documentation for the the routines provided by `Randomtilings` can be found in the pdf `Documentation for Randomtilings`. 
In the current version, not all options from `Matlabtilings` have been implemented in the Python
package. Most notably, the package only produces images of tilings, and not of the corresponding
non-intersecting path systems. Therefore, `Documentation for RandomTilings` mainly discusses the necessary changes important for
the Python package. Some parts of the [`Help file`](https://sites.google.com/view/cchar/random-tilings)  for `Matlabtilings` are copied for context. For a
more complete description, the reader should consult the [`Help file`](https://sites.google.com/view/cchar/random-tilings); please keep in mind that only some
features are available in the Python package.

Some examples using the routines from the Python package can be found in the Jupyter notebook `Examples - How to use` or in the `How to use` section below.

**The Power of Numba**      
The creation of these random tilings is numerically expensive. In order to drastically improve the performance, this package makes heavy use of the `Numba` compiler `jit`.
Therefore, installation of `Numba` is required to be able to run the package. Note that importing `Randomtilings` might take some time, as all subroutines have to be compiled.

# Acknowledgments
We would like to express our sincere gratitude to Christophe Charlier for providing the original Matlab implementation and for giving us permission to release this Python adaptation of his program.

# How to get started
The package `Randomtilings` makes heavy use of:
 - NumPy
 - Matplotlib
 - Numba

Therefore, make sure that these libraries are installed beforehand. Once the installation is successful, you only have to copy the folder `Randomtilings` in the same folder as your Python script or Jupyter notebook, and you can import the routines by simply calling:
```python
 from RandomTilings import *
```
Note that by importing all necessary subroutines will be compiled by `Numba.jit`, therefore it might take some time.

If you are working in a Jupyter notebook it is recommended to use the following import:
```python
from RandomTilings import *
import matplotlib.pyplot as plt
plt.figure()
plt.close()
_ = plt.ioff()
```
Doing so it will prevent the issue that each image is plotted twice.

# Example - How to use
## Random tilings of a hexagon
To create random tilings of a hexagon, use the function: 
```python
draw_hexagon(n,w,a=1,b=1,c=1,edge=0,dpi=200,show_figure=True)
```
The inputs correspond to the following:
 - `n`           : positive integer, denoting the size of the tiling.
 - `w`           : 2d - numpy.ndarray, giving the weighting  used for the hexagon tiling.
 - `a,b,c`       : positive integer, denotinge the side length multiplicator.
                   They can be used to create tilings of a special shape.
 - `edge`        : float, giving the edge width of the tilings. If `edge=0`, then no
                   edges will be displayed. 
 - `dpi`         : integer, defining the resolution used for the resulting figure.
                   Note, while using Jupyter notebook it will also increase the plot.
 - `show_figure` : If true, the plot will be shown atomatically.

Output:
 - `fig`         : `matplotlib.pyplot.figure` containing the plot. 

For more information see `Documentation for RandomTilings`.

### Uniform weighting 
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
<p align="center">
<img width="460" height="389" alt="image" src="https://github.com/user-attachments/assets/cbe3e328-93b0-4cce-9e0a-6d78c5c2c4d5" />
</p>
 
Alternatively, you can also make use of the `show_figure` option:

```python
import numpy as np
import matplotlib.pyplot as plt
from RandomTilings import *

n     = 2                                   # Side length
a,b,c = 1,1,1                               # Side length multiplicator
w     = np.array([[1,1]]).T                 # initializing uniform weighting

fig = draw_hexagon(n,w,a,b,c,edge=1,dpi=100)
```
<p align="center">
<img width="460" height="389" alt="image" src="https://github.com/user-attachments/assets/b40cdc8c-5e12-4aca-a698-aea479cabe60" />
</p>

### Doubly periodic weighting 
Here we will use non-trivial weightings `w`.

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

# Creating the random tiling of the hexagon
fig = draw_hexagon(n,w,1,1,1,dpi=100)
```
<p align="center">
<img width="340" height="389" alt="image" src="https://github.com/user-attachments/assets/f868f0bf-3530-434e-9c41-5a105b908b78" />
</p>

## Random tilings of the hexagon with gaps
In order to create a random tiling of a hexagon with gaps, call:
```python
draw_hexagon_gap(n,w,gap,a=1,b=1,c=1,edge=0,dpi=200,show_figure=True)
```
The inputs correspond to the following:
 - `n`           : positive integer, denoting the size of the tiling.
 - `w`           : 2d - numpy.ndarray, giving the weighting used for the hexagon tiling.
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

For more information see `Documentation for RandomTilings`.

Example:
```python
import numpy as np
import matplotlib.pyplot as plt
from RandomTilings import *

n     = 1                                 # Side length
a,b,c = 120,120,80                        # Side length multiplicator
w     = np.array([[1,1]]).T               # Weighting
gap   = np.array([[130,100.5,140.5]])     # Gap

# Creating the random tiling of the hexagon with a gap
_,fig = draw_hexagon_gap(n,w,gap,a,b,c,dpi=100)
```
<p align="center">
<img width="311" height="389" alt="image" src="https://github.com/user-attachments/assets/e163dbb5-70c3-4de8-94cb-6d6dbf041aaa" />
</p>

## Random tilings of the Aztec diamond
To create a random tiling of the Aztec diamond, call the routine:
```python
draw_aztec(n,w,edge=0,dpi=200,rotated=True,show_figure=True)
```
The inputs correspond to the following:
 - `n`           : positive integer, denoting the size of the tiling.
 - `w`           : 2d - numpy.ndarray, giving the weighting  used for the hexagon tiling.
 - `edge`        : float, giving the edge width of the tilings. If `edge=0`, then no
                   edges will be displayed. 
 - `dpi`         : integer, defining the resolution used for the resulting figure.
                   Note, while using Jupyter notebook it will also increase the plot.
 - `rotated`     : bool, if set to `False` the Aztec diamond will not be rotated and will appear as a 'regular square'.
 - `show_figure` : If true, the plot will be shown atomatically.

Output:
 - `fig`         : `matplotlib.pyplot.figure` containing the plot.

For more information see `Documentation for RandomTilings`.

### Uniform weighting 
Here in the more common way to plot it.
```python
import numpy as np
import matplotlib.pyplot as plt
from RandomTilings import *

n = 30                                # Size
w = np.array([[1]]).T                 # Weighting

# Creating the random tiling of the Aztec diamond, without rotating the image.
fig = draw_aztec(n,w,edge=1,dpi=100) 
```
<p align="center">
<img width="383" height="389" alt="image" src="https://github.com/user-attachments/assets/9be46e59-0748-4d63-a01c-94242e9f0d27" />
</p>

In case one wants it non-rotated:
```python
import numpy as np
import matplotlib.pyplot as plt
from RandomTilings import *

n = 30                                # Size
w = np.array([[1]]).T                 # Weighting

## Random tilings of the Aztec diamond with gaps
fig = draw_aztec(n,w,edge=1,dpi=100,rotated=False) 
```

<p align="center">
<img width="389" height="389" alt="image" src="https://github.com/user-attachments/assets/4b4a17da-9ab8-4c33-8a26-43d660a00f5f" />
</p>

### Random tilings of the Aztec diamond with gaps
To create a random tiling of the Aztec diamond with gaps, call:
```python
draw_aztec_gap(n,w,gap,edge=0,dpi=200,rotated=True,show_figure=True)
```

The inputs correspond to the following:
 - `n`           : positive integer, denoting the size of the tiling.
 - `w`           : 2d - numpy.ndarray, giving the weighting  used for the hexagon tiling.
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

For more information see `Documentation for RandomTilings`.

Example:
```python
import numpy as np
import matplotlib.pyplot as plt
from RandomTilings import *

n   = 300                               # Size
w   = np.array([[1]]).T                 # Weighting
gap = np.array([[2*150-1,80,150]])      # Gap

# Creating the random tiling of the Aztec diamond with a gap.
_,fig = draw_aztec_gap(n,w,gap,dpi=100) 
```
<p align="center">
<img width="389" height="389" alt="image" src="https://github.com/user-attachments/assets/bbda479c-e3b1-4a94-a48b-b3cad4a8d0c8" />
</p>
