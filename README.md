[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![PyPI-Server](https://img.shields.io/pypi/v/tiledbarray.svg)](https://pypi.org/project/tiledbarray/)
[![Monthly Downloads](https://pepy.tech/badge/tiledbarray/month)](https://pepy.tech/project/tiledbarray)
![Unit tests](https://github.com/BiocPy/tiledbarray/actions/workflows/pypi-test.yml/badge.svg)

# tiledbarray


This is the Python equivalent of Bioconductor's [**TileDBArray**](https://bioconductor.org/packages/TileDBArray) package,
providing a representation of TileDB-backed arrays within the [**delayedarray**](https://github.com/BiocPy/delayedarray) framework.
The idea is to allow users to store, manipulate and operate on large datasets without loading them into memory,
in a manner that is trivially compatible with other data structures in the [**BiocPy**](https::/github.com/BiocPy) ecosystem.

## Installation

This package can be installed from [PyPI](https://pypi.org/project/tiledbarray/) with the usual commands:

```shell
pip install tiledbarray
```

## Quick start

Let's mock up a dense array:

```python
import numpy
data = numpy.random.rand(40, 50)

tiledb.from_numpy("dense.tiledb", data)
```

We can now represent it as a `TileDbArray`:

```python
import tiledbarray
arr = tiledbarray.TileDbArray("dense.tiledb", attribute_name="")
# <40 x 50> TileDbArray object of type 'float64'
# [[0.96316214, 0.90187013, 0.55767551, ..., 0.81663263, 0.57660051,
#   0.3986336 ],
#  [0.72578394, 0.06328588, 0.9473141 , ..., 0.89977069, 0.34617884,
#   0.09208036],
#  [0.87291607, 0.01714908, 0.96570953, ..., 0.28404601, 0.20394673,
#   0.6454273 ],
#  ...,
#  [0.21565857, 0.11721607, 0.45146332, ..., 0.18565937, 0.348599  ,
#   0.16050929],
#  [0.95061188, 0.71917657, 0.33039149, ..., 0.60267692, 0.28035863,
#   0.56416845],
#  [0.40462116, 0.61058508, 0.5067807 , ..., 0.64234988, 0.5881812 ,
#   0.17138409]]
```

This is just a subclass of a `DelayedArray` and can be used anywhere in the BiocPy framework.
Parts of the NumPy API are also supported - for example, we could apply a variety of delayed operations:

```python
scaling = numpy.random.rand(100)
transformed = numpy.log1p(arr / scaling)
# <40 x 50> DelayedArray object of type 'float64'
# [[1.29646391, 2.05014167, 0.48661736, ..., 0.90574803, 2.38890685,
#   1.1277655 ],
#  [1.09916863, 0.38865342, 0.72500505, ..., 0.96463182, 1.93797807,
#   0.39371608],
#  [1.22596458, 0.12107778, 0.73496894, ..., 0.41384292, 1.50457489,
#   1.47747976],
#  ...,
#  [0.46673182, 0.63114795, 0.41040352, ..., 0.28897665, 1.94394461,
#   0.61032586],
#  [1.28695229, 1.85595293, 0.31579293, ..., 0.73604123, 1.76033915,
#   1.37526146],
#  [0.74949037, 1.71968269, 0.45082104, ..., 0.76976215, 2.40698455,
#   0.64080734]]
```

Check out the [documentation](https://biocpy.github.io/tiledbarray/) for more details.

## Sparse Matrices

We can perform similar operations on a sparse matrix as well. Lets mock a sparse matrix and store it as a tiledb file.

```python
dir_path = "sparse_array.tiledb"
dom = tiledb.Domain(
     tiledb.Dim(name="rows", domain=(0, 4), tile=5, dtype=np.int32),
     tiledb.Dim(name="cols", domain=(0, 4), tile=5, dtype=np.int32),
)
schema = tiledb.ArraySchema(
     domain=dom, sparse=True, attrs=[tiledb.Attr(name="", dtype=np.int32)]
)
tiledb.SparseArray.create(f"{dir_path}", schema)

tdb = tiledb.SparseArray(f"{dir_path}", mode="w")
i, j = [1, 2, 2], [1, 4, 3]
data = np.array(([1, 2, 3]))
tdb[i, j] = data
```

We can now represent this as a `TileDbArray`:

```python
import tiledbarray
arr = tiledbarray.TileDbArray(dir_path, attribute_name="")

slices = (slice(0,3), [2, 4])

import delayedarray
subset = delayedarray.extract_sparse_array(arr, (*slices,))
print(subset)
# <3 x 2> SparseNdarray object of type 'int32'
# [[2, 0],
#  [0, 0],
#  [0, 0]]
```

Check out the [delayedarray](https://biocpy.github.io/delayedarray/) for more details.

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
