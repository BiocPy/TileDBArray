import numpy as np
import tiledb
from tempfile import mkdtemp

from tiledbarray import TileDbArraySeed, TileDbArray
import os
import delayedarray
from scipy import sparse as sp

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_TileDbArraySeed_dense():
    test_shape = (100, 200)
    y = np.random.rand(*test_shape)

    dir = os.path.join(mkdtemp(), "test")
    data = tiledb.from_numpy(f"{dir}", y)

    roundtrip = TileDbArraySeed(dir, attribute_name="")
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype

    assert delayedarray.is_sparse(roundtrip) == False

    slices = (slice(3, 5), [4, 5, 6, 7, 8, 9])
    ref = y[slices]
    assert (
        np.allclose(delayedarray.extract_dense_array(roundtrip, (*slices,)), ref)
        is True
    )

    assert np.allclose(delayedarray.to_dense_array(roundtrip), y) is True


def test_TileDbArraySeed_sparse():
    dir = os.path.join(mkdtemp(), "sparse_array")
    dom = tiledb.Domain(
        tiledb.Dim(name="rows", domain=(0, 4), tile=4, dtype=np.int32),
        tiledb.Dim(name="cols", domain=(0, 4), tile=4, dtype=np.int32),
    )
    schema = tiledb.ArraySchema(
        domain=dom, sparse=True, attrs=[tiledb.Attr(name="", dtype=np.int32)]
    )
    tiledb.SparseArray.create(f"{dir}", schema)

    tdb = tiledb.SparseArray(f"{dir}", mode="w")
    i, j = [1, 2, 2], [1, 4, 3]
    data = np.array(([1, 2, 3]))
    tdb[i, j] = data

    roundtrip = TileDbArraySeed(dir, attribute_name="")
    assert roundtrip.shape == (5, 5)

    assert delayedarray.is_sparse(roundtrip) == True

    slices = (slice(0, 2), [2, 3])
    _msubset = delayedarray.extract_sparse_array(roundtrip, (*slices,))
    assert isinstance(_msubset, delayedarray.SparseNdarray)
    assert _msubset.contents == None

    _full = delayedarray.to_scipy_sparse_matrix(roundtrip, "coo")
    assert isinstance(_full, sp.spmatrix)
    assert np.allclose(_full.data, [1, 3, 2])


def test_TileDbArray():
    test_shape = (100, 200)
    y = np.random.rand(*test_shape)

    dir = os.path.join(mkdtemp(), "test")
    data = tiledb.from_numpy(f"{dir}", y)

    roundtrip = TileDbArray(dir, attribute_name="")
    assert roundtrip.shape == y.shape
    assert roundtrip.dtype == y.dtype

    assert delayedarray.is_sparse(roundtrip) == False

    slices = (slice(3, 5), [4, 5, 6, 7, 8, 9])
    ref = y[slices]
    assert (
        np.allclose(delayedarray.extract_dense_array(roundtrip, (*slices,)), ref)
        is True
    )

    assert np.allclose(delayedarray.to_dense_array(roundtrip), y) is True
