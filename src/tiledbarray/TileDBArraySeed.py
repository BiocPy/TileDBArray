from typing import List, Sequence, Tuple

import numpy
import tiledb
from delayedarray import (
    SparseNdarray,
    chunk_grid,
    chunk_shape_to_grid,
    extract_dense_array,
    extract_sparse_array,
    is_masked,
    is_sparse,
    wrap,
)

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class TileDbArraySeed:
    """TileDB-backed dataset as a ``DelayedArray`` array seed."""

    def __init__(self, path: str, name: str) -> None:
        """
        Args:
            path:
                Path or URI to the TileDB file.

            name:
                Attribute name inside the TileDB file that contains the array.
        """
        self._path = path
        self._name = name

        _schema = tiledb.ArraySchema.load(self._path)

        self._is_sparse = _schema.sparse
        self._shape = _schema.shape

        _all_attr = []
        for i in range(_schema.nattr):
            _all_attr.append(_schema.attr(i).name)

        if self._name not in _all_attr:
            raise ValueError(f"Attribute '{self._name}' not in the tiledb schema.")

        _attr_schema = _schema.attr(self._name)
        self._dtype = _attr_schema.dtype

        _all_dimnames = []
        _all_dimnames_tile = []
        for i in range(_schema.domain.ndim):
            _dim = _schema.attr(i)
            _all_dimnames.append(_dim.name)
            _all_dimnames_tile.append(_dim.tile)

        self._dimnames = _all_dimnames
        self._tiles = _all_dimnames_tile

    @property
    def dtype(self) -> numpy.dtype:
        """
        Returns:
            NumPy type of this array.
        """
        return self._dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns:
            Tuple containing the dimensions of this array.
        """
        return self._shape

    @property
    def path(self) -> str:
        """
        Returns:
            Path to the HDF5 file.
        """
        return self._path

    @property
    def name(self) -> str:
        """
        Returns:
            Attribute name inside the TileDB file that contains the array.
        """
        return self._name

    @property
    def is_sparse(self) -> bool:
        """
        Returns:
            Whether the Array is sparse.
        """
        return self._is_sparse

    @property
    def dimnames(self) -> List[str]:
        """
        Returns:
            Names of each dimension of the matrix.
        """
        return self._dimnames


@chunk_grid.register
def chunk_grid_TileDbArraySeed(x: TileDbArraySeed):
    """
    See :py:meth:`~delayedarray.chunk_grid.chunk_grid`.

    The cost factor is set to 20 to reflect the computational work involved in
    extracting data from disk.
    """
    return chunk_shape_to_grid(x._tiles, x._shape, cost_factor=20)


@is_sparse.register
def is_sparse_TileDbArraySeed(x: TileDbArraySeed):
    """See :py:meth:`~delayedarray.is_sparse.is_sparse`."""
    return x.is_sparse


@is_masked.register
def is_masked_TileDbArraySeed(x: TileDbArraySeed):
    """See :py:meth:`~delayedarray.is_masked.is_masked`."""
    return False


def _extract_array(x: TileDbArraySeed, subset: Tuple[Sequence[int], ...]):

    _first_subset = subset[0]
    if _first_subset == slice(None):
        _first_subset = slice(x._shape[0])

    _second_subset = slice(x._shape[1])
    if len(subset) > 1:
        _second_subset = subset[1]
        if _second_subset == slice(None):
            _second_subset = slice(x._shape[1])

    with tiledb.open(x._path, "r") as mat:
        _data = mat.multi_index[subset]
        if x.is_sparse:
            output = numpy.zeros(
                (len(_first_subset), len(_second_subset)), dtype=x.dtype, order="F"
            )

            for idx, ival in enumerate(_data[x._name]):
                output[_data[x._dimnames[0]][idx], _data[x._dimnames[0]][idx][1]] = ival

            return output

        return (len(_first_subset), len(_second_subset)), numpy.array(_data[x._name])


@extract_dense_array.register
def extract_dense_array_TileDbArraySeed(
    x: TileDbArraySeed, subset: Tuple[Sequence[int], ...]
) -> numpy.ndarray:
    """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`.

    subset parameter is passed to tiledb's
    `multi_index operation <https://tiledb-inc-tiledb.readthedocs-hosted.com/projects/tiledb-py/en/stable/python-api.html#tiledb.libtiledb.Array.multi_index>`__.
    """
    _, _output = _extract_array(x, subset)
    return _output


@extract_sparse_array.register
def extract_sparse_array_TileDbArraySeed(
    x: TileDbArraySeed, subset: Tuple[Sequence[int], ...]
) -> SparseNdarray:
    """See :py:meth:`~delayedarray.extract_sparse_array.extract_sparse_array`.

    subset parameter is passed to tiledb's
    `multi_index operation <https://tiledb-inc-tiledb.readthedocs-hosted.com/projects/tiledb-py/en/stable/python-api.html#tiledb.libtiledb.Array.multi_index>`__.
    """
    _subset, _output = _extract_array(x, subset)

    return SparseNdarray(
        shape=_subset,
        contents=_output,
        dtype=x._dtype,
        index_dtype=x._dtype,
        check=False,
    )
