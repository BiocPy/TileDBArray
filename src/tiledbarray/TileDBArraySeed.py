from typing import Tuple

import tiledb
from delayedarray import extract_dense_array, extract_sparse_array, is_sparse
from numpy import dtype

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

        _attr_schema = _schema.attr(self._name)
        self._dtype = _attr_schema.dtype

    @property
    def dtype(self) -> dtype:
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
            Name of the dataset inside the file.
        """
        return self._name

    @property
    def is_sparse(self) -> bool:
        """
        Returns:
            Whether the Array is sparse.
        """
        return self._is_sparse


@is_sparse.register
def is_sparse_TileDbArraySeed(x: TileDbArraySeed):
    """See :py:meth:`~delayedarray.is_sparse.is_sparse`."""
    return x.is_sparse
