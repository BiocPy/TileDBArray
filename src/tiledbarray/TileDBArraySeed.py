from typing import List, Optional, Sequence, Tuple, Union

import numpy
import tiledb
from delayedarray import (
    DelayedArray,
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

    def __init__(self, path: str, attribute_name: str) -> None:
        """
        Args:
            path:
                Path or URI to the TileDB file.

            name:
                Attribute name inside the TileDB file that contains the array.
        """
        self._path = path
        self._attribute_name = attribute_name

        _schema = tiledb.ArraySchema.load(self._path)

        self._is_sparse = _schema.sparse
        self._shape = _schema.shape

        _all_attr = []
        for i in range(_schema.nattr):
            _all_attr.append(_schema.attr(i).name)

        if self._attribute_name not in _all_attr:
            raise ValueError(
                f"Attribute '{self._attribute_name}' not in the tiledb schema."
            )

        _attr_schema = _schema.attr(self._attribute_name)
        self._dtype = _attr_schema.dtype

        _all_dimnames = []
        _all_dimnames_tile = []
        for i in range(_schema.domain.ndim):
            _dim = _schema.domain.dim(i)
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
    def attribute_name(self) -> str:
        """
        Returns:
            Attribute name inside the TileDB file that contains the array.
        """
        return self._attribute_name

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
    """See :py:meth:`~delayedarray.chunk_grid.chunk_grid`.

    The cost factor is set to 20 to reflect the computational work involved in extracting data from disk.
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


def _sanitize_subset(subset, dimlength):
    if isinstance(subset, slice):
        if subset == slice(None):
            subset = slice(dimlength)

        subset = list(range(*subset.indices(dimlength)))
    elif isinstance(subset, range):
        subset = list(subset)

    return sorted(subset)


def _extract_array(x: TileDbArraySeed, subset: Tuple[Sequence[int], ...]):
    """Extract slices from a TileDB Array."""
    _parsed_subset = []

    _first_subset = _sanitize_subset(subset[0], x._shape[0])
    _parsed_subset.append(_first_subset)

    if len(subset) > 1:
        _second_subset = _sanitize_subset(subset[1], x._shape[1])
        _parsed_subset.append(_second_subset)
    else:
        _second_subset = _sanitize_subset(slice(x._shape[1]), x._shape[1])

    with tiledb.open(x._path, "r") as mat:
        _data = mat.multi_index[tuple(_parsed_subset)]
        if x.is_sparse is True:
            return (len(_first_subset), len(_second_subset)), (
                _data[x._dimnames[0]],
                _data[x._dimnames[1]],
                _data[x._attribute_name],
                _parsed_subset,
            )

        return (len(_first_subset), len(_second_subset)), numpy.array(
            _data[x._attribute_name]
        )


@extract_dense_array.register
def extract_dense_array_TileDbArraySeed(
    x: TileDbArraySeed, subset: Tuple[Sequence[int], ...]
) -> numpy.ndarray:
    """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`.

    Subset parameter is passed to tiledb's
    `multi_index operation <https://tiledb-inc-tiledb.readthedocs-hosted.com/projects/tiledb-py/en/stable/python-api.html#tiledb.libtiledb.Array.multi_index>`__.
    """
    _, _output = _extract_array(x, subset)
    return _output


def _SparseNdarray_contents_from_coordinates(
    rows, cols, vals, shape, val_dtype, parsed_subset
):
    output = [None] * shape[-1]
    for i, val in enumerate(vals):
        _offset_col = parsed_subset[1].index(cols[i])
        _offset_row = parsed_subset[0].index(rows[i])
        if output[_offset_col] is None:
            output[_offset_col] = [
                numpy.array([], dtype=numpy.int32),
                numpy.array([], dtype=val_dtype),
            ]

        output[_offset_col][0] = numpy.append(output[_offset_col][0], _offset_row)
        output[_offset_col][1] = numpy.append(output[_offset_col][1], val)

    for i, o in enumerate(output):
        if o is not None:
            _idx_order = numpy.argsort(o[0])
            _indices = o[0][_idx_order].astype(numpy.int32)
            _vals = o[1][_idx_order]
            output[i] = (_indices, _vals)

    if all([x is None for x in output]):
        output = None

    return output


@extract_sparse_array.register
def extract_sparse_array_TileDbArraySeed(
    x: TileDbArraySeed, subset: Tuple[Sequence[int], ...]
) -> SparseNdarray:
    """See :py:meth:`~delayedarray.extract_sparse_array.extract_sparse_array`.

    Subset parameter is passed to tiledb's
    `multi_index operation <https://tiledb-inc-tiledb.readthedocs-hosted.com/projects/tiledb-py/en/stable/python-api.html#tiledb.libtiledb.Array.multi_index>`__.
    """
    _subset_shape, _output = _extract_array(x, subset)

    _content = _SparseNdarray_contents_from_coordinates(
        _output[0],
        _output[1],
        _output[2],
        _subset_shape,
        x._dtype,
        _output[3],
    )

    return SparseNdarray(
        shape=_subset_shape, contents=_content, dtype=x._dtype, index_dtype=numpy.int32
    )


class TileDbArray(DelayedArray):
    """Sparse or Dense arrays from TileDB file as a ``DelayedArray``.

    This subclass allows developers to implement custom methods for tiledb-backed sparse or dense matrices.
    """

    def __init__(
        self,
        path: Union[str, TileDbArraySeed],
        attribute_name: Optional[str],
    ):
        """To construct a ``TileDbArray`` from an existing :py:class:`~TileDbArraySeed`, use
        :py:meth:`~delayedarray.wrap.wrap` instead.

        Args:
            path:
                Path to the TileDB file or a :class:`~TileDbArraySeed` object.

            attribute_name:
                Name of the attribute containing the array.
        """
        if isinstance(path, TileDbArraySeed):
            seed = path
        else:
            if attribute_name is None:
                raise ValueError("'attribute_name' cannot be 'None'.")

            seed = TileDbArraySeed(path, attribute_name)

        super(TileDbArray, self).__init__(seed)

    @property
    def path(self) -> str:
        """
        Returns:
            Path to the TileDB file.
        """
        return self.seed.path

    @property
    def attribute_name(self) -> Optional[str]:
        """
        Returns:
            Name of the TileDB attribute containing the matrix contents.
        """
        return self.seed.attribute_name


@wrap.register
def wrap_Hdf5CompressedSparseMatrixSeed(x: TileDbArraySeed):
    """See :py:meth:`~delayedarray.wrap.wrap`."""
    return TileDbArray(x, None)
