import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
    "hutchpp",
    "norm",
    "trace",
]


def norm(x: pxt.NDArray, **kwargs):
    """
    This function is identical to :py:func:`numpy.linalg.norm`.
    It exists to correct bugs in Dask's implementation.
    """
    xp = pxu.get_array_module(x)
    nrm = xp.linalg.norm(x, **kwargs)
    nrm = nrm.astype(x.dtype, copy=False)  # dask bug: linalg.norm() always returns float64
    return nrm


def trace(
    op: pxa.SquareOp,
    xp: pxt.ArrayModule = None,
    dtype: pxt.DType = None,
) -> pxt.Real:
    r"""
    Exact trace of a linear operator based on multiple evaluation of the forward operator.

    Parameters
    ----------
    op: ~pyxu.abc.operator.SquareOp
    xp: ArrayModule
        Array module used for internal computations. (Default: NumPy.)
    dtype: DType
        Precision to use for internal computations. (Default: current runtime precision.)

    Returns
    -------
    tr: Real
        Exact value of tr(op).
    """
    if xp is None:
        xp = pxd.NDArrayInfo.default().module()

    if dtype is None:
        dtype = pxrt.getPrecision().value
    width = pxrt.Width(dtype)

    tr = 0
    for i in range(op.dim):
        e = xp.zeros(op.dim, dtype=dtype)
        e[i] = 1
        with pxrt.Precision(width):
            tr += op.apply(e)[i]
    return float(tr)


def hutchpp(
    op: pxa.SquareOp,
    m: pxt.Integer = 4002,
    xp: pxt.ArrayModule = None,
    dtype: pxt.DType = None,
    seed: pxt.Integer = None,
) -> pxt.Real:
    r"""
    Stochastic trace estimation of a linear operator based on the Hutch++ algorithm.
    (Specifically `algorithm 3 from this paper <https://arxiv.org/abs/2010.09649>`_.)

    Parameters
    ----------
    op: ~pyxu.abc.operator.SquareOp
    m: Integer
        Number of queries used to estimate the trace of the linear operator.

        `m` is set to 4002 by default based on the analysis of the variance described in theorem 10.
        This default corresponds to having an estimation error smaller than 0.01 with probability 0.9.
    xp: ArrayModule
        Array module used for internal computations. (Default: NumPy.)
    dtype: DType
        Precision to use for internal computations. (Default: current runtime precision.)
    seed: Integer
        Seed for the random number generator.

    Returns
    -------
    tr: Real
        Stochastic estimate of tr(op).
    """
    if xp is None:
        xp = pxd.NDArrayInfo.default().module()

    if dtype is None:
        dtype = pxrt.getPrecision().value
    width = pxrt.Width(dtype)

    rng = xp.random.default_rng(seed=seed)
    s = rng.standard_normal(size=(op.dim, (m + 2) // 4), dtype=dtype)
    g = rng.integers(0, 2, size=(op.dim, (m - 2) // 2)) * 2 - 1

    with pxrt.Precision(width):
        data = op.apply(s.T).T

    kwargs = dict(mode="reduced")
    if xp == pxd.NDArrayInfo.DASK.module():
        data = data.rechunk({0: "auto", 1: -1})
        kwargs.pop("mode", None)

    q, _ = xp.linalg.qr(data, **kwargs)
    proj = g - q @ (q.T @ g)

    tr = (op.apply(q.T) @ q).trace()
    tr += (2 / (m - 2)) * (op.apply(proj.T) @ proj).trace()
    return float(tr)
