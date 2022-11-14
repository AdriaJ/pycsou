import enum
import itertools

import numpy as np
import numpy.linalg as npl
import pytest
import scipy.ndimage as scimage

import pycsou.math.linalg as pylinalg
import pycsou.operator as pycob
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou_tests.operator.conftest as conftest

if pycd.CUPY_ENABLED:
    import cupy.linalg as cpl

import collections.abc as cabc


@enum.unique
class Boundaries(enum.Enum):

    NONE = "none"
    PERIODIC = "periodic"
    REFLECT = "reflect"
    NEAREST = "nearest"


# We disable PrecisionWarnings since Stencil() is not precision-agnostic, but the outputs
# computed must still be valid. We also disable NumbaPerformanceWarnings, as these appear
# due to using the overkill use of GPU for very small test input arrays.
@pytest.mark.filterwarnings(
    "ignore::pycsou.util.warning.PrecisionWarning", "ignore::numba.core.errors.NumbaPerformanceWarning"
)
class TestStencil(conftest.LinOpT):
    @pytest.fixture(
        params=itertools.product(
            [3], [2]  # [1, 3], # (1) 1D and (3) 3D kernels  # [0, 2]  # right or left centered kernels (wrt origin=1)
        )
    )
    def stencil_params(self, request):
        ndim, pos = request.param
        stencil_coefs = np.expand_dims(self._random_array((3,), seed=20), axis=list(np.arange(ndim - 1)))
        center = np.zeros(ndim, dtype=int)
        center[-1] = pos
        return stencil_coefs, center

    @pytest.fixture
    def stencil_coefs(self, stencil_params):
        return stencil_params[0]

    @pytest.fixture
    def center(self, stencil_params):
        return stencil_params[1]

    @pytest.fixture
    def arg_shape(self, stencil_params):
        return (6,) * stencil_params[0].ndim

    @pytest.fixture(params=pycd.NDArrayInfo)
    def ndi(self, request):
        return request.param

    @pytest.fixture(params=pycrt.Width)
    def width(self, request):
        return request.param

    @pytest.fixture(params=Boundaries)
    def boundary(self, request):
        return request.param

    @pytest.fixture
    def spec(
        self, stencil_coefs, center, arg_shape, ndi, width, boundary
    ) -> tuple[pyct.OpT, pycd.NDArrayInfo, pycrt.Width]:
        op = pycob.Stencil(
            stencil_coefs=ndi.module().asarray(stencil_coefs, dtype=width.value),
            center=center,
            arg_shape=arg_shape,
            boundary=boundary.value,
        )
        return op, ndi, width

    @pytest.fixture
    def data_apply(self, op) -> conftest.DataLike:
        arr = self._random_array((op.dim,), seed=20)  # random seed for reproducibility
        kernel, center = op.stencil_coefs, op.center
        xp = pycu.get_array_module(kernel)
        kernel = kernel.get() if xp.__name__ == "cupy" else kernel

        origin = [
            0,
        ] + list(center)
        origin[-1] -= 1

        mode = op._padding_kwargs["mode"][0]
        mode = mode if mode != "symmetric" else "reflect"  # Scipy and numpy "reflect" are not doing the same.
        mode = mode if mode != "edge" else "nearest"  # Numpy's edge is and Scipy's "nearest"

        out = scimage.correlate(
            arr.reshape(-1, *op.arg_shape), kernel.reshape(1, *kernel.shape), mode=mode, origin=origin, cval=0.0
        ).ravel()
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_adjoint(self, op) -> conftest.DataLike:
        arr = self._random_array((op.dim,), seed=20)  # random seed for reproducibility
        kernel, center = op.stencil_coefs, op.center
        xp = pycu.get_array_module(kernel)
        kernel = kernel.get() if xp.__name__ == "cupy" else kernel

        origin = [
            0,
        ] + list(center)
        origin[-1] -= 1
        mode = op._padding_kwargs["mode"][0]
        mode = mode if mode != "symmetric" else "reflect"  # Scipy and numpy "reflect" are not doing the same.
        mode = mode if mode != "edge" else "nearest"  # Numpy's edge is and Scipy's "nearest"
        if mode in ["reflect", "nearest"]:

            E = np.eye(op.dim)
            A = scimage.correlate(
                E.reshape(-1, *op.arg_shape), kernel.reshape(1, *kernel.shape), mode=mode, origin=origin, cval=0.0
            ).reshape(op.dim, op.dim)
            out = (A @ arr.T).T
        else:
            out = scimage.convolve(
                arr.reshape(-1, *op.arg_shape), kernel.reshape(1, *kernel.shape), mode=mode, origin=origin, cval=0.0
            ).ravel()
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_shape(self, arg_shape) -> pyct.Shape:
        size = np.prod(arg_shape)
        sh = (size, size)
        return sh

    @pytest.fixture
    def data_pinv(self, op, _damp, data_apply):

        arr = data_apply["out"]
        xp = pycu.get_array_module(arr)
        xpl = cpl if xp.__name__ == "cupy" else npl

        B = op.asarray(xp=xp, dtype=pycrt.Width.DOUBLE.value)
        A = B.T @ B
        # -------------------------------------------------
        for i in range(op.dim):
            A[i, i] += _damp

        out, *_ = xpl.lstsq(A, B.T @ arr)
        data = dict(
            in_=dict(
                arr=arr,
                damp=_damp,
                kwargs_init=dict(),
                kwargs_fit=dict(),
            ),
            out=out,
        )
        return data

    @pytest.fixture
    def data_pinvT(self, op, _damp, data_apply):
        arr = data_apply["in_"]["arr"]
        xp = pycu.get_array_module(arr)
        xpl = cpl if xp.__name__ == "cupy" else npl
        B = op.asarray(xp=xp, dtype=pycrt.Width.DOUBLE.value)
        A = B.T @ B
        # -------------------------------------------------
        for i in range(op.dim):
            A[i, i] += _damp

        out, *_ = xpl.lstsq(A, arr)
        out = B @ out
        data = dict(
            in_=dict(
                arr=arr,
                damp=_damp,
                kwargs_init=dict(),
                kwargs_fit=dict(),
            ),
            out=out,
        )
        return data

    @pytest.fixture
    def data_math_lipschitz(self, op) -> cabc.Collection[np.ndarray]:
        N_test = 5
        return self._random_array((N_test, op.dim))

    @pytest.fixture
    def data_math_diff_lipschitz(self, op) -> cabc.Collection[np.ndarray]:
        N_test = 5
        return self._random_array((N_test, op.dim))
