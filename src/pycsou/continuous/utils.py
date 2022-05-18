import typing as typ

import numpy as np

import pycsou.abc as pycabc
import pycsou.util.ptype as pyct

# Measurement operators are defined as semi infinite-dimensional operators, InfDNDLinOp

SemiInfDShape = typ.Union[typ.Tuple[int, np.infty], typ.Tuple[np.infty, int]]


class InfDAdjoint(pycabc.Property):
    r"""
    Mixin class defining the *adjoint* property for InfDNDLinOp.
    """

    def adjoint(self, arr: pyct.NDArray) -> typ.Callable:
        r"""
        When evaluating the adjoint of a InfDNDLinOp at a given point ``arr``, the output is a function. This method returns that function.

        **Note:**
        The output function belongs to the predual of the class of functions handled by the InfDNDLinOp. When dealing
        with Radon measures, the evaluation of the adjoint of a semi infinite-dimensional operator is a continuous
        function over the same domain.

        Parameters
        ----------
        arr: NDArray
            (...,N) input.

        Returns
        -------
        Callable
            A Callable function.

        """
        raise NotImplementedError


class SemiInfDLinOp(pycabc.DiffMap, InfDAdjoint):
    r"""
    Base class for semi infinite-dimensional linear operators.
    The Lipschitz constant must be manually provided.

    Closely inspired from LinOp class, some methods may be lacking or others non-appropriate for infinite-dimensional domain.
    """

    def __init__(self, shape: SemiInfDShape, lipschitz_cst: float):
        r"""

        Parameters
        ----------
        shape: tuple(int, int)
           Shape of the linear operator (N, np.infty) or (np.infty, N).

        Notes
        -----
        Since linear maps have constant Jacobians (see :py:meth:`~pycsou.abc.operator.LinOp.jacobian`), the private instance attribute ``_diff_lipschitz`` is initialized to zero by this method.
        """
        super(SemiInfDLinOp, self).__init__((2, None))
        self._shape = shape
        self._diff_lipschitz = 0
        self._lipschitz = lipschitz_cst

    def jacobian(self, arr: pyct.NDArray) -> "InfDNDLinOp":
        return self

    @property
    def T(self) -> "SemiInfDLinOp":
        raise NotImplementedError


class NDInfDLinOp(SemiInfDLinOp):
    r"""
    Class for semi infinite-dimensional linear operators NDInfDLinOp :math:`L:\mathbb{C}^L\to\mathcal{M}(\mathcal{X})`.
    """

    @property
    def T(self) -> "InfDNDLinOp":
        r"""
        Return the adjoint of the linear operator.

        Returns
        -------
        :py:class:`~pycsou.abc.operator.LinOp`
            Adjoint of the linear operator.
        """
        adj = InfDNDLinOp(shape=self.shape[::-1])
        adj.apply = self.adjoint
        adj.adjoint = self.apply
        adj._lipschitz = self._lipschitz
        return adj


class InfDNDLinOp(SemiInfDLinOp):
    r"""
    Class for semi infinite-dimensional linear operators InfDNDLinOp :math:`L:\mathcal{M}(\mathcal{X})\to\mathbb{C}^L`.
    """

    @property
    def T(self) -> "NDInfDLinOp":
        r"""
        Return the adjoint of the linear operator.

        Returns
        -------
        :py:class:`~pycsou.abc.operator.LinOp`
            Adjoint of the linear operator.
        """
        adj = NDInfDLinOp(shape=self.shape[::-1])
        adj.apply = self.adjoint
        adj.adjoint = self.apply
        adj._lipschitz = self._lipschitz
        return adj
