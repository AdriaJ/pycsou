import typing as typ

import numpy as np

import pycsou.abc as pycabc
import pycsou.util.ptype as pyct


class DiracStream:
    r"""
    Base class for any N-dimensional domain Dirac stream.
    """

    def __init__(self, positions: np.ndarray, weights: np.ndarray, support_width: np.ndarray, is_null: bool = False):
        r"""

        Parameters
        ----------
        positions
        weights
        support: np.ndarray
            Should be given as a set of intervals.
        is_null
        """
        self.is_null = is_null
        self.domain_dim = positions.shape[1]  # d
        self.positions = positions  # (_, d)
        self.weights = weights

        # the following attributes might be removed
        self.support_width = support_width  # (d,)
        self.support = np.stack([-self.support_width / 2, self.support_width / 2])  # (d, 2)

    @property
    def size(self):
        return self.weights.shape[0]

    def add_impulse(self, pos, weight):
        if self.is_null:
            self.is_null = False
        self.positions = np.vstack([self.positions, pos])
        self.weights = np.hstack([self.weights, weight])

    def add_position(self, new_pos: np.ndarray):
        new_pos = new_pos.reshape((-1, self.domain_dim))
        if self.is_null and len(new_pos) > 0:
            self.is_null = False
        self.positions = np.vstack([self.positions, new_pos])
        self.weights = np.append(self.weights, np.zeros(new_pos.shape[0]))  # 0 valued impulses

    def set_weights(self, weights: np.ndarray):
        assert weights.size == self.positions.shape[0]
        self.weights = weights
        if np.abs(weights).sum() == 0.0:
            self.is_null = True

    def dmin(self):
        distances = ((self.positions[..., np.newaxis, :] - self.positions[..., :, np.newaxis]) ** 2).sum(
            axis=0
        ) + self.domain_dim * np.max(self.support_width) ** 2 * np.eye(self.size)
        return distances.min()

    def convex_combination_extension(self, gamma: float, new_pos: np.ndarray, new_weight: float):
        if len(new_pos) > 0:
            self.is_null = False
        self.positions = np.vstack([self.positions, new_pos.reshape((1, -1))])
        self.weights = np.hstack([(1 - gamma) * self.weights, gamma * new_weight])


class TVNorm(pycabc.Func):
    def __init__(self):
        super(TVNorm, self).__init__(shape=None)

    def __call__(self, dirac_stream: DiracStream):
        if dirac_stream.is_null:
            return 0.0
        else:
            return np.linalg.norm(dirac_stream.weights, 1)


class FiniteDimMeasurementOpCstrctr:
    r"""
    Base class modelling for a semi-infinite dimensional operator. It provides two constructor functions, each of them
    is designed to instantiate finite dimensional linear operators, accounting for direct and adjoint calls of the
    source operator.

    Any instance/subclass of this class must implement the methods :meth:`.FiniteDimMeasurementOpCstrctr.fixedKnotsForwardOp`
    and :meth:`.FiniteDimMeasurementOpCstrctr.fixedEvaluationPointsAdjointOp`. Additional attributes and submethods may
    be implemented for subclasses if needed.
    """

    def __init__(self, domain_dim: int, support_width: np.ndarray):
        r"""

        Parameters
        ----------
        domain_dim: int
            Dimension of the domain of the input Radon measures.
        support_width: np.ndarray
            Dimensions of the support of the Dirac stream to reconstruct, should be provided as a d-dimensional array.
        """
        self.domain_dim = domain_dim  # d
        self.support_width = support_width  # (d, )

    def fixedKnotsForwardOp(self, knots: np.ndarray) -> pycabc.LinOp:
        r"""
        Models for the direct call of a linear operator applied to a Dirac stream once the knots of the stream have
        been set. Indeed, the call of the operator is still linear with respect to the weights of the stream.

        Parameters
        ----------
        knots: np.ndarray
            The fixed knots of the Dirac stream

        Returns
        -------
        pycsou.abc.operator.LinOp
            The output is an instance of a linear operator.
        """
        raise NotImplementedError

    def fixedEvaluationPointsAdjointOp(self, evaluation_points: np.ndarray) -> pycabc.LinOp:
        r"""
        Models for the adjoint call of a linear operator that operates on Radon measures. The adjoint operator is
        supposed to output a continuous function over the same domain. Once the evaluation points of the continuous
        function are set, the call of the adjoint is linear with respect to its input.

        Parameters
        ----------
        evaluation_points: np.ndarray
            The points at which the output continuous function is evaluated.

        Returns
        -------
        pycsou.abc.operator.LinOp
            The output is an instance of a linear operator.
        """
        raise NotImplementedError


#################  USELESS  ####################

# Tentative for SemiInfinite Dimensional operator (continuos-domain to discrete or vice versa).
# After discussion, this will probably be useless.

# Measurement operators are defined as semi infinite-dimensional operators, InfDNDLinOp

SemiInfDShape = typ.Union[typ.Tuple[int, float], typ.Tuple[float, int]]


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
