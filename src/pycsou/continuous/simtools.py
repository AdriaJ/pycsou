import numpy as np
from utils import FiniteDimMeasurementOpCstrctr

import pycsou.operator.linop as pyclop


class NUFSamplingCstrctr(FiniteDimMeasurementOpCstrctr):
    def __init__(self, frequencies: np.ndarray, support_width):
        self.frequencies = frequencies  # shape (L, d)
        super(NUFSamplingCstrctr, self).__init__(domain_dim=frequencies.shape[1], support_width=support_width)

    def fixedKnotsForwardOp(self, knots: np.ndarray):
        r"""

        Parameters
        ----------
        knots: np.ndarray
            Knots of the input Dirac stream, shape (Q, d).

        Returns
        -------
        pyclop.ExplicitLinOp
            Linear operator with explicit expression and shape (2*L, Q).
        """
        tmp = (self.frequencies[:, :, None] * knots.transpose()[None, :, :]).sum(axis=1)
        mat = np.exp((-2j * np.pi) * tmp)
        return pyclop.ExplicitLinOp(np.vstack([mat.real, mat.imag]))

    def fixedEvaluationPointsAdjointOp(self, evaluation_points: np.ndarray):
        r"""

        Parameters
        ----------
        evaluation_points: np.ndarray
            Points at which the function resulting from the call of the adjoint operator should be evaluated,
            shape (P, d).

        Returns
        -------
        pyclop.ExplicitLinOp
            Linear operator with explicit expression and shape (P, 2*L).
        """
        evaluation_points = evaluation_points.reshape((-1, self.domain_dim))
        tmp = (evaluation_points[:, :, None] * self.frequencies.transpose()[None, :, :]).sum(axis=1)
        mat = np.exp((2j * np.pi) * tmp)  # shape (P, L)
        return pyclop.ExplicitLinOp(np.hstack([mat.real, -mat.imag]))
