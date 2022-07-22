import typing as typ

import numpy as np
from utils import FiniteDimMeasurementOpCstrctr

import pycsou._dev.fw_utils as pycdevu
import pycsou.abc.solver as pycs
import pycsou.opt.solver as pycsol
import pycsou.opt.stop as pycos
import pycsou.runtime as pycrt
import pycsou.util.ptype as pyct


class GenericFWforBLasso(pycs.Solver):
    r"""
    Abstract base class.

    Notes
    _____
    :py:var:`data` should be real-valued.

    If the parameter :py:var:`optim_strategy` is set to "grid", the size of the gris must be specified using
    `grid_size=n`.
    """

    def __init__(
        self,
        data: pyct.NDArray,
        forwardOpCstrctr: FiniteDimMeasurementOpCstrctr,
        lambda_: float,
        *,
        optim_strategy: str = "grid",
        folder: typ.Optional[pyct.PathLike] = None,
        exist_ok: bool = False,
        writeback_rate: typ.Optional[int] = None,
        verbosity: int = 1,
        show_progress: bool = True,
        log_var: pyct.VarName = (
            "positions",
            "weights",
            "dcv",
            "ofv",
        ),
        **kwargs,
    ):
        super().__init__(
            folder=folder,
            exist_ok=exist_ok,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            show_progress=show_progress,
            log_var=log_var,
        )
        self.lambda_ = pycrt.coerce(lambda_)
        self.forwardOpCstrctr = forwardOpCstrctr
        self.data = pycrt.coerce(data)
        try:
            assert optim_strategy in ["grid", "stochastic"]
            self._optim_strategy = optim_strategy
        except:
            raise ValueError(f"optim_strategy must be in ['grid', 'stochastic'], got {optim_strategy}.")

        self.domain_dim = self.forwardOpCstrctr.domain_dim  # d
        self.dstream_support = np.stack(
            [-self.forwardOpCstrctr.support_width / 2, self.forwardOpCstrctr.support_width / 2]
        ).T  # (d, 2)

        self._bound = 0.5 * pycdevu.SquaredL2Norm()(data)[0] / self.lambda_
        self._sampling_scheme = None
        self.grid_size = kwargs.pop("grid_size", 50)
        self.n_random_samples = kwargs.pop("n_random_samples", 2500)
        self.seed = kwargs.pop("seed", None)

    def m_init(self, **kwargs):
        # todo: Maybe I should put the optim_strategy within the .fit(...) ?
        mst = self._mstate

        mst["positions"] = np.zeros((0, self.domain_dim))
        mst["weights"] = np.zeros(0)
        mst["dcv"] = None  # TODO this is not true, do we need to compute it ? check the indices of the iterations
        mst["ofv"] = 0.5 * pycdevu.SquaredL2Norm()(self.data)

    def default_stop_crit(self) -> pycs.StoppingCriterion:
        stop_crit = pycos.RelError(
            eps=1e-4,
            var="ofv",
            f=None,
            norm=2,
            satisfy_all=True,
        )
        return stop_crit

    def solution(self) -> pyct.NDArray:
        """
        Returns
        -------
        p: NDArray
            (..., N) solution.
        """
        data, _ = self.stats()
        return data.get("positions"), data.get("weights")

    def new_atoms_locations(self) -> typ.Tuple[np.ndarray, float]:
        self.design_scheme()
        samples = self.evaluate_candidates()
        return self.extract_new_locations(samples)

    def design_scheme(self):
        r"""
        Updates the attribute :py:attribute: sampling_scheme (if needed).
        """
        if self._optim_strategy == "grid":
            if self._sampling_scheme is None:
                self._sampling_scheme = self.d_dim_grid()
        elif self._optim_strategy == "stochastic":
            self._sampling_scheme = self.stochastic_eval_points()

    def d_dim_grid(self):
        grid_ticks = np.linspace(
            self.dstream_support[:, 0], self.dstream_support[:, 1], num=self.grid_size + 1, endpoint=False
        )[
            1:, :
        ].transpose()  # (d, grid_size)
        flat_grid = np.array(np.meshgrid(*grid_ticks)).T.reshape(-1, self.domain_dim)  # (grid_size**2, d)
        return flat_grid

    def stochastic_eval_points(self):
        rng = np.random.default_rng(self.seed)
        return rng.uniform(
            low=self.dstream_support[:, 0],
            high=self.dstream_support[:, 1],
            size=(self.n_random_samples, self.domain_dim),
        )

    def evaluate_candidates(self) -> np.ndarray:
        r"""
        Evaluates the empirical dual certificate over the sampling scheme.

        Returns
        -------
        np.ndarray
            Values of the certificate at over the sampling scheme.

        Notes
        _____
        If the sampling scheme is modified, it should update the attribute, so that the indices match with the indices
        of the output.
        """
        forwardOp = self.forwardOpCstrctr.fixedKnotsForwardOp(self._mstate["positions"])
        adjointOp = self.forwardOpCstrctr.fixedEvaluationPointsAdjointOp(self._sampling_scheme)
        return adjointOp(self.data - forwardOp(self._mstate["weights"])) / self.lambda_

    def extract_new_locations(self, samples: np.ndarray) -> typ.Tuple[np.ndarray, float]:
        r"""
        The behavior of this method depends on the Vanilla or Polyatomic version of the algorithm.

        Parameters
        ----------
        samples: np.ndarray
            Values of the empirical dual certificate over the sampling scheme.

        Returns
        -------
        np.ndarray
            The location(s) of the new atom(s).
        float
            The associated values of the certificate.
        """
        raise NotImplementedError

    def evaluate_objective(self) -> np.ndarray:
        forward = self.forwardOpCstrctr.fixedKnotsForwardOp(self._mstate["positions"])
        return 0.5 * pycdevu.SquaredL2Norm()(
            self.data - forward(self._mstate["weights"])
        ) + self.lambda_ * pycdevu.L1Norm()(
            self._mstate["weights"]
        )  # (1,)

    def partial_correction(self, locations: np.ndarray, init_weights: np.ndarray, eps: float) -> np.ndarray:
        def correction_stop_crit(eps) -> pycs.StoppingCriterion:
            stop_crit = pycos.RelError(
                eps=eps,
                var="x",
                f=None,
                norm=2,
                satisfy_all=True,
            )
            return stop_crit

        forward = self.forwardOpCstrctr.fixedKnotsForwardOp(locations)
        forward.lipschitz()
        data_fid = 0.5 * pycdevu.SquaredL2Norm().asloss(self.data) * forward
        regul = self.lambda_ * pycdevu.L1Norm()
        apgd = pycsol.PGD(data_fid, regul, show_progress=False)
        apgd.fit(x0=init_weights, stop_crit=correction_stop_crit(eps))
        sol, _ = apgd.stats()
        return sol["x"]


class VanillaFWforBLasso(GenericFWforBLasso):
    def __init__(
        self,
        data: pyct.NDArray,
        forwardOpCstrctr: FiniteDimMeasurementOpCstrctr,
        lambda_: float,
        *,
        step_size: str = "regular",
        optim_strategy: str = "grid",
        folder: typ.Optional[pyct.PathLike] = None,
        exist_ok: bool = False,
        writeback_rate: typ.Optional[int] = None,
        verbosity: int = 1,
        show_progress: bool = True,
        log_var: pyct.VarName = (
            "positions",
            "weights",
            "dcv",
            "ofv",
        ),
        **kwargs,
    ):
        super().__init__(
            data=data,
            forwardOpCstrctr=forwardOpCstrctr,
            lambda_=lambda_,
            optim_strategy=optim_strategy,
            folder=folder,
            exist_ok=exist_ok,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            show_progress=show_progress,
            log_var=log_var,
            **kwargs,
        )
        try:
            assert step_size in ["regular", "line_search", "optimal"]
            self._step_size_strategy = step_size
        except:
            raise ValueError(f"step_size must be in ['regular', 'line_search', 'optimal'], got {step_size}.")

    def m_init(self, **kwargs):
        super().m_init(**kwargs)
        self._mstate["lift_variable"] = 0.0

    def m_step(self):
        mst = self._mstate
        new_loc, dcv = self.new_atoms_locations()
        mst["dcv"] = dcv

        if self._step_size_strategy == "regular":
            gamma = 2 / (2 + self._astate["idx"])

        elif self._step_size_strategy == "line_search":
            forwardOp = self.forwardOpCstrctr.fixedKnotsForwardOp(mst["positions"])
            adjointOp = self.forwardOpCstrctr.fixedEvaluationPointsAdjointOp(mst["positions"])
            gamma = -np.dot(adjointOp(self.data - forwardOp(mst["weights"])), mst["weights"]).real
            if abs(dcv) > 1.0:
                gamma += self.lambda_ * (mst["lift_variable"] + (abs(dcv) - 1.0) * self._bound)
                pass
                gamma /= pycdevu.SquaredL2Norm()(
                    self._bound
                    * np.sign(dcv)
                    * self.forwardOpCstrctr.fixedKnotsForwardOp(new_loc.reshape((1, -1)))(np.array([1.0]))
                    - forwardOp(mst["weights"])
                )[0]
            else:
                gamma += self.lambda_ * mst["lift_variable"]
                gamma /= pycdevu.SquaredL2Norm()(forwardOp(mst["weights"]))[0]
        else:
            # Fully corrective step
            pass
        mst["weights"] *= 1 - gamma
        mst["lift_variable"] *= 1 - gamma

        if abs(dcv) > 1.0:  # we add a new atom
            if mst["positions"] is None:  # previously empty solution
                mst["positions"] = new_loc.reshape((1, -1))
                mst["weights"] = gamma * self._bound * np.sign(dcv)
            else:  # check if the candidate location has already been visited
                test_presence = np.all(new_loc == mst["positions"], axis=1)
                if not np.any(test_presence):
                    mst["positions"] = np.vstack([mst["positions"], new_loc])
                    mst["weights"] = np.append(mst["weights"], gamma * self._bound * np.sign(dcv))
                else:
                    idx_presence = np.where(test_presence)[0][0]
                    mst["weights"][idx_presence] += gamma * self._bound * np.sign(dcv)
            mst["lift_variable"] += gamma * self._bound

        mst["ofv"] = self.evaluate_objective()

    def extract_new_locations(self, samples: np.ndarray):
        idx = np.argmax(np.abs(samples))
        new_loc = self._sampling_scheme[idx, :]
        dcv = samples[idx]
        return new_loc, dcv


class PolyatomicFWforBLasso(GenericFWforBLasso):
    def __init__(
        self,
        data: pyct.NDArray,
        forwardOpCstrctr: FiniteDimMeasurementOpCstrctr,
        lambda_: float,
        ms_threshold: float = 0.7,  # multi spikes threshold at init
        init_correction_prec: float = 0.2,
        final_correction_prec: float = 1e-4,
        *,
        optim_strategy: str = "grid",
        folder: typ.Optional[pyct.PathLike] = None,
        exist_ok: bool = False,
        writeback_rate: typ.Optional[int] = None,
        verbosity: int = 1,
        show_progress: bool = True,
        log_var: pyct.VarName = (
            "positions",
            "weights",
            "dcv",
            "ofv",
        ),
        **kwargs,
    ):
        super().__init__(
            data=data,
            forwardOpCstrctr=forwardOpCstrctr,
            lambda_=lambda_,
            optim_strategy=optim_strategy,
            folder=folder,
            exist_ok=exist_ok,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            show_progress=show_progress,
            log_var=log_var,
            **kwargs,
        )
        self._ms_threshold = ms_threshold
        self._init_correction_prec = init_correction_prec
        self._final_correction_prec = final_correction_prec
        self._delta = None

    def m_init(self, **kwargs):
        super().m_init(**kwargs)
        self._mstate["correction_prec"] = self._init_correction_prec

    def m_step(self):
        mst = self._mstate
        new_loc, dcv = self.new_atoms_locations()
        mst["dcv"] = dcv

        if abs(dcv) > 1.0:  # we add a new atom
            if mst["positions"] is None:  # previously empty solution
                mst["positions"] = new_loc
            else:  # check if the candidate location has already been visited
                test_presence = np.all(new_loc[:, None, :] == mst["positions"][None, :, :], axis=2)
                if not np.any(test_presence):
                    mst["positions"] = np.vstack([mst["positions"], new_loc])
                else:
                    idx_not_present = np.where(np.invert(np.any(test_presence, axis=1)))[0]
                    mst["positions"] = np.vstack([mst["positions"], new_loc[idx_not_present]])

        if mst["positions"].shape[0] == 1:
            pass  # epxlicit solution to derive
        else:
            init_weights = np.zeros(mst["positions"].shape[0])
            if mst["weights"] is not None:
                init_weights[: mst["weights"].shape[0]] = mst["weights"]
            mst["correction_prec"] = max(self._init_correction_prec / self._astate["idx"], self._final_correction_prec)

            mst["weights"] = self.partial_correction(mst["positions"], init_weights, mst["correction_prec"])

        mst["ofv"] = self.evaluate_objective()

    def extract_new_locations(self, samples: np.ndarray):
        dcv = max(samples.max(), samples.min(), key=abs)
        maxi = abs(dcv)
        if self._astate["idx"] == 1:
            self._delta = maxi * (1.0 - self._ms_threshold)
        thresh = maxi - (2 / self._astate["idx"] + 1) * self._delta
        new_indices = (abs(samples) > max(thresh, 1.0)).nonzero()[0]
        new_loc = self._sampling_scheme[new_indices, :].reshape((-1, 2))
        return new_loc, dcv
