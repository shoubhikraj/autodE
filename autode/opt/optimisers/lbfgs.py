"""
Gradient-only Limited-memory BFGS optimiser algorithm,
suitable for large molecular systems where calculating
or storing Hessian would be difficult.
"""
import numpy as np
from typing import Optional
from collections import deque
from autode.opt.coordinates import CartesianCoordinates
from autode.opt.optimisers.base import NDOptimiser
from autode.log import logger


class LBFGSOptimiser(NDOptimiser):
    """
    L-BFGS optimisation in Cartesian coordinates
    """

    def __init__(
        self,
        *args,
        max_step=0.2,
        max_vecs: int = 15,
        h0: np.ndarray = None,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self.alpha = max_step
        self._max_vecs = abs(int(max_vecs))
        self._s = deque(maxlen=max_vecs)  # stores changes in coords -2D
        self._y = deque(maxlen=max_vecs)  # stores changes in grad - 2D
        self._rho = deque(maxlen=max_vecs)  # stores 1/(y.T @ s) - 1D
        self._h0: Optional[np.ndarray] = h0  # initial 1D diagonal of Hess

    def _initialise_run(self) -> None:
        # NOTE: While LBFGS could be used for internal coordinates, it is
        # practically pointless because to store the matrices required to
        # convert between internal <-> cartesian would require almost the
        # same amount of memory and effort as calculating Hessian
        self._coords = CartesianCoordinates(self._species.coordinates)
        self._update_gradient_and_energy()
        n_atoms = self._species.n_atoms
        self._s = np.zeros(shape=(self._max_vecs, n_atoms))
        self._y = np.zeros(shape=(self._max_vecs, n_atoms))
        self._rho = np.zeros(shape=(self._max_vecs,))
        if self._h0 is not None and self._h0.shape == (self._max_vecs,):
            assert isinstance(self._h0, np.ndarray)
        else:
            self._h0 = np.ones(shape=(self._max_vecs,))
        return None

    def _step(self) -> None:

        # NOTE: To reduce memory operations, self._s, self._y and self._rho are used
        # in a cyclic manner i.e. when max_vecs rows are filled up, it reuses
        # row 0 and over-writes the information there, removing that datapoint
        # from the list of stored vectors. row_k points to the current row
        # that should be written to.
        if self.iteration >= 1:
            # todo check above formula is it correct for python or fortran
            s = self._coords - self._history[-2]
            y = self._coords.g - self._history[-2].g
            y_s = np.dot(y, s)
            if abs(y_s) < 1.0e-16:
                logger.warning("Resetting y_s in LBFGS to 1.0")
                y_s = 1.0
            # save in memory
            self._s.append(s)
            self._y.append(y)
            self._rho.append(1.0 / y_s)
            # update the hessian diagonal
            y_norm = np.linalg.norm(y)
            if abs(y_norm) < 1.0e-16:
                logger.warning("Resetting y_norm in LBFGS to 1.0")
            gamma = y_s / (y_norm**2)
            self._h0 *= gamma
            step = _get_lbfgs_step_py(
                self._coords.g,
                self._s,
                self._y,
                self._rho,
                self._h0,
                self.iteration,
            )
        else:
            g_size = np.linalg.norm(self._coords.g)
            step = self._h0 * self._coords.g
            step *= min(g_size, 1 / g_size)

        logger.info("Taking an L-BFGS step")
        self._coords = self._coords + step


# todo port this to cython after finishing checking
def _get_lbfgs_step_py(
    grad: np.ndarray,
    s_matrix: deque,
    y_matrix: deque,
    rho_array: deque,
    hess_diag: np.ndarray,
    iteration: int,
):
    max_vecs = len(s_matrix)
    q = grad.copy()
    a = np.zeros(max_vecs)
    # todo double check the formulas
    if iteration <= max_vecs:
        iter_range = range(0, iteration)
    else:
        iter_range = range(0, max_vecs)

    for i in reversed(iter_range):
        # todo fix this!! i is not the same as iter_range?
        a[i] = rho_array[i] * np.dot(s_matrix[i], q)
        q -= a[i] * y_matrix[i]

    q *= hess_diag  # use q as working space for z
    for i in iter_range:
        beta = rho_array[i] * np.dot(y_matrix[i], q)
        q += (a[i] - beta) * s_matrix[i]

    return -q
