"""
Low-memory BFGS optimiser, for large dimensional optimisation
problems, where storing the Hessian would be difficult

References:
[1] J. Nocedal, Mathematics of Computation, 35, 1980, 773-782
[2] D. C. Liu, J. Nocedal, Mathematical Programming, 45, 1989, 503-528
[3] J. Nocedal, S. Wright, "Numerical Optimization", 2nd ed. Springer, 2006
"""
from typing import Optional, Deque, Union, TYPE_CHECKING
import numpy as np
from scipy.optimize import root_scalar
from collections import deque

from autode.values import Distance
from autode.opt.coordinates import CartesianCoordinates
from autode.opt.optimisers.base import NDOptimiser
from autode.log import logger


class LBFGSOptimiser(NDOptimiser):
    """
    L-BFGS optimiser in Cartesian coordinates
    """

    def __init__(
        self,
        *args,
        max_vecs: int = 20,
        init_trust: float = 0.1,
        **kwargs,
    ):
        """Initialise an L-BFGS optimiser"""
        super().__init__(*args, **kwargs)

        self._max_vecs = abs(int(max_vecs))
        self._trust = init_trust
        # storage for steps and gradient changes
        self._s: Deque[np.ndarray] = deque(maxlen=max_vecs)
        self._y: Deque[np.ndarray] = deque(maxlen=max_vecs)

    def _initialise_run(self) -> None:
        """Initialise the LBFGS run"""
        assert self._species is not None
        self._coords = CartesianCoordinates(self._species.coordinates)
        self._update_gradient_and_energy()
        # unit Hessian diagonal
        self._h_inv_d = np.ones(shape=self._coords.shape[0])
        return None

    def _update_lbfgs_storage(self):
        """
        Store the coordinate and gradient change in the last step in
        L-BFGS and update the Hessian diagonal
        """
        if self._first_step:
            return None

        coords_k = self._history.penultimate
        s = self._coords - coords_k
        y = self._coords.g - coords_k.g
        self._s.append(s)
        self._y.append(y)
        y_s = np.dot(y, s)
        y_y = np.dot(y, y)
        # TODO: handle negative y_s and near zero y_s, y_y
        self._rho.append(1.0 / y_s)
        # update the inverse Hessian diagonal
        gamma = y_s / y_y
        self._h_inv_d = np.ones_like(self._h_inv_d) * gamma
        return None

    def _step(self) -> None:
        """Take an L-BFGS step"""
        assert self._coords is not None
        assert self._coords.g is not None
        self._update_lbfgs_storage()

        # First step or after reset
        if len(self._s) == 0:
            step = -self._coords.g
            # take step within trust radius
            self._first_step = False
            return None

        def lbfgs_step_err(mu):
            """dx - trust radius"""
            dx = self._get_lbfgs_step(self._coords.g, self._s, self._y, mu)
            return dx - self._trust

        step = self._get_lbfgs_step(self._coords.g, self._s, self._y, 0.0)
        if np.linalg.norm(step) <= self._trust:
            self._coords = self._coords + step
            return None

        # μ has to be found in range (0, inf)
        left_bound = right_bound = None
        mu = 1.0
        for _ in range(10):
            if lbfgs_step_err(mu) < 0:
                right_bound = mu
            if lbfgs_step_err(mu) > 0:
                left_bound = mu
            if left_bound is not None:
                mu *= 2
            if right_bound is not None:
                mu *= 0.5

        res = root_scalar(
            lbfgs_step_err, bracket=[left_bound, right_bound], maxiter=1
        )
        if not res.converged:
            pass

    @staticmethod
    def _get_lbfgs_step(
        grad: np.ndarray,
        s_list: Deque[np.ndarray],
        y_list: Deque[np.ndarray],
        mu: float,
    ):
        """
        Obtain the regularised L-BFGS step, which corresponds to
        (H + μI)^-1 . g, where μ is a regularisation parameter

        Args:
            grad: The gradient vector
            s_list: List of coordinate changes x_i+1 - x_i
            y_list: List of gradient changes g_i+1 - g_i
            mu: The regularisation parameter μ

        Returns:
            (np.ndarray): The L-BFGS step
        """
        # L-BFGS two loop iteration
        n_vecs = len(s_list)
        q = grad.copy()
        # TODO: double check the formulas
        iter_range = range(0, n_vecs)
        # TODO: replace these with python list after debug
        alpha = np.zeros(shape=n_vecs)
        rho_arr = np.zeros(shape=n_vecs)
        mu_arr = np.array([mu] * n_vecs)

        gamma = 1.0
        for i in reversed(iter_range):
            s_i, y_i = s_list[i], y_list[i]
            y_hat = y_i + mu_arr[i] * s_i
            y_s = y_hat.dot(s_i)
            # ensure that y.dot(s) > 0
            if y_s <= 0:
                mu_arr[i] = max(0, -y_i.dot(s_i) / s_i.dot(s_i)) + mu

            y_hat = y_i + mu_arr[i] * s_i
            if i == max(iter_range):
                gamma = y_hat.dot(s_i) / y_hat.dot(y_hat)

            rho_arr[i] = 1 / y_hat.dot(s_i)
            alpha[i] = rho_arr[i] * np.dot(s_list[i], q)
            q -= alpha[i] * y_hat

        # regularised inverse diagonal
        h_inv_d = np.ones(shape=n_vecs) * gamma
        h_inv_d = 1 / ((1 / h_inv_d) + mu)
        q *= h_inv_d

        for i in iter_range:
            y_hat = y_list[i] + mu_arr[i] * s_list[i]
            beta = rho_arr[i] * np.dot(y_hat, q)
            q += (alpha[i] - beta) * s_list[i]

        return -q
