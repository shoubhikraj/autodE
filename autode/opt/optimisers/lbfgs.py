"""
Gradient-only limited-memory BFGS optimiser algorithm,
suitable for large molecular systems where calculating
or storing Hessian would be difficult.

References:
[1] J. Nocedal, Mathematics of Computation, 35, 1980, 773-782
[2] D. C. Liu, J. Nocedal, Mathematical Programming, 45, 1989, 503-528
[3] J. Nocedal, S. Wright, "Numerical Optimization", 2nd ed., Springer, 2006
"""
import numpy as np
from typing import Optional
from collections import deque
from autode.opt.coordinates import CartesianCoordinates
from autode.opt.optimisers import RFOptimiser
from autode.opt.optimisers.polynomial_fit import (
    two_point_cubic_fit,
    get_poly_minimum,
)
from autode.values import Energy
from autode.log import logger

# Maximum consecutive energy rise steps allowed
_max_allow_n_e_rise_steps = 5


class LBFGSOptimiser(RFOptimiser):
    """
    L-BFGS optimisation in Cartesian coordinates
    """

    def __init__(
        self,
        *args,
        max_step: float = 0.2,
        max_vecs: int = 15,
        h0: np.ndarray = None,
        max_e_rise: Energy = Energy(0.004, "Ha"),
        **kwargs,
    ):
        """
        Initialise an L-BFGS optimiser. The most important parameter is
        the total number of vectors to store for estimation of the curvature.
        Choosing a number in the range (3, 20) is recommended by Nocedal and
        Wright.

        Args:
            *args:
            max_step:
            max_vecs:
            h0:
            max_e_rise:
            **kwargs:
        """
        super().__init__(*args, **kwargs)

        self.alpha = max_step
        self._max_vecs = abs(int(max_vecs))
        self._s = deque(maxlen=max_vecs)  # stores changes in coords - arrays
        self._y = deque(maxlen=max_vecs)  # stores changes in grad - arrays
        self._rho = deque(maxlen=max_vecs)  # stores 1/(y.T @ s) - floats
        self._h0: Optional[np.ndarray] = h0  # initial 1D diagonal of Hessian
        self._max_e_rise = Energy(
            abs(max_e_rise), "Ha"
        )  # 0.004 Ha ~ 2 kcal/mol
        self._n_e_rise = 0
        self._n_e_decrease = 0
        self._n_resets = 0
        self._first_step = True

    @property
    def converged(self) -> bool:
        return self._g_norm < self.gtol and self._abs_delta_e < self.etol

    @property
    def _exceeded_maximum_iteration(self) -> bool:
        # todo stop when too many reset or subsequent energy rise
        return super()._exceeded_maximum_iteration or self._n_resets > 2

    def _initialise_run(self) -> None:
        # NOTE: While LBFGS could be used for internal coordinates, it is
        # practically pointless because to store the matrices required to
        # convert between internal <-> cartesian would require almost the
        # same amount of memory and effort as storing the Hessian
        self._coords = CartesianCoordinates(self._species.coordinates)
        self._update_gradient_and_energy()
        if self._h0 is not None:
            assert isinstance(self._h0, np.ndarray)
            if self._h0.shape != self._coords.shape:
                raise ValueError(
                    "The provided diagonal Hessian guess does not "
                    "have the correct shape: must be a flat 1D array of"
                    " the same length as the coordinates"
                )
            if (self._h0 < 0.0).any():
                logger.error(
                    "The provided diagonal Hessian guess contains negative"
                    " entries (i.e., not positive definite), setting to an"
                    " unit matrix"
                )
                self._h0 = np.ones(shape=self._coords.shape[0])

        else:
            logger.debug(
                "No diagonal Hessian guess provided, assuming unit matrix"
            )
            # unit diagonal matrix
            self._h0 = np.ones(shape=self._coords.shape[0])
        return None

    def _step(self) -> None:
        """Take an L-BFGS step, using pure python routines"""

        # NOTE: reset_lbfgs function must be called before update_trust
        # as it can remove the last step from memory so that the energy
        # rise is not registered by reset_lbfgs
        self._reset_lbfgs_if_required()
        self._update_trust_radius()

        # First step or after LBFGS reset
        if self._first_step:
            step = -(self._h0 * self._coords.g)
            step_size = np.linalg.norm(step)
            # Make the first step size cautious, quarter of trust radius
            # or 0.01 angstrom whichever is bigger
            if step_size > max(self.alpha / 4, 0.01):
                step = step * max(self.alpha / 4, 0.01) / step_size
            self._first_step = False

        # NOTE: self._s, self._y and self._rho are python deques with a maximum
        # length, which means that appending more items at the end should remove
        # items from the beginning of the deque
        else:
            s = self._coords - self._history.penultimate
            y = self._coords.g - self._history.penultimate.g
            y_s = np.dot(y, s)
            # look out for near zero values of y_s
            if abs(y_s) < 1.0e-15:
                logger.warning("Resetting y_s in LBFGS to 1.0")
                y_s = 1.0
            # save in memory
            self._s.append(s)
            self._y.append(y)
            self._rho.append(1.0 / y_s)
            # update the hessian diagonal
            y_y = np.dot(y, y)
            if abs(y_y) < 1.0e-15:
                logger.warning("Resetting y_y in LBFGS to 1.0")
            gamma = y_s / y_y
            h_diag = self._h0 * gamma
            step = _get_lbfgs_step_py(
                self._coords.g,
                self._s,
                self._y,
                self._rho,
                h_diag,
            )

        logger.info(
            f"Taking an L-BFGS step: current trust radius"
            f" = {self.alpha:.3f}"
        )
        self._take_step_within_trust_radius(step)

    def _update_trust_radius(self):
        """
        Update the trust radius and also reject steps
        where energy rises beyond a pre-chosen threshold.
        """
        # first iteration, so no trust update possible
        if self.iteration == 0:
            return None

        # NOTE: Here a very simple trust radius update method is used,
        # if the energy rises beyond the threshold, reject last step and
        # set the trust radius to 1/4 of the value
        if self.last_energy_change > self._max_e_rise:
            logger.warning(
                f"Energy increased by {self.last_energy_change},"
                f" rejecting last step and reducing trust radius"
            )
            self._history.pop()
            self._n_e_decrease = 0
            self.alpha /= 4
            return None

        # If energy increases, but within the threshold, reduce trust radius
        # more cautiously
        if self.last_energy_change > 0:
            logger.warning("Energy rising - reducing trust radius")
            self._n_e_decrease = 0
            self.alpha /= 1.5

        # If energy going down for last 4 steps AND the last step was
        # nearly at trust radius then increase trust radius, cautiously
        if self.last_energy_change < 0:
            last_step_size = np.linalg.norm(self._coords - self._history[-2])
            self._n_e_decrease += 1
            if self._n_e_decrease >= 4 and np.isclose(
                last_step_size, self.alpha, rtol=0.01
            ):
                logger.debug(
                    "Energy falling smoothly - increasing trust radius"
                )
                self.alpha *= 1.2
                # reset after increasing trust radius
                self._n_e_decrease = 0

        return None

    def _reset_lbfgs_if_required(self):
        """
        If the energy is rising consecutively for a chosen number of
        iterations, the L-BFGS memory is reset i.e., the stored
        gradient and coordinate changes are lost so that a steepest
        descent step is taken again
        """
        if self.iteration == 0:
            return None

        if self.last_energy_change > 0:
            self._n_e_rise += 1
        else:
            self._n_e_rise = 0

        if self._n_e_rise >= _max_allow_n_e_rise_steps:
            logger.warning(
                f"Energy rising for consecutive {_max_allow_n_e_rise_steps} "
                f"steps, resetting LBFGS"
            )
            self._n_resets += 1
            self._y.clear()
            self._s.clear()
            self._rho.clear()
            self._first_step = True
        return None

    def _adjust_last_step(self):
        # Check if the last taken step satisfies the Wolfe conditions.
        # Values taken from Nocedal and Wright for quasi-Newton methods
        c_1 = 1.0e-4
        c_2 = 0.9
        last_step = self._coords.raw - self._history.penultimate.raw
        # first condition: f(x + a * p) <= f(x) + c_1 * a * dot(p, f'(x))
        # OR, f(x + step) <= f(x) + c_1 dot(step, f'(x))
        first_wolfe = self._coords.e <= (
            self._history[-2].e + c_1 * np.dot(last_step, self._history[-2].g)
        )
        # second condition: dot(-p, f'(x+ a * p)) <= -c_2 * dot(p, f'(x))
        # OR, a * dot(-p, f'(x + a*p)) <= -c_2 * a * dot(p, f'(x))
        # OR, dot(-step, f'(x + a*p)) <= -c_2 * dot(step, f'(x))
        second_wolfe = np.dot(-last_step, self._coords.g) <= (
            -c_2 * np.dot(last_step, self._history[-2].g)
        )
        if first_wolfe and second_wolfe:
            return None

        # if not satisfied, perform cubic interpolation/extrapolation with
        # the two gradients to obtain the minimiser
        pass


def _get_lbfgs_step_py(
    grad: np.ndarray,
    s_matrix: deque,
    y_matrix: deque,
    rho_array: deque,
    hess_diag: np.ndarray,
):
    # todo is python implementation fast enough or do we need cython
    n_vecs = len(s_matrix)
    q = grad.copy()
    # todo double check the formulas
    iter_range = range(0, n_vecs)
    a = np.zeros(shape=(len(iter_range)))
    for i in reversed(iter_range):
        a[i] = rho_array[i] * np.dot(s_matrix[i], q)
        q -= a[i] * y_matrix[i]

    q *= hess_diag  # use q as working space for z
    for i in iter_range:
        beta = rho_array[i] * np.dot(y_matrix[i], q)
        q += (a[i] - beta) * s_matrix[i]

    return -q
