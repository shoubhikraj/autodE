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
        self._h_diag: Optional[np.ndarray] = h0  # 1D diagonal of Hessian
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
        if self._h_diag is not None:
            assert isinstance(self._h_diag, np.ndarray)
            if self._h_diag.shape != self._coords.shape:
                raise ValueError(
                    "The provided diagonal Hessian guess does not "
                    "have the correct shape: must be a flat 1D array of"
                    " the same length as the coordinates"
                )
            if (self._h_diag < 0.0).any():
                logger.error(
                    "The provided diagonal Hessian guess contains negative"
                    " entries (i.e., not positive definite), setting to an"
                    " unit matrix"
                )
                self._h_diag = np.ones(shape=self._coords.shape[0])

        else:
            logger.debug(
                "No diagonal Hessian guess provided, assuming unit matrix"
            )
            # unit diagonal matrix
            self._h_diag = np.ones(shape=self._coords.shape[0])
        return None

    def _update_lbfgs_storage(self):
        """
        Store the coordinate and gradient change in last step in L-BFGS
        and update the Hessian diagonal
        """
        if self.iteration == 0:
            return None

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
        # TODO: how to handle negative gamma values?
        gamma = y_s / y_y
        self._h_diag = np.ones_like(self._h_diag) * gamma
        return None

    def _get_coords_from_line_search(self):
        """
        Check if the last step satisfies the Wolfe conditions, if not
        then perform a cubic interpolation/extrapolation. Returns the
        coordinates for the next step (which is the current set of
        coordinates if no line search, or the coordinates from cubic
        fit with estimates gradients)

        Returns:
            (CartesianCoordinates): The set of coordinates from which
                                    to take the next step
        """
        if self.iteration == 0:
            return self._coords
        # Check if the last taken step satisfies the Wolfe conditions.
        # Values taken from Nocedal and Wright for quasi-Newton methods
        c_1 = 1.0e-4
        c_2 = 0.9
        last_coords = self._history.penultimate
        last_step = self._coords.raw - last_coords.raw
        # first condition: f(x + a * p) <= f(x) + c_1 * a * dot(p, f'(x))
        # OR, f(x + step) <= f(x) + c_1 dot(step, f'(x))
        first_wolfe = self._coords.e <= (
            last_coords.e + c_1 * np.dot(last_step, last_coords.g)
        )
        # second condition: dot(-p, f'(x+ a * p)) <= -c_2 * dot(p, f'(x))
        # OR, a * dot(-p, f'(x + a*p)) <= -c_2 * a * dot(p, f'(x))
        # OR, dot(-step, f'(x + a*p)) <= -c_2 * dot(step, f'(x))
        second_wolfe = np.dot(-last_step, self._coords.g) <= (
            -c_2 * np.dot(last_step, last_coords.g)
        )
        if first_wolfe and second_wolfe:
            logger.info("Wolfe conditions fulfilled, skipping line search")
            return self._coords

        logger.warning(
            "Wolfe conditions not satisfied, fitting cubic polynomial"
            " to obtain the minimum point along last search direction"
        )
        # if not satisfied, perform cubic interpolation/extrapolation with
        # the directional gradients to obtain the minimiser
        g0 = float(np.dot(last_coords.g, last_step))
        g1 = float(np.dot(self._coords.g, last_step))
        e0 = float(last_coords.e)
        e1 = float(self._coords.e)
        cubic_poly = two_point_cubic_fit(e0, g0, e1, g1)
        minim = get_poly_minimum(cubic_poly)
        if minim is None:
            # polynomial has no minimum, skip linear fit
            logger.warning(
                "Fitted polynomial has no minimum, skipping line search"
            )
            return self._coords

        interp_coords = last_coords + last_step * minim
        interp_coords.e = Energy(cubic_poly(minim))
        interp_coords.g = (1 - minim) * last_coords.g + minim * self._coords.g

        if -1 < minim < 2:
            # todo is the range reasonable? Gaussian16 says it is
            logger.info(
                "Successful cubic fit: taking next step from "
                "interpolated coordinates"
            )
            return interp_coords
        else:
            logger.warning("Cubic fit step too large, skipping line search")
            return self._coords

    def _step(self) -> None:
        """Take an L-BFGS step, using pure python routines"""

        # NOTE: reset_lbfgs function must be called before update_trust
        # as it can remove the last step from memory so that the energy
        # rise is not registered by reset_lbfgs
        self._update_lbfgs_storage()

        coords = self._get_coords_from_line_search()
        # First step or after LBFGS reset
        if self._first_step:
            step = -(self._h_diag * coords.g)
            max_component = np.max(np.abs(step))
            # Make the first step size more cautious, quarter of trust radius
            # or 0.01 angstrom whichever is bigger
            if max_component > max(self.alpha / 4, 0.01):
                step = step * max(self.alpha / 4, 0.01) / max_component
            self._first_step = False

        # NOTE: self._s, self._y and self._rho are python deques with a maximum
        # length, which means that appending more items at the end should remove
        # items from the beginning of the deque
        else:
            step = _get_lbfgs_step_py(
                coords.g,
                self._s,
                self._y,
                self._rho,
                self._h_diag,
            )

        logger.info(
            f"Taking an L-BFGS step: current trust radius"
            f" = {self.alpha:.3f}"
        )

        # Take step within trust radius
        max_component = np.max(np.abs(step))
        if max_component > self.alpha:
            logger.info(
                f"Calculated step is too large ({max_component:.3f} Å)"
                f" - scaling down"
            )
            step = step * self.alpha / max_component
        self._coords = coords + step
        return None

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


class LBFGSFunctionOptimiser(LBFGSOptimiser):
    """
    A class that can optimise external functions, and provides an
    interface similar to scipy
    """

    def __init__(self, fn, x0, jac, func_args, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._fn = fn
        self._x0 = x0
        self._jac = jac
        if func_args is None:
            func_args = []
        self._fn_args = func_args

    def _update_gradient_and_energy(self) -> None:
        self._coords.e = Energy(self._fn(self._coords.raw, *self._fn_args))
        self._coords.g = self._jac(self._coords.raw, *self._fn_args).flatten()

    def _initialise_run(self) -> None:
        self._coords = CartesianCoordinates(self._x0)
        self._update_gradient_and_energy()
        if self._h_diag is None:
            self._h_diag = np.ones(shape=self._x0.shape[0])

    # todo check the gtol and etol values
    @classmethod
    def minimise_function(
        cls, fn, x0, jac, func_args=None, gtol=1e-3, etol=1e-4, *args, **kwargs
    ):
        from autode import Molecule
        from autode.methods import XTB

        opt = cls(
            fn, x0, jac, func_args, *args, gtol=gtol, etol=etol, **kwargs
        )
        # generate a dummy species
        tmp_spc = Molecule(smiles="N#N")
        opt.run(tmp_spc, XTB())
