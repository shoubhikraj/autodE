"""
Fast Inertial Relaxation Engine (FIRE) Optimiser. It uses
only gradients and energies (i.e. first order) to minimise.
"""
import numpy as np

from autode.opt.coordinates import CartesianCoordinates
from autode.opt.optimisers import NDOptimiser
from autode.values import Time, Mass
from autode.log import logger


class FIREOptimiser(NDOptimiser):
    """Fast Inertial Relaxation Engine (FIRE) Optimiser"""

    def __init__(
        self,
        *args,
        dt_init: Time = Time(0.25, "fs"),
        dt_min: Time = Time(0.001, "fs"),
        dt_max: Time = Time(1.0, "fs"),
        alpha_start: float = 0.1,
        f_alpha: float = 0.99,
        f_dec: float = 0.5,
        f_inc: float = 1.1,
        n_delay: int = 2,
        max_up: int = 3,
        initial_delay: bool = True,
        atom_mass: Mass = Mass(4, "amu"),
        **kwargs,
    ):
        """
        The FIRE Optimiser is a gradient-only optimiser based on
        MD integrators used on Newton's equations of motion. It is
        best used for large systems (e.g., >100 atoms) where
        calculating or storing Hessian is difficult or impossible

        Args:
            *args: Arguments passed on to NDOptimiser
            dt_init: Initial time step (in fs)
            dt_min: Lowest allowed time step (in fs)
            dt_max: Highest allowed time step (in fs)
            alpha_start: Initial steering coefficient (mixing factor)
                         should be in (0, 1)
            f_alpha: Reduction factor for steering coefficient if
                     force(dot)velocity is positive, should be in (0, 1)
            f_dec: Reduction factor for time step, should be in (0, 1)
            f_inc: Increase factor for time step, should be in (1, inf)
            n_delay: Number of steps to delay before increasing time step
                     and decreasing alpha after an uphill step
            max_up: Maximum number of uphill steps (force.v > 0) before
                    the optimiser gives up
            initial_delay: Set True to also put an additional delay at the
                           beginning of optimisation for n_delay iterations
                           to not decrease time step
            atom_mass: Mass of each atom (in amu)
            **kwargs: Keyword arguments passed on to NDOptimiser
        """
        super().__init__(*args, **kwargs)

        self._dt = Time(dt_init, "fs")
        self._dt_min = Time(dt_min, "fs")  # todo check
        self._dt_max = Time(dt_max, "fs")
        self._alpha_start = float(alpha_start)
        self._alpha = self._alpha_start
        self._f_alpha = float(f_alpha)
        self._f_dec = float(f_dec)
        self._f_inc = float(f_inc)

        self._mass = Mass(atom_mass, "amu")
        assert 0 < self._f_alpha < 1
        assert 0 < self._f_dec < 1
        assert self._f_inc > 1
        assert self._mass > 0

        self._v = None
        self._N_plus = 0
        self._N_minus = 0
        self._N_delay = int(n_delay)
        self._max_N_min = int(max_up)
        self._initial_delay = bool(initial_delay)
        self._masses = None

    def _initialise_run(self) -> None:
        self._coords = CartesianCoordinates(self._species.coordinates)
        self._update_gradient_and_energy()
        self._v = np.zeros_like(self._coords)
        self._masses = np.ones_like(self._coords) * self._mass

    @property
    def converged(self) -> bool:
        """
        For FIRE, there is an additional criteria, which is the number of
        successive uphill (force.dot.v > 0) steps has to be less than a
        certain threshold (set by max_up argument). If that happens then
        optimisation is assumed to be finished (i.e. unable to proceed)

        Returns:
            (bool): True if converged, False if not
        """

        if self._species is not None and self._species.n_atoms == 1:
            return True  # Optimisation 0 DOF is always converged

        if self._N_minus > self._max_N_min:
            logger.warning(
                f"FIRE optimiser taking uphill steps for more than"
                f" {self._N_minus} iterations, further optimisation"
                f" is not possible. The molecule is likely stuck in a "
                f" narrow potential well; please check results carefully!"
            )
            return True

        return self._abs_delta_e < self.etol and self._g_norm < self.gtol

    def _step(self) -> None:
        """
        FIRE optimisation step, currently only implemented with
        Euler semi-implicit integrator
        """
        forces = -self._coords.g
        power = np.dot(forces, self._v)
        if power > 0:
            self._N_plus += 1
            self._N_minus = 0
            if self._N_plus > self._N_delay:
                self._dt = min(self._dt * self._f_inc, self._dt_max)
                self._alpha = self._alpha * self._f_alpha
            coords = self._coords

        else:  # p <= 0
            self._N_plus = 0
            self._N_minus += 1
            logger.info(
                "Force and velocity are no longer aligned, resetting alpha, "
                "and setting velocities to zero. Reducing timestep if allowed"
            )
            if not (self._initial_delay and self.iteration < self._N_delay):
                if self._dt * self._f_dec >= self._dt_min:
                    self._dt = self._dt * self._f_dec

            self._alpha = self._alpha_start
            # Correction for uphill motion
            coords = self._coords - 0.5 * self._dt * self._v
            self._v[:] = 0.0
        logger.info("Taking a FIRE Euler semi-implicit step")
        self._coords = self._euler_semi_implicit(forces, coords)

    def _euler_semi_implicit(
        self, forces: np.ndarray, coords: CartesianCoordinates
    ):
        """
        Euler semi-implicit integration step for FIRE

        Args:
            forces: Forces on each atom in Cartesian coordinates
            coords: Coordinates on which to take the steps

        Returns:
            (CartesianCoordinates): New coordinates after integration
        """
        v_2 = self._v + self._dt * (forces / self._masses)
        # FIRE 2.0 mixing step
        v_2 = (1 - self._alpha) * v_2 + (
            self._alpha * forces * np.linalg.norm(v_2) / np.linalg.norm(forces)
        )
        self._v = v_2
        return coords.raw + self._dt * v_2
