"""
Fast Inertial Relaxation Engine (FIRE) Optimiser. It uses
only gradients and energies (i.e. first order) to minimise.
"""
import numpy as np

from autode.opt.coordinates import CartesianCoordinates
from autode.opt.optimisers import NDOptimiser
from autode.values import Time, Velocity, Distance


class FIREOptimiser(NDOptimiser):
    def __init__(
        self,
        *args,
        dt_init: Time = Time(0.25, "fs"),
        dt_min: Time = Time(0.01, "fs"),
        dt_max: Time = Time(1.0, "fs"),
        alpha_start: float = 0.1,
        f_alpha: float = 0.99,
        f_dec: float = 0.5,
        f_inc: float = 1.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._dt = Time(dt_init, "fs")
        self._dt_min = Time(dt_min, "fs")  # todo check
        self._dt_max = Time(dt_max, "fs")
        self._alpha_start = float(alpha_start)
        self._alpha = self._alpha_start
        self._f_alpha = float(f_alpha)
        assert 0 < self._f_alpha < 1
        self._f_dec = float(f_dec)
        assert 0 < self._f_dec < 1
        self._f_inc = float(f_inc)
        assert self._f_inc > 1

        self._v = None
        self._N_plus = 0
        self._N_minus = 0
        self._N_delay = None  # todo complete
        self._N_min_thresh = None  # todo complete
        self._initial_delay = None  # todo

    def _initialise_run(self) -> None:
        self._coords = CartesianCoordinates(self._species.coordinates)
        self._update_gradient_and_energy()
        self._v = np.zeros_like(self._coords)

    def _step(self) -> None:
        force = -self._coords.g
        p = np.dot(force, self._v)
        if p > 0:
            self._N_plus += 1
            self._N_minus = 0
            if self._N_plus > self._N_delay:
                self._dt = min(self._dt * self._f_inc, self._dt_max)
                self._alpha = self._alpha * self._f_alpha

        else:  # p <= 0
            self._N_plus = 0
            self._N_minus += 1
            if self._N_minus > self._N_min_thresh:
                return None
            # todo put code also in converged
            if not (self._initial_delay and self.iteration < self._N_delay):
                self._dt = max(self._dt * self._f_dec, self._dt_min)
            self._alpha = self._alpha_start
            # Correction for uphill motion
            self._coords = (
                self._coords - 0.5 * self._dt * self._v
            )  # todo should we change self._coords??
            self._v[:] = 0.0
