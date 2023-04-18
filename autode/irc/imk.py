"""
The earliest (probably) algorithm for calculation of IRC,
due to Ishida, Morokuma and Komornicki.

Later modified by Gordon and co-workers.

[1] K. Ishida, K. Morokuma, A. Komornicki, J. Chem. Phys., 1977, 66, 2153-2156
[2] M. W. Schmidt, M. S. Gordon, M. Dupuis, J. Am. Chem. Soc. 1985, 107, 2585-2589
"""
from typing import TYPE_CHECKING
import math
import numpy as np
from autode.values import MWDistance, Angle
from autode.irc.base import MWIntegrator
from autode.opt.coordinates import MWCartesianCoordinates
from autode.log import logger

if TYPE_CHECKING:
    from autode.species.species import Species
    from autode.wrappers.methods import Method


class IMKIntegrator(MWIntegrator):
    def __init__(
        self,
        max_points: int,
        step_size: MWDistance = MWDistance(0.08, "ang amu^1/2"),
        elbow_thresh: Angle = Angle(179, "degrees"),
        corr_delta: MWDistance = MWDistance(0.013, "ang amu^1/2"),
        *args,
        **kwargs,
    ):
        super().__init__(max_points, step_size, *args, **kwargs)

        self._elbow = Angle(elbow_thresh, units="degrees")
        self._corr_delta = MWDistance(corr_delta)

    def _update_energy_for(self, coords: MWCartesianCoordinates):

        self._species.coordinates = coords.to("cart")
        from autode.calculations import Calculation

        sp_calc = Calculation(
            name=f"{self._species.name}_irc_{self.n_points}_sp",
            molecule=self._species,
            method=self._method,
            keywords=self._method.keywords.low_sp,  # use low_sp
            n_cores=self._n_cores,
        )

        sp_calc.run()
        sp_calc.clean_up(force=True, everything=True)

        coords.e = self._species.energy

    def _initialise_species_and_method(
        self, species: "Species", method: "Method"
    ) -> None:
        super()._initialise_species_and_method(species, method)

        # For IMK method, an sp calculation is also required, so check
        # that low_sp is the same as grad

        if (
            self._method.keywords.low_sp.bstring
            == self._method.keywords.grad.bstring
        ):
            raise RuntimeError(
                "For Ishida-Morokuma-Komornicki IRC algorithm, low single"
                " point (low_sp) calculations must have the same keywords as"
                " the gradient calculations, but that is not satisfied"
            )

    def _predictor_step(self) -> MWCartesianCoordinates:

        # IMK predictor step is a simple gradient step (normalized)
        g_hat = self._coords.g / np.linalg.norm(self._coords.g)
        new_coords = self._coords + self._step_size * g_hat

        return new_coords

    def _corrector_step(self, coords: MWCartesianCoordinates):

        g_0_hat = self._coords.g / np.linalg.norm(self._coords.g)
        g_1_hat = coords.g / np.linalg.norm(coords.g)

        # calculate the "elbow" angle between g0 and g1
        cos_elbow = float(np.dot(g_0_hat, g_1_hat))
        if math.acos(cos_elbow) > self._elbow.to("radians"):
            logger.info(
                "Angle between subsequent gradient vectors is larger than"
                f" threshold {self._elbow}°, skipping correction step"
            )
            self._coords = coords
            return None

        # obtain the bisector
        d = g_0_hat - g_1_hat
        d_hat = d / np.linalg.norm(d)

        # take step along bisector
        coords_2 = coords + d_hat * self._corr_delta
        self._update_energy_for(coords_2)
