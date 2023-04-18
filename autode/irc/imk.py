"""
The earliest (probably) algorithm for calculation of IRC,
due to Ishida, Morokuma and Komornicki.

Later modified by Gordon and co-workers.

[1] K. Ishida, K. Morokuma, A. Komornicki, J. Chem. Phys., 1977, 66, 2153-2156
[2] M. W. Schmidt, M. S. Gordon, M. Dupuis, J. Am. Chem. Soc. 1985, 107, 2585-2589
"""
import numpy as np
from autode.values import MWDistance, Angle
from autode.irc.base import MWIntegrator
from autode.opt.coordinates import MWCartesianCoordinates


class IMKIntegrator(MWIntegrator):
    def __init__(
        self,
        max_points: int,
        step_size: MWDistance = MWDistance(0.08, "ang amu^1/2"),
        elbow_thresh: Angle = Angle(179, "degrees"),
        *args,
        **kwargs,
    ):
        super().__init__(max_points, step_size, *args, **kwargs)

        self._elbow = Angle(elbow_thresh, units="degrees").to("radians")

    def _predictor_step(self) -> MWCartesianCoordinates:

        # IMK predictor step is a simple gradient step (normalized)
        g_hat = self._coords.g / np.linalg.norm(self._coords.g)
        new_coords = self._coords + self._step_size * g_hat

        return new_coords

    def _corrector_step(self, coords: MWCartesianCoordinates):
        pass
