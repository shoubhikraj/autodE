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

    def _first_step(self) -> MWCartesianCoordinates:
        # todo put this in base
        # todo implement energy represented displacement
        eigvals, eigvecs = np.linalg.eigh(self._coords.h)
        ts_eigvec = eigvecs[:, 0]

        scaled_ts_vec = ts_eigvec / np.linalg.norm(ts_eigvec)
        largest_comp = np.argmax(np.abs(scaled_ts_vec))
        if scaled_ts_vec[largest_comp] > 0:
            pass
        else:
            scaled_ts_vec = -scaled_ts_vec
        step = self._step_size * scaled_ts_vec
        if self._direction == "forward":
            pass
        else:
            step = -step

        return self._coords + step
