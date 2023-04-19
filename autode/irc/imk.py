"""
The earliest (probably) algorithm for calculation of IRC,
due to Ishida, Morokuma and Komornicki.

Later modified by Gordon and co-workers.

[1] K. Ishida, K. Morokuma, A. Komornicki, J. Chem. Phys., 1977, 66, 2153-2156
[2] M. W. Schmidt, M. S. Gordon, M. Dupuis, J. Am. Chem. Soc. 1985, 107, 2585-2589
"""
from typing import TYPE_CHECKING, Optional
import math
import numpy as np
from autode.values import MWDistance, Angle
from autode.irc.base import MWIntegrator
from autode.opt.coordinates import MWCartesianCoordinates
from autode.opt.optimisers.polynomial_fit import (
    two_point_exact_parabolic_fit,
    get_poly_minimum,
    two_point_cubic_fit,
)
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

        # First correction: if the predictor step predicts a point with
        # higher energy, then do a cubic fit to get minimum along
        # predictor step direction (inspired by ORCA)
        # NOTE: ORCA does quadratic fit with energy, but since we have
        # both gradients, it is better to use cubic fit
        if coords.e > self._coords.e:
            logger.info(
                "Energy higher after predictor step, using cubic "
                "fit to obtain minimum"
            )
            coords = _cubic_fit_get_minimum_irc(self._coords, coords)
            if coords is None:
                raise RuntimeError(
                    "Correction of large predictor step failed"
                    " Try restarting with lower step size"
                )
            self._update_gradient_and_energy_for(coords)

        # Second correction: The gradient descent step will veer away
        # from the true MEP due to its finite size, so step along the
        # bisector between the two gradient vectors (which represents
        # how much the reaction path is curving) and do a parabolic fit
        # to get back to the MEP
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

        # take small step along bisector
        coords_2 = coords + d_hat * self._corr_delta
        self._update_energy_for(coords_2)

        # parabolic fit to obtain corrected coordinates
        corr_3 = _parabolic_fit_get_minimum_imkmod(coords, coords_2, d_hat)
        if corr_3 is None:
            raise RuntimeError(
                "IRC correction step failed, aborting run..."
                "Try restarting with smaller step size"
            )
        self._coords = coords + d_hat * self._corr_delta
        self._update_gradient_and_energy_for(self._coords)
        return None


def _cubic_fit_get_minimum_irc(
    coords0: MWCartesianCoordinates,
    coords1: MWCartesianCoordinates,
) -> Optional[MWCartesianCoordinates]:
    """
    Fit a cubic polynomial using both energies and gradients
    and then get the minimum point as coordinates

    Args:
        coords0: First point
        coords1: Second point

    Returns:
        (MWCartesianCoordinates): The minimum coordinates
    """
    assert coords0.e is not None and coords0.g is not None
    assert coords1.e is not None and coords1.g is not None
    # distance vector
    dist_vec = coords1.raw - coords0.raw
    d_hat = dist_vec / np.linalg.norm(dist_vec)
    # project gradients
    g0 = float(np.dot(coords0.g, d_hat))
    g1 = float(np.dot(coords1.g, d_hat))
    e0 = float(coords0.e.to("Ha"))
    e1 = float(coords1.e.to("Ha"))
    # fit cubic and get minimum
    cubic_poly = two_point_cubic_fit(e0, g0, e1, g1)
    x_min = get_poly_minimum(cubic_poly, u_bound=1, l_bound=0)
    if x_min is None:
        return None
    return coords0 + (x_min * d_hat)


def _parabolic_fit_get_minimum_imkmod(
    coords0: MWCartesianCoordinates,
    coords1: MWCartesianCoordinates,
    d_hat: np.ndarray,
    max_step: float = 3,
) -> float:
    """
    Fit a parabolic function with two coordinate points,
    by using the energy and gradient from first point
    and only energy from second point

    Args:
        coords0: Coordinates of first point
        coords1: Coordinates of second point
        d_hat: Unit vector in the direction coord0->coord1
        max_step: Upper bound of x for searching minima
                  (beyond which the fit is considered uncertain)

    Returns:
        (float): The minimum location x
    """
    assert coords0.e is not None and coords1.e is not None
    assert coords0.g is not None
    # project the first gradient
    g0 = float(np.dot(coords0.g, d_hat))
    e0 = float(coords0.e.to("Ha"))
    e1 = float(coords1.e.to("Ha"))
    # fit parabola and get minimum
    parabola = two_point_exact_parabolic_fit(e0, g0, e1)
    return get_poly_minimum(parabola, u_bound=max_step, l_bound=0)
