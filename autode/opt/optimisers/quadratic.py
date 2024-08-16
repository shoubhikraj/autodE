"""
Base classes of second-order (quadratic) optimisers
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Type, TYPE_CHECKING
from enum import Enum
import numpy as np

from autode.values import Distance
from autode.opt.optimisers.base import NDOptimiser
from autode.opt.coordinates.base import OptCoordinates
from autode.opt.coordinates import CartesianCoordinates
from autode.opt.coordinates.internals import AnyPIC
from autode.opt.coordinates.dic import DICWithConstraints
from autode.log import logger
from autode.config import Config
from autode.utils import work_in_tmp_dir
from autode.exceptions import CoordinateTransformFailed

if TYPE_CHECKING:
    from autode.species.species import Species
    from autode.wrappers.methods import Method
    from autode.hessians import Hessian


MAX_TRUST = 0.2
MIN_TRUST = 0.01


class QuadraticOptimiserBase(NDOptimiser, ABC):
    """
    Base class for Hessian based optimisers in internal coordinates
    that use trust radius
    """

    def __init__(
        self,
        maxiter: int,
        conv_tol,
        calc_hess: bool,
        recalc_hess_every: int,
        init_trust: float,
        trust_update: bool,
        max_move,
        extra_prims,
        **kwargs,
    ):
        """ """
        super().__init__(maxiter=maxiter, conv_tol=conv_tol, **kwargs)

        self._calc_hess = calc_hess
        self._recalc_hess_every = recalc_hess_every
        self._trust = float(init_trust)
        if not MIN_TRUST < self._trust < MAX_TRUST:
            self._trust = min(max(init_trust, MIN_TRUST), MAX_TRUST)
            logger.warning(f"Setting trust radius to {self._trust:.3f}")

        assert self._trust > 0, "Trust radius has to be positive!"
        self._trust_update = bool(trust_update)
        self._maxmove = Distance(max_move, units="ang")
        assert self._maxmove > 0, "Max movement has to be positive!"
        self._extra_prims = extra_prims
        self._last_pred_de = 0.0
        # TODO: remove hessian_update_types from NDOPtimiser

    def _step(self) -> None:
        """
        Take a quadratic step, ensuring the trust radius is updated and
        the coordinate system rebuilt if needed
        """
        self._update_trust_radius()
        step = self._get_quadratic_step()
        try:
            self._take_step_within_max_move(step)
        except CoordinateTransformFailed as exc:
            logger.warning(
                f"Coordinate failure: {str(exc)}, rebuilding coordinate"
                f" system and trying again..."
            )
            self._build_coordinates()
            try:
                step = self._get_quadratic_step()
                self._take_step_within_max_move(step)
            except CoordinateTransformFailed:
                raise RuntimeError(
                    "Repeated failure in coordinate system, unable to recover"
                )

        last_coords = self._history.penultimate
        self._last_pred_de = last_coords.pred_quad_delta_e(self._coords)
        return None

    def _take_step_within_max_move(self, delta_s):
        """
        Take the optimiser step ensuring that the maximum movement
        of any atom in Cartesian space does not exceed the max_move
        threshold

        Args:
            delta_s: Step in current coordinate system
        """
        assert self._coords is not None

        self._coords.allow_unconverged_back_transform = True
        new_coords = self._coords + delta_s
        cart_delta = new_coords.to("cart") - self._coords.to("cart")
        cart_displ = np.linalg.norm(cart_delta.reshape((-1, 3)), axis=1)
        max_displ = np.abs(cart_displ).max()
        self._coords.allow_unconverged_back_transform = False

        if max_displ > self._maxmove:
            logger.info(
                f"Calculated step too large: max. displacement = "
                f"{max_displ:.3f} Å, scaling down"
            )
            # Note because the transformation is not linear this will not
            # generate a step exactly max(∆x) ≡ α, but is empirically close
            factor = self._maxmove / max_displ
            self._coords = self._coords + (factor * delta_s)
        else:
            self._coords = self._coords + delta_s

        return None

    @abstractmethod
    def _get_quadratic_step(self) -> np.ndarray:
        """Obtain the quadratic step"""

    @abstractmethod
    def _update_trust_radius(self):
        """Update the trust radius"""

    def _build_coordinates(self, rebuild=False):
        """(Re-)build the coordinates for this optimiser from the species"""
        if self._species is None:
            raise RuntimeError("Cannot build coordinates, species is not set!")
        cart_coords = CartesianCoordinates(self._species.coordinates)
        primitives = AnyPIC.from_species(self._species)
        dic = DICWithConstraints.from_cartesian(
            x=cart_coords, primitives=primitives
        )

        if rebuild:
            old_g = self._coords.to("cart").g
            # TODO: how to get old Hessian

    @property
    @work_in_tmp_dir(use_ll_tmp=True)
    def _low_level_cart_hessian(self) -> "Hessian":
        """
        Calculate a Hessian matrix using a low-level method, used as the
        estimate from which Hessian updates are applied.
        """
        from autode.methods import get_lmethod

        assert self._species is not None, "Must have a species"

        logger.info("Calculating low-level Hessian")

        species = self._species.copy()
        species.calc_hessian(method=get_lmethod(), n_cores=self._n_cores)
        assert species.hessian is not None, "Hessian calculation must be ok"

        return species.hessian

    def _initialise_run(self) -> None:
        """Initialise the optimisation"""
        logger.info("Initialising optimisation")
        self._build_coordinates()
        assert self._coords is not None
        if self._calc_hess:
            self._update_hessian_gradient_and_energy()
        else:
            self._coords.update_h_from_cart_h(self._low_level_cart_hessian)
            self._update_gradient_and_energy()
        return None
