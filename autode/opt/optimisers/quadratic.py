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
from autode.log import logger
from autode.config import Config
from autode.exceptions import CoordinateTransformFailed

if TYPE_CHECKING:
    from autode.species.species import Species
    from autode.wrappers.methods import Method


MAX_TRUST = 0.2
MIN_TRUST = 0.01


class _InitHessStrategy(Enum):
    CALC = 1
    READ = 2
    UPDATE = 3
    LL_GUESS = 4


class InitialHessian:
    """This class defines various ways of obtaining the initial Hessian"""

    def __init__(self, strategy: _InitHessStrategy):
        """Different ways of obtaining the initial Hessian"""
        self.strategy = strategy
        self.to_calc = None
        self.cart_arr = None
        self.old_coords = None
        self.ll_guess = None

    @classmethod
    def from_cart_h(cls, arr):
        """
        Use a Cartesian hessian to initialise the optimisation

        Args:
            arr: The Hessian array

        Returns:
            (InitialHessian):
        """
        assert isinstance(arr, np.ndarray)
        inhess = cls(strategy=_InitHessStrategy.READ)
        inhess.cart_arr = arr
        return inhess

    @classmethod
    def from_old_coords(cls, coords):
        """
        Obtain updated Hessian from an old set of coordinates

        Args:
            coords: Old set of coordinates

        Returns:
            (InitialHessian):
        """
        assert isinstance(coords, OptCoordinates)
        inhess = cls(strategy=_InitHessStrategy.UPDATE)
        inhess.old_coords = coords
        return inhess

    @classmethod
    def from_ll_guess(cls):
        """
        Obtain Hessian from low-level guess

        Returns:
            (InitalHessian):
        """
        inhess = cls(strategy=_InitHessStrategy.LL_GUESS)
        return inhess

    @classmethod
    def from_calc(cls):
        """
        Obtain Hessian from calculation

        Returns:
            (InitialHessian):
        """
        inhess = cls(strategy=_InitHessStrategy.CALC)
        return inhess


class QuadraticOptimiserBase(NDOptimiser, ABC):
    """Base class for Hessian based optimisers that use trust radius"""

    def __init__(
        self,
        maxiter: int,
        conv_tol,
        init_hess: InitialHessian,
        recalc_hess_every,
        init_trust: float,
        trust_update: bool,
        max_move,
        extra_prims,
        **kwargs,
    ):
        """ """
        super().__init__(maxiter=maxiter, conv_tol=conv_tol, **kwargs)

        self._init_hess = init_hess
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

    @abstractmethod
    def _take_step_within_max_move(self, delta_s):
        """"""

    @abstractmethod
    def _get_quadratic_step(self) -> np.ndarray:
        """Obtain the quadratic step"""

    def _update_trust_radius(self):
        """Update the trust radius"""

    @abstractmethod
    def _build_coordinates(self):
        """(Re-)build the coordinates for this optimiser from the species"""

    def _initialise_run(self) -> None:
        """Initialise optimisation"""
        logger.info("Initialising optimisation")
        self._build_coordinates()

        # handle initial Hessian
        if self._init_hess.strategy == _InitHessStrategy.CALC:
            self._update_hessian_gradient_and_energy()

        self._update_gradient_and_energy()

        if self._init_hess == _InitHessStrategy.READ:
            self._coords.update_h_from_cart_h(self._init_hess.cart_arr)
