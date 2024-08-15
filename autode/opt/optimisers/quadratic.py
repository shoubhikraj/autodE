"""
Base classes of second-order (quadratic) optimisers
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Type
from enum import Enum
import numpy as np

from autode.values import Distance
from autode.opt.optimisers.base import NDOptimiser
from autode.opt.coordinates.base import OptCoordinates
from autode.log import logger


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
        init_hess: str,
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
        # TODO: remove hessian_update_types from NDOPtimiser

    @abstractmethod
    def _build_coordinates(self):
        """Build the coordinates for this optimiser from the species"""

    def _initialise_run(self) -> None:
        """Initialise optimisation"""
        logger.info("Initialising optimisation")
        self._build_coordinates()
        self._update_gradient_and_energy()
        if self._init_hess == _InitHessStrategy.READ:
            self._coords.update_h_from_cart_h(self._init_hess.cart_arr)
