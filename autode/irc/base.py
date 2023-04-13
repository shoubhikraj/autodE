"""
Base classes for IRC calculation
"""
from abc import abstractmethod
from typing import Optional
from autode.opt.coordinates import OptCoordinates
from autode.opt.optimisers.base import NDOptimiser, OptimiserHistory
from autode.opt.optimisers.hessian_update import BofillUpdate


class BaseIntegrator(NDOptimiser):
    def __init__(
        self,
        read_init_hess: bool = False,
        recalc_hess: Optional[int] = None,
        direction: str = "forward",
    ):
        super().__init__()
        self._should_calc_hess = not bool(read_init_hess)

        self._recalc_hess_freq = None
        if recalc_hess is not None:
            self._recalc_hess_freq = int(recalc_hess)

        direction = direction.lower()
        if direction in ["forward", "backward", "downhill"]:
            self._direction = direction

        self._species = None
        self._history = OptimiserHistory()

        self._hessian_update_type = BofillUpdate

    @abstractmethod
    def _initialise_run(self):
        pass

    @abstractmethod
    def _first_step(self):
        pass

    @abstractmethod
    def _predictor_step(self) -> OptCoordinates:
        pass

    @abstractmethod
    def _corrector_step(self, coords: OptCoordinates):
        pass

    def run(self, species, method):
        pass
