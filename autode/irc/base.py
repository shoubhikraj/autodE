"""
Base classes for IRC calculation
"""
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
from autode.opt.coordinates import OptCoordinates
from autode.opt.optimisers.base import OptimiserHistory
from autode.opt.optimisers.hessian_update import BofillUpdate
from autode.exceptions import CalculationException

if TYPE_CHECKING:
    from autode.species.species import Species
    from autode.wrappers.methods import Method


class BaseIntegrator(ABC):
    def __init__(
        self,
        max_points: int,
        read_init_hess: bool = False,
        recalc_hess: Optional[int] = None,
        direction: str = "forward",
    ):
        super().__init__()
        self._should_init_hess = not bool(read_init_hess)

        self._recalc_hess_freq = None
        if recalc_hess is not None:
            self._recalc_hess_freq = int(recalc_hess)

        direction = direction.lower()
        if direction in ["forward", "backward", "downhill"]:
            self._direction = direction
        else:
            raise ValueError(
                "The direction for IRC integration must either"
                "be 'forward', 'backward' or 'downhill' "
            )
        self._maxpoints = int(max_points)
        self._species: Optional["Species"] = None
        self._method: Optional["Method"] = None
        self._n_cores: Optional[int] = None
        self._history = OptimiserHistory()

        self._hessian_updater = BofillUpdate

    def completed(self) -> bool:
        """
        IRC is completed if (1) the maximum number of points requested
        have been integrated, or (2) the gradient has decreased below
        the requested tolerance
        """
        pass

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

    @property
    def _coords(self):
        if len(self._history) == 0:
            return None
        return self._history[-1]

    @_coords.setter
    def _coords(self, value):
        if isinstance(value, OptCoordinates):
            self._history.append(value.copy())
        else:
            raise ValueError

    @property
    def iteration(self):
        return len(self._history) - 1

    def run(self, species, method):
        self._initialise_species_and_method(species, method)
        self._initialise_run()

        pass

    def _initialise_species_and_method(self, species, method):

        # check species and method

        hess_calc_reqd = (
            self._should_init_hess or self._recalc_hess_freq is not None
        )

        if (
            hess_calc_reqd
            and method.keywords.hess.bstring != method.keywords.grad.bstring
        ):
            raise CalculationException(
                "At least one Hessian calculation is required for this IRC"
                "run, however, hessian keywords are different from gradient"
                "keywords"
            )

    def _update_gradient_and_energy(self, coords: OptCoordinates):

        self._species.coordinates = coords.to("cart")
        from autode.calculations import Calculation

        engrad_calc = Calculation(
            name=f"{self._species.name}_irc_{self.iteration}_engrad",
            molecule=self._species,
            method=self._method,
            keywords=self._method.keywords.grad,
            n_cores=self._n_cores,
        )

        engrad_calc.run()
        engrad_calc.clean_up(force=True, everything=True)

        coords.update_g_from_cart_g(self._species.gradient)
        coords.e = self._species.energy
        # todo raise calculation exceptions

        return None

    def _update_hessian_gradient_and_energy(self, coords: OptCoordinates):

        self._species.coordinates = coords.to("cart")
        from autode.calculations import Calculation

        hess_calc = Calculation(
            name=f"{self._species.name}_irc_{self.iteration}_hess",
            molecule=self._species,
            method=self._method,
            keywords=self._method.keywords.hess,
            n_cores=self._n_cores,
        )

        hess_calc.run()
        hess_calc.clean_up(force=True, everything=True)

        coords.update_h_from_cart_h(self._species.hessian)
        coords.update_g_from_cart_g(self._species.gradient)
        coords.e = self._species.energy

        return None
