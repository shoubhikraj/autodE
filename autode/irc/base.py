"""
Base classes for IRC calculation
"""
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
import numpy as np
from autode.values import GradientRMS
from autode.opt.coordinates import OptCoordinates
from autode.opt.optimisers.base import OptimiserHistory
from autode.opt.optimisers.hessian_update import BofillUpdate
from autode.exceptions import CalculationException
from autode.log import logger

if TYPE_CHECKING:
    from autode.species.species import Species
    from autode.wrappers.methods import Method

_flush_old_hessians = True


class BaseIntegrator(ABC):
    def __init__(
        self,
        max_points: int,
        gtol: GradientRMS = GradientRMS(1e-3, "ha/ang"),
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
        self._gtol = GradientRMS(gtol, "ha/ang")
        self._species: Optional["Species"] = None
        self._method: Optional["Method"] = None
        self._n_cores: Optional[int] = None
        self._history = OptimiserHistory()

        self._hessian_updater = BofillUpdate

    @property
    def completed(self) -> bool:
        """
        IRC is completed if (1) the maximum number of points requested
        have been integrated, or (2) the gradient has decreased below
        the requested tolerance
        """
        if self.n_points >= self._maxpoints or self._g_norm < self._gtol:
            return True
        else:
            return False

    @property
    def _g_norm(self):
        """
        RMS(∇E) of the current Cartesian gradient

        Returns:
            (autode.values.GradientRMS): Gradient norm. Infinity if the
                                          gradient is not defined
        """
        if self._coords is None:
            logger.warning("Had no coordinates - cannot determine ||∇E||")
            return GradientRMS(np.inf)

        if self._coords.g is None:
            return GradientRMS(np.inf)

        cart_g = self._coords.to("cart", pass_tensors=True)
        return GradientRMS(np.sqrt(np.average(np.square(cart_g))))

    @abstractmethod
    def _initialise_run(self):
        pass

    @abstractmethod
    def _first_step(self) -> OptCoordinates:
        pass

    @abstractmethod
    def _predictor_step(self) -> OptCoordinates:
        pass

    @abstractmethod
    def _corrector_step(self, coords: OptCoordinates):
        pass

    @property
    def _coords(self) -> Optional[OptCoordinates]:
        if len(self._history) == 0:
            return None
        return self._history[-1]

    @_coords.setter
    def _coords(self, value):
        if isinstance(value, OptCoordinates):
            self._history.append(value.copy())
        else:
            raise ValueError

        if _flush_old_hessians:
            old_coords = self._history[-3]
            if old_coords is not None:
                old_coords.h = None

    @property
    def n_points(self):
        """Number of points integrated so far"""
        return len(self._history) - 1

    def run(self, species: "Species", method: "Method"):
        self._initialise_species_and_method(species, method)
        self._initialise_run()

        logger.info(
            f"Integrating a maximum of {self._maxpoints} points"
            f" in {self._direction} direction"
        )

        self._first_step()
        while not self.completed:
            if self.n_points == 1 and self._direction != "downhill":
                pred_coords = self._first_step()
            else:
                pred_coords = self._predictor_step()
            if (
                self._recalc_hess_freq is not None
                and self.n_points % self._recalc_hess_freq == 0
            ):
                self._update_hessian_gradient_and_energy_for(pred_coords)
            else:
                self._update_gradient_and_energy_for(pred_coords)
                self._update_hessian_by_formula_for(pred_coords, self._coords)
            self._corrector_step(pred_coords)

        logger.info(
            f"Finished integrating IRC pathway: generated"
            f" {self.n_points} on the reaction path"
        )

    def _initialise_species_and_method(self, species, method):

        from autode.species.species import Species
        from autode.wrappers.methods import Method

        if not isinstance(species, Species):
            raise ValueError(
                f"{species} must be a autoode.Species instance "
                f"but had {type(species)}"
            )

        if not isinstance(method, Method):
            raise ValueError(
                f"{method} must be a autoode.wrappers.base.Method "
                f"instance but had {type(method)}"
            )

        if species.constraints.any:
            raise NotImplementedError(
                "IRC with constraints is not implemented"
            )

        hess_calc_reqd = (
            self._should_init_hess or self._recalc_hess_freq is not None
        )

        if (
            hess_calc_reqd
            and method.keywords.hess.bstring != ""
            and method.keywords.hess.bstring != method.keywords.grad.bstring
        ):
            raise CalculationException(
                "At least one Hessian calculation is required for this IRC"
                "run, however, hessian keywords are different from gradient"
                "keywords (OR keywords are not detectable)"
            )

        self._species, self._method = species, method

    def _update_gradient_and_energy_for(self, coords: OptCoordinates):

        self._species.coordinates = coords.to("cart")
        from autode.calculations import Calculation

        engrad_calc = Calculation(
            name=f"{self._species.name}_irc_{self.n_points}_engrad",
            molecule=self._species,
            method=self._method,
            keywords=self._method.keywords.grad,
            n_cores=self._n_cores,
        )

        engrad_calc.run()
        engrad_calc.clean_up(force=True, everything=True)

        coords.update_g_from_cart_g(self._species.gradient)
        coords.e = self._species.energy

        if self._species.gradient is None:
            raise CalculationException(
                "Calculation failed to calculate a gradient. "
                "Cannot continue!"
            )

        return None

    def _update_hessian_gradient_and_energy_for(self, coords: OptCoordinates):

        self._species.coordinates = coords.to("cart")
        from autode.calculations import Calculation

        hess_calc = Calculation(
            name=f"{self._species.name}_irc_{self.n_points}_hess",
            molecule=self._species,
            method=self._method,
            keywords=self._method.keywords.hess,
            n_cores=self._n_cores,
        )

        hess_calc.run()
        hess_calc.clean_up(force=True, everything=True)

        if self._species.hessian is None or self._species.gradient is None:
            raise CalculationException(
                "Calculation failed to calculate a hessian/gradient. "
                "Cannot continue!"
            )

        coords.update_h_from_cart_h(self._species.hessian)
        coords.update_g_from_cart_g(self._species.gradient)
        coords.e = self._species.energy

        return None

    def _update_hessian_by_formula_for(
        self, coords: OptCoordinates, old_coords: OptCoordinates
    ):
        """
        Update Hessian by using a formula instead of calculation

        Args:
            coords:
        """
        assert old_coords.h is not None
        updater = self._hessian_updater(
            h=old_coords.h,
            s=coords.raw - old_coords.raw,
            y=coords.g - old_coords.g,
        )

        coords.h = updater.updated_h
        return None
