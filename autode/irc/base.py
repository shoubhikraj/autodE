"""
Base classes for IRC calculation
"""
from abc import ABC, abstractmethod
from typing import Optional, Union, TYPE_CHECKING
import numpy as np
from autode.values import GradientRMS, MWDistance, Energy
from autode.opt.coordinates import OptCoordinates, MWCartesianCoordinates
from autode.opt.optimisers.base import OptimiserHistory
from autode.opt.optimisers.hessian_update import BofillUpdate
from autode.exceptions import CalculationException
from autode.log import logger
from autode.config import Config

if TYPE_CHECKING:
    from autode.species.species import Species
    from autode.wrappers.methods import Method

_flush_old_hessians = True


class BaseIntegrator(ABC):
    """
    Base class for reaction coordinate integrators
    """

    def __init__(
        self,
        max_points: int,
        step_size: float,
        gtol: GradientRMS = GradientRMS(1e-3, "ha/ang"),
        read_init_hess: bool = False,
        recalc_hess: Optional[int] = None,
        direction: str = "forward",
    ):
        """
        Create a new reaction coordinate integrator. The read_init_hess
        argument is very important, because if set to True, the hessian
        from the species is read in. Please be careful when using this
        because the hessian in the species must be of the same level
        as the method you are using for the IRC run (i.e. the Hessian
        must be accurate), especially for Hessian based integrators.

        The integration continues until max_points are integrated, or
        the gradient falls below gtol, signalling that a minimum has
        been reached.

        Args:
            max_points (int): Maximum number of points to integrate
            step_size (float): The size of each integration step
            gtol (GradientRMS): The gradient tolerance below which
                                integration is stopped (minima reached)
            read_init_hess (bool): Whether to read the initial Hessian
                                   from the species provided in run()
            recalc_hess (int|None): Frequency of hessian calculation for
                                    predictor steps, None means never
            direction (str): 'forward', 'reverse' or 'downhill'
        """
        super().__init__()
        self._should_init_hess = not bool(read_init_hess)

        self._recalc_hess_freq = None
        if recalc_hess is not None:
            self._recalc_hess_freq = int(recalc_hess)

        direction = direction.lower()
        if direction in ["forward", "reverse", "downhill"]:
            self._direction = direction
        else:
            raise ValueError(
                "The direction for IRC integration must either"
                "be 'forward', 'reverse' or 'downhill' "
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

        cart_g = self._coords.to("cart", pass_tensors=True).g
        return GradientRMS(np.sqrt(np.average(np.square(cart_g))))

    @property
    def last_energy_change(self):
        """Last ∆E found in this integrator"""
        if self.n_points > 0:
            delta_e = self._history.final.e - self._history.penultimate.e
            return Energy(delta_e, units="Ha")
        else:
            return Energy(np.inf)

    @abstractmethod
    def _initialise_run(self):
        """
        Initialise the integrator run, i.e. set self._coords,
        and set self._coords.g, self._coords.h etc. required
        for the first step
        """
        pass

    @abstractmethod
    def _first_step(self) -> None:
        """
        First step off the saddle point (TS), not used when downhill
        reaction coordinate integration is requested
        """

    @abstractmethod
    def _predictor_step(self) -> OptCoordinates:
        """
        Predictor that uses information from last coordinates to
        predict the next point in the reaction coordinate. This
        function must *not* set new coordinates

        Returns:
            (OptCoordinates): The predicted point
        """

    @abstractmethod
    def _corrector_step(self, coords: OptCoordinates) -> None:
        """
        Takes the predicted point from the predictor step and then
        corrects the step so that it lies more closely in the
        reaction path. The corrected point is set as the current
        coordinates in history

        Args:
            coords: Predicted coordinates
        """

    @property
    def _coords(self) -> Optional[OptCoordinates]:
        """
        Current set of coordinates in this integrator

        Returns:
            (OptCoordinates):
        """
        if len(self._history) == 0:
            return None
        return self._history[-1]

    @_coords.setter
    def _coords(self, value):
        """
        Set a new set of coordinates for this integrator

        Args:
            value (OptCoordinates): new set of coordinates
        """
        if value is None:
            return
        elif isinstance(value, OptCoordinates):
            self._history.append(value.copy())
        else:
            raise ValueError

        if _flush_old_hessians and self.n_points > 2:
            old_coords = self._history[-3]
            if old_coords is not None:
                old_coords.h = None

    @property
    def n_points(self):
        """Number of points integrated so far"""
        return len(self._history) - 1

    def run(
        self, species: "Species", method: "Method", n_cores: Optional[int]
    ) -> None:
        """
        Run a reaction coordinate integration for the species with
        the supplied method and number of cores (optional)

        Args:
            species (Species):
            method (Method):
            n_cores (int):
        """
        self._initialise_species_and_method(species, method)
        self._n_cores = int(n_cores) if n_cores is not None else Config.n_cores
        self._initialise_run()

        logger.info(
            f"Integrating a maximum of {self._maxpoints} points"
            f" in {self._direction} direction"
        )

        if self._direction != "downhill":
            self._first_step()

        while not self.completed:
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

    def _initialise_species_and_method(
        self, species: "Species", method: "Method"
    ) -> None:
        """
        Set the species and method, checking that the method
        has the same level of theory for hessian and gradient
        calculation (otherwise the IRC calculation will have
        inconsistent levels of theory)

        Args:
            species:
            method:
        """
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
            and method.keywords.hess.bstring != method.keywords.grad.bstring
        ):
            raise CalculationException(
                "At least one Hessian calculation is required for this IRC"
                "run, however, hessian keywords are different from gradient"
                "keywords (OR keywords are not detectable)"
            )

        self._species, self._method = species, method

    def _update_gradient_and_energy_for(self, coords: OptCoordinates) -> None:
        """
        Calculate the gradient and energy for a set of coordinates,
        and modify the coordinate in-place

        Args:
            coords (OptCoordinates):
        """
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

    def _update_hessian_gradient_and_energy_for(
        self, coords: OptCoordinates
    ) -> None:
        """
        Calculate the hessian, gradient and energy for a set of coordinates,
        and then modify the coordinate in-place

        Args:
            coords (OptCoordinates):
        """
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
    ) -> None:
        """
        Update Hessian by using a formula instead of calculation,
        modifies coords in place

        Args:
            coords (OptCoordinates): The new coordinates
            old_coords (OptCoordinates): The old coordinates
        """
        assert old_coords.h is not None
        updater = self._hessian_updater(
            h=old_coords.h,
            s=coords.raw - old_coords.raw,
            y=coords.g - old_coords.g,
        )

        coords.h = updater.updated_h
        return None


class MWIntegrator(BaseIntegrator, ABC):
    """
    Reaction Coordinate integrated in mass-weighted Cartesian
    coordinates, which is usually called "Intrinsic Reaction
    Coordinate" or IRC.
    """

    def __init__(
        self,
        max_points: int,
        step_size: MWDistance,
        init_step: Union[MWDistance, Energy] = Energy(1e-3, "Ha"),
        *args,
        **kwargs,
    ):
        """
        Create an IRC integrator in mass-weighted cartesian coordinates
        The init_step argument is very important, as it determines
        the first step from the TS. It can be given in either energy or
        in mass-weighted distance. If given in energy, the first
        displacement aims to reduce the energy by that amount, if in
        distance, the TS mode eigenvector is scaled back to have that
        size

        Args:
            max_points: Maximum number of points to integrate
            step_size: The step size in mass weighted distance
            init_step: The intial step, either in energy or in mw distance
        """
        super().__init__(max_points, step_size, *args, **kwargs)
        self._step_size = MWDistance(step_size, "ang amu^1/2")

        if isinstance(init_step, (Energy, MWDistance)):
            self._init_step = init_step
        else:
            raise TypeError(
                "init_step must be either autode.values.Energy or autode."
                f"values.MWDistance, but {type(init_step)} was provided"
            )

    def _initialise_run(self):
        logger.info("Initialising IRC integration run")
        if self._species is None:
            raise RuntimeError("Species must be defined before IRC run")
        self._coords = MWCartesianCoordinates.from_species(self._species)

        if self._should_init_hess:
            self._update_hessian_gradient_and_energy_for(self._coords)
        else:
            if self._species.hessian is None:
                raise RuntimeError(
                    "Requested reading initial Hessian, but species does"
                    " not have any calculated hessian data"
                )
            self._coords.update_h_from_cart_h(self._species.hessian)
            self._update_gradient_and_energy_for(self._coords)

        eigvals = np.linalg.eigvalsh(self._coords.h)
        if eigvals[0] > 0 and self._direction != "downhill":
            raise RuntimeError(
                "For IRC runs starting from TS, there must be"
                " at least one negative frequency"
            )

        if self._g_norm > self._gtol and self._direction != "downhill":
            raise RuntimeError(
                "IRC run is starting from the transition state but "
                f"gradient norm is greater then gtol: {self._gtol:.3f}."
                f" Transition state geometry is likely not converged!"
            )

    def _first_step(self) -> None:
        """
        Take the first step in the IRC run, by stepping off the saddle point
        following the imaginary TS mode
        """
        eigvals, eigvecs = np.linalg.eigh(self._coords.h)
        ts_eigvec = eigvecs[:, 0]
        ts_eigval = eigvals[0]

        logger.info(
            f"First IRC step from saddle point in {self._direction} direction"
            f" TS mode (eigenvalue) = {ts_eigval}"
        )

        scaled_ts_vec = ts_eigvec / np.linalg.norm(ts_eigvec)
        largest_comp = np.argmax(np.abs(scaled_ts_vec))
        if scaled_ts_vec[largest_comp] > 0:
            pass
        else:
            scaled_ts_vec = -scaled_ts_vec

        if isinstance(self._init_step, MWDistance):
            logger.info(
                f"Taking a step of size {self._init_step:.3f} Å amu^1/2"
            )
            step = self._init_step * scaled_ts_vec
        elif isinstance(self._init_step, Energy):
            self._init_step = self._init_step.to("Ha")
            step_length = np.sqrt(self._init_step * 2 / np.abs(ts_eigval))
            logger.info(
                f"Energy-represented step from TS: expected energy "
                f"reduction = {self._init_step} Ha"
            )
            step = step_length * scaled_ts_vec
        else:
            raise ValueError("Unknown type of initial step")

        if self._direction == "forward":
            pass
        elif self._direction == "reverse":
            step = -step
        else:
            raise ValueError("Invalid direction for starting step")

        self._coords = self._coords + step
        step_size = np.linalg.norm(step)
        self._coords.ircdist = step_size
        # todo remove ircdist in favour of list for IMK and different arguments for step size

        self._update_gradient_and_energy_for(self._coords)
        self._update_hessian_by_formula_for(self._coords, self._history[-2])
        logger.info(
            f"Energy change after first IRC step "
            f"= {self.last_energy_change}"
        )
        return None
