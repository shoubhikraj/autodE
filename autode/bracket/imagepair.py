"""
Base classes for implementing all bracketing methods
that require a pair of images
"""

from typing import Optional, Tuple
import numpy as np

from autode.values import PotentialEnergy, Gradient
from autode.hessians import Hessian
from autode.geom import get_rot_mat_kabsch
from autode.opt.coordinates import CartesianCoordinates, OptCoordinates
from autode.opt.optimisers.hessian_update import BofillUpdate
from autode.opt.optimisers.base import _OptimiserHistory
from autode.utils import work_in_tmp_dir, ProcessPool
from autode.exceptions import CalculationException
from autode.log import logger

import autode.species.species
import autode.wrappers.methods


def _calculate_engrad_for_species(
    species: autode.species.species.Species,
    method: autode.wrappers.methods.Method,
    n_cores: int,
) -> Tuple[PotentialEnergy, Gradient]:
    from autode.calculations import Calculation

    engrad_calc = Calculation(
        name=f"{species.name}_engrad",
        molecule=species,
        method=method,
        keywords=method.keywords.grad,
        n_cores=n_cores,
    )
    engrad_calc.run()
    engrad_calc.clean_up(force=True, everything=True)
    return species.energy, species.gradient


def _calculate_energy_for_species(
    species: autode.species.species.Species,
    method: autode.wrappers.methods.Method,
    n_cores: int,
) -> PotentialEnergy:
    from autode.calculations import Calculation

    sp_calc = Calculation(
        name=f"{species.name}_sp",
        molecule=species,
        method=method,
        keywords=method.keywords.sp,
        n_cores=n_cores,
    )
    sp_calc.run()
    sp_calc.clean_up(force=True, everything=True)
    return species.energy


@work_in_tmp_dir()
def _calculate_hessian_for_species(
    species: autode.species.species.Species,
    method: autode.wrappers.methods.Method,
    n_cores: int,
) -> Hessian:
    """
    Convenience function for calculating the Hessian for a
    molecule; removes all input and output files for the
    calculation
    """
    from autode.calculations import Calculation

    hess_calc = Calculation(
        name=f"{species.name}_hess",
        molecule=species,
        method=method,
        keywords=method.keywords.hess,
        n_cores=n_cores,
    )
    hess_calc.run()
    hess_calc.clean_up(force=True, everything=True)
    return species.hessian


class ImgPairSideError(ValueError):
    """
    Error if side is neither 'left' nor 'right', used only for internal
    consistency, as the functions should not be called by user
    """

    def __init__(self):
        super().__init__("Side supplied must be either 'left' or 'right'")


class BaseImagePair:
    """
    Base class for a pair of images (e.g., reactant and product) of
    the same species. The images are called 'left' and 'right' to
    distinguish them, but there is no requirement for one to be
    reactant or product.
    """

    def __init__(
        self,
        left_image: autode.species.species.Species,
        right_image: autode.species.species.Species,
    ):
        """
        Initialize the image pair, does not set methods/n_cores or
        hessian update types!

        Args:
            left_image: One molecule of the pair
            right_image: Another molecule of the pair
        """
        assert isinstance(left_image, autode.species.species.Species)
        assert isinstance(right_image, autode.species.species.Species)
        self._left_image = left_image.new_species(name="left_image")
        self._right_image = right_image.new_species(name="right_image")
        self._sanity_check()
        self._align_species()
        # todo do you need to update the grad and hess of the species?

        # separate methods for engrad and hessian calc
        self._engrad_method = None
        self._hess_method = None
        self._n_cores = None
        # Bofill has no conditions, so kept as default
        self._hessian_update_types = [BofillUpdate]

        self._left_history = _OptimiserHistory()
        self._right_history = _OptimiserHistory()
        # push the first coordinates into history
        self.left_coord = self._left_image.coordinates.to("ang").flatten()
        self.right_coord = self._right_image.coordinates.to("ang").flatten()

    def _sanity_check(self) -> None:
        """
        Check if the two supplied images have the same solvent,
        charge, multiplicity and the same atoms in the same order
        """

        if self._left_image.n_atoms != self._right_image.n_atoms:
            raise ValueError(
                "The initial_species and final_species must "
                "have the same number of atoms!"
            )

        if (
            self._left_image.charge != self._right_image.charge
            or self._left_image.mult != self._right_image.mult
            or self._left_image.solvent != self._right_image.solvent
        ):
            raise ValueError(
                "Charge/multiplicity/solvent of initial_species "
                "and final_species supplied are not the same"
            )

        for idx in range(len(self._left_image.atoms)):
            if (
                self._left_image.atoms[idx].label
                != self._right_image.atoms[idx].label
            ):
                raise ValueError(
                    "The order of atoms in initial_species "
                    "and final_species must be the same. The "
                    f"atom at position {idx} is different in"
                    "the two species"
                )

        return None

    def _align_species(self) -> None:
        """
        Translates both molecules to origin and then performs
        a Kabsch rotation to orient the molecules as close as
        possible against each other
        """
        # first translate the molecules to the origin
        logger.info(
            "Translating initial_species (reactant) "
            "and final_species (product) to origin"
        )
        p_mat = self._left_image.coordinates.copy()
        p_mat -= np.average(p_mat, axis=0)
        self._left_image.coordinates = p_mat

        q_mat = self._right_image.coordinates.copy()
        q_mat -= np.average(q_mat, axis=0)
        self._right_image.coordinates = q_mat

        logger.info(
            "Rotating initial_species (reactant) "
            "to align with final_species (product) "
            "as much as possible"
        )
        rot_mat = get_rot_mat_kabsch(p_mat, q_mat)
        rotated_p_mat = np.dot(rot_mat, p_mat.T).T
        self._left_image.coordinates = rotated_p_mat

    def set_method_and_n_cores(
        self,
        engrad_method: autode.wrappers.methods.Method,
        n_cores: int,
        hess_method: Optional[autode.wrappers.methods.Method] = None,
    ) -> None:
        """
        Sets the methods for engrad and hessian calculation, and the
        total number of cores used for any calculation in this image pair

        Args:
            engrad_method (autode.wrappers.methods.Method):
            n_cores (int):
            hess_method (autode.wrappers.methods.Method|None):
        """
        if not isinstance(engrad_method, autode.wrappers.methods.Method):
            raise ValueError(
                f"The engrad_method needs to be of type autode."
                f"wrappers.method.Method, But "
                f"{type(engrad_method)} was supplied."
            )
        self._engrad_method = engrad_method
        if hess_method is None:
            pass
        elif not isinstance(hess_method, autode.wrappers.methods.Method):
            raise ValueError(
                f"The hess_method needs to be of type autode."
                f"wrappers.method.Method, But "
                f"{type(hess_method)} was supplied."
            )
        self._hess_method = hess_method
        self._n_cores = int(n_cores)
        return None

    @property
    def n_atoms(self) -> int:
        """Number of atoms"""
        return self._left_image.n_atoms

    @property
    def total_iters(self) -> int:
        """Total number of iterations done on this image pair"""
        return len(self._left_history) + len(self._right_history) - 2

    @property
    def left_coord(self) -> Optional[OptCoordinates]:
        """The coordinates of the left image"""
        if len(self._left_history) == 0:
            return None
        return self._left_history[-1]

    @left_coord.setter
    def left_coord(self, value: Optional[OptCoordinates]):
        """
        Sets the coordinates of the left image, also updates
        the coordinates of the species

        Args:
            value (OptCoordinates|None): new set of coordinates

        Raises:
            (TypeError): On invalid input
        """
        if value is None:
            return
        elif isinstance(value, OptCoordinates):
            self._left_history.append(value.copy())
        else:
            raise TypeError

        self._left_image.coordinates = value.to("cart")
        # todo should we remove old hessians that are not needed?

    @property
    def right_coord(self) -> Optional[OptCoordinates]:
        """The coordinates of the right image"""
        if len(self._right_history) == 0:
            return None
        return self._right_history[-1]

    @right_coord.setter
    def right_coord(self, value: Optional[OptCoordinates]):
        """
        Sets the coordinates of the right image, also updates
        the coordinates of the species

        Args:
            value (OptCoordinates|None): new set of coordinates

        Raises:
            (TypeError): On invalid input
        """
        if value is None:
            return
        elif isinstance(value, OptCoordinates):
            self._right_history.append(value.copy())
        else:
            raise TypeError

        self._right_image.coordinates = value.to("cart")

    def get_coord_by_side(self, side: str) -> OptCoordinates:
        """For external usage, supplies only the coordinate object"""
        _, coord, _, _ = self._get_img_by_side(side)
        return coord

    def _get_img_by_side(
        self, side: str
    ) -> Tuple[autode.Species, OptCoordinates, _OptimiserHistory, float]:
        """
        Access an image and some properties by a string that
        represents side. Returns a tuple of the species, the
        current coordinate object, and a factor that is necessary
        for calculation

        Args:
            side (str): 'left' or 'right'

        Returns:
            (tuple) : tuple(image, current coord, history, fac)
        """
        if side == "left":
            img = self._left_image
            coord = self.left_coord
            hist = self._left_history
            fac = 1.0
        elif side == "right":
            img = self._right_image
            coord = self.right_coord
            hist = self._right_history
            fac = -1.0
        else:
            raise ImgPairSideError()

        return img, coord, hist, fac

    def update_one_img_mol_energy(self, side: str) -> None:
        """
        Update only the molecular energy using the supplied
        engrad_method for one image only

        Args:
            side (str): 'left' or 'right'
        """
        assert self._engrad_method is not None
        assert self._n_cores is not None
        img, coord, _, _ = self._get_img_by_side(side)

        logger.debug(
            f"Calculating energy for {side} side"
            f" with {self._engrad_method}"
        )

        en = _calculate_energy_for_species(
            species=img.copy(),
            method=self._engrad_method,
            n_cores=self._n_cores,
        )
        # update coord
        coord.e = en.to("Ha")

    def update_one_img_mol_engrad(self, side: str) -> None:
        """
        Update the molecular energy and gradient using the supplied
        engrad_method for one image only

        Args:
            side (str): 'left' or 'right'
        """
        assert self._engrad_method is not None
        assert self._n_cores is not None
        img, coord, _, _ = self._get_img_by_side(side)

        logger.debug(
            f"Calculating engrad for {side} side"
            f" with {self._engrad_method}"
        )
        en, grad = _calculate_engrad_for_species(
            species=img.copy(),
            method=self._engrad_method,
            n_cores=self._n_cores,
        )
        # update coord
        coord.e = en.to("Ha")
        coord.update_g_from_cart_g(grad.to("Ha/ang"))
        return None

    def update_one_img_mol_hess_by_calc(self, side: str) -> None:
        """
        Updates the molecular hessian using supplied hess_method
        for one image only

        Args:
            side (str): 'left' or 'right'
        """
        assert self._hess_method is not None
        assert self._n_cores is not None
        img, coord, _, _ = self._get_img_by_side(side)

        logger.debug(
            f"Calculating Hessian for {side} side" f" with {self._hess_method}"
        )
        hess = _calculate_hessian_for_species(
            species=img.copy(), method=self._hess_method, n_cores=self._n_cores
        )
        # update coord
        coord.update_h_from_cart_h(hess)
        return None

    def update_one_img_mol_hess_by_formula(self, side: str) -> None:
        """
        Updates the molecular hessian of one side by using Hessian
        update formula; requires the gradient and hessian for the
        last coordinates, and gradient for the current coordinates

        Args:
            side (str): 'left' or 'right'
        """
        img, coord, hist, _ = self._get_img_by_side(side)
        assert len(hist) > 1, "Hessian update not possible!"
        assert coord.h is None, "Hessian already exists!"
        assert coord.g is not None, "Gradient should be present!"
        last_coord = hist.penultimate
        for update_type in self._hessian_update_types:
            updater = update_type(
                h=last_coord.h,
                h_inv=last_coord.h_inv,
                s=coord.raw - last_coord.raw,
                y=coord.g - last_coord.g,
                subspace_idxs=coord.indexes,
            )
            if not updater.conditions_met:
                continue

            coord.h = updater.updated_h
            break

        assert coord.h is not None, "Hessian update failed!"

        return None


class TwoSidedImagePair(BaseImagePair):
    """
    Derived from BaseImagePair, with the capability of doing
    engrad and Hessian calculations parallely for both
    images at the same time
    """

    def update_both_img_mol_engrad(self):
        """
        Update the molecular energy and gradient using the supplied
        engrad_method for both images in parallel (if possible)
        """
        assert self._engrad_method is not None
        assert self._n_cores is not None

        logger.error(
            f"Calculating engrad for both sides" f" with {self._engrad_method}"
        )

        if self._n_cores == 1:  # no need for parallel
            self.update_one_img_mol_engrad("left")
            self.update_one_img_mol_engrad("right")
            return None

        n_cores_per_pp = self._n_cores // 2
        with ProcessPool(max_workers=2) as pool:
            engrad_jobs = [
                pool.submit(
                    _calculate_engrad_for_species,
                    species=img.copy(),
                    method=self._engrad_method,
                    n_cores=n_cores_per_pp,
                )
                for img in [self._left_image, self._right_image]
            ]
            left_en, left_grad = engrad_jobs[0].result()
            right_en, right_grad = engrad_jobs[1].result()

        # cast into units
        right_en, right_grad = right_en.to("Ha"), right_grad.to("Ha/ang")
        left_en, left_grad = left_en.to("Ha"), left_grad.to("Ha/ang")

        # update both coords
        self.left_coord.e = left_en
        self.left_coord.update_g_from_cart_g(left_grad)

        self.right_coord.e = right_en
        self.right_coord.update_g_from_cart_g(right_grad)

        return None

    def update_both_img_mol_hess_by_calc(self) -> None:
        """
        Updates the molecular hessian using supplied hess_method
        for both images in parallel (if possible)
        """
        assert self._hess_method is not None
        assert self._n_cores is not None

        logger.error(
            "Calculating Hessian for both sides" f" with {self._hess_method}"
        )

        if self._n_cores == 1:
            self.update_one_img_mol_hess_by_calc("left")
            self.update_one_img_mol_hess_by_calc("right")
            return None

        n_cores_per_pp = self._n_cores // 2
        with ProcessPool(max_workers=2) as pool:
            hess_jobs = [
                pool.submit(
                    _calculate_hessian_for_species,
                    species=img.copy(),
                    method=self._hess_method,
                    n_cores=n_cores_per_pp,
                )
                for img in [self._left_image, self._right_image]
            ]
            left_hess = hess_jobs[0].result()
            right_hess = hess_jobs[1].result()

        # cast into base units
        left_hess = left_hess.to("Ha/ang^2")
        right_hess = right_hess.to("Ha/ang^2")

        # update both coords
        self.left_coord.update_h_from_cart_h(left_hess)
        self.right_coord.update_h_from_cart_h(right_hess)

        return None

    def update_both_img_mol_hess_by_formula(self) -> None:
        """
        Updates the molecular hessians of both images by using an
        update formula
        """
        # simply call the functions for each side
        self.update_one_img_mol_hess_by_formula("left")
        self.update_one_img_mol_hess_by_formula("right")
