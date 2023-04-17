from typing import Union, TYPE_CHECKING, Optional
import numpy as np
from autode.opt.coordinates import OptCoordinates, CartesianCoordinates
from autode.units import ang_amu_half

if TYPE_CHECKING:
    from autode.units import Unit
    from autode.species.species import Species


class MWCartesianCoordinates(OptCoordinates):
    """Mass-weighted Cartesian coordinates of shape = (3 × n_atoms, )"""

    implemented_units = [ang_amu_half]

    def __repr__(self):
        return (
            f"MW Cartesian Coordinates"
            f"({np.ndarray.__str__(self)} {self.units.name})"
        )

    def __new__(
        cls, input_array: np.ndarray, units: Union[str, "Unit"] = "Å amu^1/2"
    ):
        arr = super().__new__(cls, np.array(input_array).flatten(), units)

        # to store square root of atomic masses for conversion
        arr.sqrt_masses = None
        # distance along reaction coordinate
        arr.ircdist = None

        return arr

    def __array_finalize__(self, obj: Optional["OptCoordinates"]) -> None:
        if obj is None:
            return
        OptCoordinates.__array_finalize__(self, obj)

        for attr in ("sqrt_masses", "ircdist"):
            setattr(self, attr, getattr(obj, attr, None))

        return

    def iadd(self, value: np.ndarray) -> OptCoordinates:
        return np.ndarray.__iadd__(self, value)

    def __add__(self, other: Union[np.ndarray, float]) -> "OptCoordinates":
        """
        Addition of another set of coordinates. Clears the current
        gradient vector and Hessian matrix.

        -----------------------------------------------------------------------
        Arguments:
            other (np.ndarray): Array to add to the coordinates

        Returns:
            (autode.opt.coordinates.OptCoordinates): Shifted coordinates
        """
        new_coords = self.copy()
        new_coords.clear_tensors()
        # take a reference instead of copy to reduce memory footprint
        new_coords.sqrt_masses = self.sqrt_masses
        new_coords.iadd(other)

        return new_coords

    @classmethod
    def from_species(cls, species: "Species"):

        sqrt_masses = np.sqrt(np.array(species.atomic_masses))
        if len(sqrt_masses) == 0:
            raise RuntimeError("Species must have atomic masses")

        mwcoords = np.array(species.coordinates.flatten()) * sqrt_masses

        coords = cls(mwcoords)
        coords.sqrt_masses = sqrt_masses

        return coords

    def _update_g_from_cart_g(
        self, arr: Optional["autode.values.Gradient"]
    ) -> None:
        """
        Updates the gradient from a calculated Cartesian gradient, for
        mass-weighted cartesian coordinates, that simply means dividing
        my the square root mass matrix

        Args:
            arr: Gradient array
        """
        if arr is not None:
            arr = np.array(arr).flatten() / self.sqrt_masses

        self.g = arr

    def _update_h_from_cart_h(
        self, arr: Optional["autode.values.Hessian"]
    ) -> None:
        if arr is not None:
            mass_matrix = np.outer(self.sqrt_masses, self.sqrt_masses)
            arr = np.array(arr) / mass_matrix

        self.h = arr

    def to(self, value: str, pass_tensors=False) -> "OptCoordinates":

        if value.lower() in ["cart", "cartesian", "cartesiancoordinates"]:
            cart_coords = CartesianCoordinates(self / self.sqrt_masses)
            if pass_tensors:
                mass_matrix = np.outer(self.sqrt_masses, self.sqrt_masses)
                cart_coords.g = (
                    self.g * self.sqrt_masses if self.g is not None else None
                )
                cart_coords.h = (
                    self.h * mass_matrix if self.h is not None else None
                )
            return cart_coords

        else:
            raise ValueError(f"Cannot convert to {value}")
