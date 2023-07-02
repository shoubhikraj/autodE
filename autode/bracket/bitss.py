import numpy as np
from typing import Optional, TYPE_CHECKING
from autode.bracket.imagepair import EuclideanImagePair
from autode.opt.coordinates import CartesianCoordinates

if TYPE_CHECKING:
    from autode.species import Species


class BinaryImagePair(EuclideanImagePair):
    def ts_guess(self) -> Optional["Species"]:
        """
        For BITSS method, images can rise and fall in energy, so we take
        the highest energy image from the last two coordinates. If peak
        from CI-NEB is available we return it.

        Returns:
            (Species): The peak species
        """

        def species_from_coords(coords) -> Species:
            tmp_spc = self._left_image.new_species(name="peak")
            tmp_spc.coordinates = coords
            tmp_spc.energy = coords.e
            tmp_spc.gradient = coords.g.reshape(-1, 3).copy()
            return tmp_spc

        if self._cineb_coords is not None:
            assert self._cineb_coords.e is not None
            coordinates = self._cineb_coords

        assert self.left_coord.e and self.right_coord.e
        if self.left_coord.e > self.right_coord.e:
            coordinates = self.left_coord
        else:
            coordinates = self.right_coord

        return species_from_coords(coordinates)

    @property
    def bitss_coords(self) -> CartesianCoordinates:
        """
        The total coordinates of the BITSS image pair system

        Returns:
            (CartesianCoordinates):
        """
        return CartesianCoordinates(
            np.concatenate((self.left_coord, self.right_coord), axis=None)
        )

    @bitss_coords.setter
    def bitss_coords(self, value: CartesianCoordinates):
        """
        Set the coordinates of each image from the total
        set of coordinates for BITSS image pair system

        Args:
            value:

        Returns:

        """
        if not isinstance(value, CartesianCoordinates):
            raise TypeError
        if not value.shape != (3 * 2 * self.n_atoms,):
            raise ValueError("Coordinates have wrong dimensions")

        self.left_coord = CartesianCoordinates(value[: 3 * self.n_atoms])
        self.right_coord = CartesianCoordinates(value[3 * self.n_atoms :])

    @property
    def bitss_energy(self):
        """
        Calculate the value of the BITSS potential (energy)

        Returns:

        """
        return
