import numpy as np

from autode.bracket.imagepair import TwoSidedImagePair
from autode.values import Distance
from autode.opt.coordinates import OptCoordinates, CartesianCoordinates


class BinaryImagePair(TwoSidedImagePair):
    """
    A Binary-Image pair use for the BITSS procedure for
    transition state search
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._k_eng = None  # energy constraint
        self._k_dist = None  # distance constraint
        self._d_i = None  # d_i

    def _check_bitss_params_defined(self):
        assert self._d_i is not None
        assert self._k_eng is not None
        assert self._k_dist is not None

    @property
    def dist_vec(self) -> np.ndarray:
        """The distance vector pointing to right_image from left_image"""
        return np.array(self.left_coord - self.right_coord)

    @property
    def dist(self) -> Distance:
        """
        Distance between BITSS images. (Currently implemented
        as Euclidean distance in Cartesian)
        Returns:
            (Distance):
        """
        return Distance(np.linalg.norm(self.dist_vec), 'ang')

    @property
    def target_dist(self) -> Distance:
        """
        The target distance (d_i) set for BITSS
        Returns:
            (Distance)
        """
        return Distance(self._d_i, 'ang')

    @target_dist.setter
    def target_dist(self, value):
        """
        Set the target distance(d_i) used for BITSS
        Args:
            value (Distance|float):
        """
        if value is None:
            return
        if isinstance(value, Distance):
            self._d_i = float(value.to('ang'))
        elif isinstance(value, float):
            self._d_i = value
        else:
            raise ValueError("Unknown type")
        assert self._d_i > 0, "Must be positive!"

    @property
    def bitss_coords(self) -> OptCoordinates:
        """
        The BITSS coordinates. Concatenated coordinates
        of the left and right images

        Returns:
            (OptCoordinates):
        """
        return CartesianCoordinates(
            np.concatenate((self.left_coord, self.right_coord))
        )

    @bitss_coords.setter
    def bitss_coords(self, value):
        if isinstance(value, OptCoordinates):
            coords = value.to('ang').flatten()
        elif isinstance(value, np.ndarray):
            coords = value.flatten()
        else:
            raise ValueError("Unknown type")

        if coords.shape[0] != (3 * 2 * self.n_atoms):
            raise ValueError("Coordinates have the wrong dimensions")

        self.left_coord = coords[:3 * self.n_atoms]
        self.right_coord = coords[3 * self.n_atoms:]

    def bitss_energy(self) -> float:
        """
        Calculate the value of the BITSS potential (energy)

        Returns:
            (float): energy in Hartree
        """
        # E_BITSS = E_1 + E_2 + k_e(E_1 - E_2)^2 + k_d(d-d_i)^2
        e_1 = float(self.left_coord.e)
        e_2 = float(self.right_coord.e)
        return float(e_1 + e_2 + self._k_eng * (e_1 - e_2)**2
                     + self._k_dist * (self.dist - self._d_i)**2)

    def bitss_grad(self):
        """
        Calculate the gradient of the BITSS energy (in Hartree/Angstrom)

        Returns:
            (np.ndarray): flat gradient of shape (n_atoms * 3 * 2,)
        """
        assert self.left_coord.g is not None
        assert self.right_coord.g is not None
        self._check_bitss_params_defined()

        # energy terms arising from coordinates of left image (r_1)
        # = grad(E_1) * (1 + 2 * k_e * (E_1 - E_2))
        left_term = self.left_coord.g * (
            1 + 2 * self._k_eng * (self.left_coord.e - self.right_coord.e)
        )
        # energy terms arising from coordinates of right image (r_2)
        # = grad(E_2) * (1 + 2 * k_e * (E_2 - E_1)) # notice signs of energy
        right_term = self.right_coord.g * (
            1 + 2 * self._k_eng * (self.right_coord.e - self.left_coord.e)
        )
        # distance term
        # = grad(d) * 2 * k_d * (d - d_i)
        dist_term = (
            (1/self.dist) * np.concatenate((self.dist_vec, -self.dist_vec))
            * 2 * self._k_dist * (self.dist - self._d_i)
        )
        # form total gradient
        return np.concatenate((left_term, right_term)) + dist_term

    def bitss_hess(self):
        pass
