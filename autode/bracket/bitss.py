import numpy as np

from autode.bracket.imagepair import TwoSidedImagePair
from autode.values import Distance, GradientRMS
from autode.opt.coordinates import OptCoordinates, CartesianCoordinates

import autode.species


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

    def _check_bitss_params_grad_defined(self):
        assert self.left_coord.g is not None
        assert self.right_coord.g is not None

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

    def bitss_grad(self) -> np.ndarray:
        """
        Calculate the gradient of the BITSS energy (in Hartree/Angstrom)

        Returns:
            (np.ndarray): flat gradient of shape (n_atoms * 3 * 2,)
        """

        self._check_bitss_params_grad_defined()

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

    def rms_bitss_grad(self) -> GradientRMS:
        grad = self.bitss_grad()
        rms_g = np.sqrt(np.average(np.mean(grad)))
        return GradientRMS(rms_g, units='ha/ang')

    def bitss_hess(self):
        """
        Calculate the Hessian(second derivative) of the BITSS
        energy, using the molecular Hessians

        Returns:
            (np.ndarray): Hessian array with shape (3 * n_atoms, 3 * n_atoms)
        """
        assert self.left_coord.h is not None
        assert self.right_coord.h is not None
        self._check_bitss_params_grad_defined()

        # terms from E_1, E_2 in upper left square of Hessian
        upper_left_sq = self.left_coord.h * (
            1
            + float(2 * self.left_coord.e * self._k_eng)
            - float(2 * self.right_coord.e * self._k_eng)
        )
        upper_left_sq += (
            2 * self._k_eng * np.outer(self.left_coord.g, self.left_coord.g)
        )

        # terms from E_1, E_2 in lower right square of Hessian
        lower_right_sq = self.right_coord.h * (
            1
            - float(2 * self.left_coord.e * self._k_eng)
            + float(2 * self.right_coord.e * self._k_eng)
        )
        lower_right_sq += (
            2 * self._k_eng * np.outer(self.right_coord.g, self.right_coord.g)
        )

        # terms from E_1, E_2 in upper right square of Hessian
        upper_right_sq = (
            -2 * self._k_eng * np.outer(self.left_coord.g, self.right_coord.g)
        )

        # terms from E_1, E_2 in lower left square of Hessian
        lower_left_sq = (
            -2 * self._k_eng * np.outer(self.right_coord.g, self.left_coord.g)
        )

        # build up the energy terms
        upper_part = np.hstack((upper_left_sq, upper_right_sq))
        lower_part = np.hstack((lower_left_sq, lower_right_sq))
        energy_terms = np.vstack((upper_part, lower_part))

        # distance terms
        # grad(d) * 2 * k_d * (d - d_i)
        total_coord_col = self.bitss_coords.reshape(-1, 1)
        # todo finish
        # todo do we really need Hessians?


class BITSS:
    """
    Binary-Image Transition State Search. It begins with
    two images (e.g. reactant and product) and then minimises
    their energies under two constraints, energy and distance.
    The energy constraint ensures that the images do not jump
    over one another, and the distance constraint ensures that
    the images are pulled closer, towards the transition state
    """
    def __init__(
        self,
        initial_species: autode.species.Species,
        final_species: autode.species.Species,
        maxiter: int = 200,
        dist_tol: Distance = Distance(1.0, 'ang'),
        reduction_factor: float = 0.5,
        gtol: GradientRMS = GradientRMS(5.0e-4, 'ha/ang')
    ):
        self.imgpair = BinaryImagePair(initial_species, final_species)
        self._dist_tol = Distance(dist_tol, 'ang')
        self._gtol = GradientRMS(gtol, 'ha/ang')

        self._engrad_method = None
        self._hess_method = None
        self._maxiter = abs(int(maxiter))
        self._reduction_fac = abs(float(reduction_factor))

    @property
    def converged(self):
        if self.imgpair.dist < self._dist_tol:
            return True
        else:
            return False

    def calculate(self):
        self.imgpair.update_one_img_molecular_engrad('left')
        self.imgpair.update_one_img_molecular_engrad('right')

        while not self.converged:
            self._reduce_target_dist()

    def _reduce_target_dist(self) -> None:
        """
        Reduces the target distance for BITSS. (This can be
        considered as a BITSS macro-iteration step)
        """
        self.imgpair.target_dist = (1-self._reduction_fac) * self.imgpair.dist
        return None

    @property
    def microiter_converged(self):
        dist_criteria_met = np.isclose(self.imgpair.dist,
                                       self.imgpair.target_dist, atol=5.e-4)
        grad_criteria_met = self.imgpair.rms_bitss_grad() < self._gtol
        if dist_criteria_met and grad_criteria_met:
            return True
        else:
            return False

    def bitss_minimise(self):
        """
        Minimises the BITSS potential with RFO method
        """
        pass

