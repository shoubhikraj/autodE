import numpy as np
from typing import Optional, Union, TYPE_CHECKING
from autode.bracket.imagepair import EuclideanImagePair
from autode.bracket.base import BaseBracketMethod
from autode.opt.coordinates import CartesianCoordinates
from autode.values import Energy, Distance
from autode.utils import ProcessPool
from autode.neb import NEB
from autode.log import logger

if TYPE_CHECKING:
    from autode.species import Species
    from autode.wrappers.methods import Method


def calculate_energy_for_species(
    species: "Species", method: "Method", n_cores: int
):
    """
    Convenience function to calculate the energy for a given species

    Args:
        species (Species): The species object
        method (Method): The method (low_sp keywords will be used)
        n_cores (int): The number of cores

    Returns:
        (Energy): The single point energy of the species
    """
    from autode import Calculation

    sp_calc = Calculation(
        name=f"{species.name}_sp",
        molecule=species,
        method=method,
        keywords=method.keywords.low_sp,  # NOTE: We use low_sp
        n_cores=n_cores,
    )

    sp_calc.run()
    sp_calc.clean_up(force=True, everything=True)

    return species.energy


class BinaryImagePair(EuclideanImagePair):
    def __init__(
        self,
        left_image: "Species",
        right_image: "Species",
        alpha: float = 10.0,
        beta: float = 0.1,
    ):
        super().__init__(left_image, right_image)

        # Parameters for BITSS
        self._alpha = abs(float(alpha))
        self._beta = abs(float(beta))

        # BITSS distance and energy constraint vars
        self._k_dist: Optional[float] = None  # units = Ha/ang**2
        self._k_eng: Optional[float] = None  # units = 1/Ha
        self._target_dist: Optional[Distance] = None

    def ts_guess(self) -> Optional["Species"]:
        """
        For BITSS method, images can rise and fall in energy, so we take
        the highest energy image from the last two coordinates. If peak
        from CI-NEB is available we return it.

        Returns:
            (Species): The peak species
        """

        def species_from_coords(coords) -> "Species":
            tmp_spc = self._left_image.new_species(name="peak")
            tmp_spc.coordinates = coords
            tmp_spc.energy = coords.e
            tmp_spc.gradient = coords.g.reshape(-1, 3).copy()
            return tmp_spc

        if self._cineb_coords is not None:
            assert self._cineb_coords.e is not None
            coordinates = self._cineb_coords
            return species_from_coords(coordinates)

        assert self.left_coord.e and self.right_coord.e
        if self.left_coord.e > self.right_coord.e:
            coordinates = self.left_coord
        else:
            coordinates = self.right_coord

        return species_from_coords(coordinates)

    @property
    def target_dist(self):
        return self._target_dist

    @target_dist.setter
    def target_dist(self, value: Union[Distance, float]):
        if not isinstance(value, float):
            raise TypeError
        assert value > 0
        self._target_dist = Distance(value, "ang")

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
            (Energy): BITSS energy in Hartree
        """
        assert self.left_coord.e and self.right_coord.e
        # E_BITSS = E1 + E2 + k_e (E1 - E2)**2 + k_d (d - d_i)**2
        e1 = float(self.left_coord.e.to("Ha"))
        e2 = float(self.right_coord.e.to("Ha"))
        return Energy(
            e1
            + e2
            + self._k_eng * (e1 - e2) ** 2
            + self._k_dist * (self.dist - self._target_dist) ** 2,
            units="Ha",
        )

    @property
    def bitss_grad(self):
        """
        Calculate the gradient of the BITSS energy at the current
        geometry

        Returns:
            (np.ndarray): The gradient as flat array in Ha/ang units
        """
        assert self.left_coord.g is not None
        assert self.right_coord.g is not None

        grad1 = self.left_coord.g
        grad2 = self.right_coord.g
        e1 = float(self.left_coord.e.to("Ha"))
        e2 = float(self.right_coord.e.to("Ha"))
        # energy gradient terms from coordinates of image 1 (x_1)
        # ∇E1 * (1 + 2 * k_e * (E1-E2))
        left_term = grad1 * (1 + 2 * self._k_eng * (e1 - e2))
        # energy gradient terms from coordinates of image 2 (x_2)
        # ∇E2 * (1 + 2 * k_e * (E2-E1))
        right_term = grad2 * (1 + 2 * self._k_eng * (e2 - e1))
        # distance gradient terms from both images
        dist_term = (
            2
            * (1 / self.dist)
            * np.concatenate((self.dist_vec, -self.dist_vec))
            * self._k_dist
            * (self.dist - self._target_dist)
        )
        # total gradient
        return np.concatenate((left_term, right_term)) + dist_term

    def _get_barrier_estimate(self, image_density: float = 1.0) -> Energy:
        """
        Get the current value of the estimated barrier by running
        an interpolation between the current images and then
        calculating the energies at interpolated points

        Args:
            image_density (float): Number of images per Angstrom to
                                   generate for the interpolation

        Returns:
            (Energy): The barrier estimate
        """
        assert self.left_coord.e and self.right_coord.e

        logger.info("Running IDPP interpolation to estimate barrier")
        n_images = int(image_density * self.dist)
        neb = NEB.from_end_points(
            self._left_image, self._right_image, n_images
        )
        n_cores_pp = max(self._n_cores // n_images, 1)
        n_workers = n_images if n_images < self._n_cores else self._n_cores

        with ProcessPool(max_workers=n_workers) as pool:
            jobs = [
                pool.submit(
                    calculate_energy_for_species,
                    species=img.new_species(name=f"img{idx}"),
                    method=self._method,
                    n_cores=n_cores_pp,
                )
                for idx, img in list(enumerate(neb.images))[1:-1]
                # skip the first and last image as they already have energies
            ]

            path_energies = [job.result() for job in jobs]

        # TODO: handle negative barriers here or anywhere
        # E_B = max(interpolated energies) - avg(left_img, right_img)
        e_b = max(path_energies) - (self.left_coord.e + self.right_coord.e) / 2
        return Energy(e_b, "Ha")

    def update_constraints(self):
        """
        Update the BITSS constraint parameters by using estimated
        barrier and current gradients
        """
        assert self.left_coord.g is not None
        assert self.right_coord.g is not None

        # k_e = alpha / E_B
        e_b = self._get_barrier_estimate()
        self._k_eng = self._alpha / float(e_b.to("Ha"))

        # k_d = max(
        # sqrt(|∇E1|**2) + sqrt(|∇E2|**2) / (2 * sqrt(2) * beta * d_i),
        # E_B / (beta * d_i**2)
        # )
        proj_left_g = abs(np.dot(self.left_coord.g, self.dist_vec) / self.dist)
        proj_right_g = abs(
            np.dot(self.right_coord.g, self.dist_vec) / self.dist
        )
        k_d_1 = np.sqrt(proj_left_g**2, proj_right_g**2)
        k_d_1 /= 2 * np.sqrt(2) * self._beta * self._target_dist

        k_d_2 = e_b / (self._beta * self._target_dist) ** 2
        self._k_dist = float(max(k_d_1, k_d_2))

        logger.info(
            f"Updated constraints: κ_e = {self._k_eng}, κ_d = {self._k_dist}"
        )
        return None


class BITSS(BaseBracketMethod):
    def __init__(
        self,
        initial_species,
        final_species,
        *args,
        reduction_fac=0.4,
        alpha=10,
        beta=0.1,
        **kwargs,
    ):
        super().__init__(initial_species, final_species, *args, **kwargs)

        self.imgpair = BinaryImagePair(
            initial_species, final_species, alpha=alpha, beta=beta
        )
        self._fac = abs(float(reduction_fac))
        self._current_macroiters: int = 0

    @property
    def _macro_iter(self) -> int:
        return self._current_macroiters

    @_macro_iter.setter
    def _macro_iter(self, value: int):
        self._current_macroiters = int(value)

    @property
    def _micro_iter(self) -> int:
        return self.imgpair.total_iters // 2

    def update_target_distance(self):
        self.imgpair.target_dist = self.imgpair.target_dist * (1 - self._fac)
        return None

    def _initialise_run(self) -> None:
        self.imgpair.update_both_img_engrad()
        self.imgpair.target_dist = self.imgpair.dist * (1 - self._fac)

    def _step(self) -> None:

        pass
