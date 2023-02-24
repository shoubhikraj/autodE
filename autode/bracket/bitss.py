import numpy as np

from autode.bracket.imagepair import (
    TwoSidedImagePair,
    _calculate_energy_for_species,
)
from autode.values import Distance, GradientRMS
from autode.opt.coordinates import OptCoordinates, CartesianCoordinates
from autode.neb import NEB
from autode.exceptions import OptimiserStepError
from autode.utils import ProcessPool
from autode.log import logger

import autode.species


class BinaryImagePair(TwoSidedImagePair):
    """
    A Binary-Image pair use for the BITSS procedure for
    transition state search
    """

    def __init__(self, *args, alpha=10.0, beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)

        self._alpha = float(abs(alpha))
        self._beta = float(abs(beta))

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
    def bitss_iters(self):
        # in BITSS both sides should updated at the same time
        assert self.total_iters % 2 == 0
        return self.total_iters / 2

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
        return Distance(np.linalg.norm(self.dist_vec), "ang")

    @property
    def target_dist(self) -> Distance:
        """
        The target distance (d_i) set for BITSS

        Returns:
            (Distance)
        """
        return Distance(self._d_i, "ang")

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
            self._d_i = float(value.to("ang"))
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
        """
        Sets the bitss coordinates. Expects concatenated
        coordinates of left and right images and then sets
        the coordinates (which updates the species)

        Args:
            value (np.ndarray|OptCoordinates):
        """
        if isinstance(value, OptCoordinates):
            coords = value.to("ang").flatten()
        elif isinstance(value, np.ndarray):
            coords = value.flatten()
        else:
            raise ValueError("Unknown type")

        if coords.shape[0] != (3 * 2 * self.n_atoms):
            raise ValueError("Coordinates have the wrong dimensions")

        self.left_coord = coords[: 3 * self.n_atoms]
        self.right_coord = coords[3 * self.n_atoms :]

    def update_bitss_constraints(self) -> None:
        """
        Updates the BITSS constraint parameters k_eng and k_dist
        """

        logger.info("Updating BITSS constraint parameters")

        e_b = self._get_estimated_barrier()

        # k_e  = alpha / (2 * E_B)
        self._k_eng = self._alpha / e_b

        # k_d = max(
        # sqrt(|grad(E_1)|^2 + sqrt(|grad(E_2)|^2) / (2 * sqrt(2) * beta * d_i),
        # E_B / (beta * d_i^2)
        # )
        # first, project the gradients
        left_g_proj = abs(
            np.dot(self.left_coord.g, self.dist_vec)
            / np.linalg.norm(self.dist_vec)
        )
        right_g_proj = abs(
            np.dot(self.right_coord.g, self.dist_vec)
            / np.linalg.norm(self.dist_vec)
        )
        k_d_1 = np.sqrt(left_g_proj**2 + right_g_proj**2)
        k_d_1 = k_d_1 / (2 * np.sqrt(2) * self._beta * self.target_dist)

        k_d_2 = e_b / (self._beta * self.target_dist**2)

        self._k_dist = max(k_d_1, k_d_2)

        return None

    def _get_estimated_barrier(self, n_images=8) -> float:
        """
        Get the current value of estimated barrier by running a linear
        interpolation using the engrad method

        Args:
            n_images (int): Number of images to use for interpolation

        Returns:
            (float): Energy in Hartree
        """
        logger.info(
            f"Using a linear interpolation of {n_images} to estimate the"
            f"current barrier of the BITSS image pair"
        )

        lin_path = LinearInterp(
            self._left_image.copy(), self._right_image.copy(), num=n_images
        )
        n_cores_pp = max(self._n_cores // n_images, 1)
        n_workers = n_images if n_images < self._n_cores else self._n_cores

        # only need to calculate the middle images
        with ProcessPool(max_workers=n_workers) as pool:
            jobs = [
                pool.submit(
                    _calculate_energy_for_species,
                    species=image.species.new_species(name=f"img{idx}"),
                    method=self._engrad_method,
                    n_cores=n_cores_pp,
                )
                for idx, image in enumerate(lin_path.images)[1:-1]
            ]

            path_energies = [job.result() for job in jobs]

        # E_B = max(interpolated E's) - avg(reactant, product)
        e_b = max(path_energies) - (self.left_coord.e + self.right_coord.e) / 2
        return float(e_b)

    def bitss_energy(self) -> float:
        """
        Calculate the value of the BITSS potential (energy)

        Returns:
            (float): energy in Hartree
        """
        # E_BITSS = E_1 + E_2 + k_e(E_1 - E_2)^2 + k_d(d-d_i)^2
        e_1 = float(self.left_coord.e)
        e_2 = float(self.right_coord.e)
        return float(
            e_1
            + e_2
            + self._k_eng * (e_1 - e_2) ** 2
            + self._k_dist * (self.dist - self._d_i) ** 2
        )

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
            (1 / self.dist)
            * np.concatenate((self.dist_vec, -self.dist_vec))
            * 2
            * self._k_dist
            * (self.dist - self._d_i)
        )
        # form total gradient
        return np.concatenate((left_term, right_term)) + dist_term

    def rms_bitss_grad(self) -> GradientRMS:
        grad = self.bitss_grad()
        rms_g = np.sqrt(np.average(np.mean(grad)))
        return GradientRMS(rms_g, units="ha/ang")

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
        i_n = np.eye(self.n_atoms)
        a_mat = np.vstack((np.hstack((i_n, -i_n)), np.hstack((-i_n, i_n))))
        total_coord_col = self.bitss_coords.reshape(-1, 1)
        grad_d = float(1 / self.dist) * (a_mat @ total_coord_col)
        hess_d = float(1 / self.dist) * (a_mat - (grad_d @ grad_d.T))
        distance_term = 2 * self._k_dist * (grad_d @ grad_d.T)

        distance_term += (
            2
            * float(self._k_dist)
            * float(self.dist)
            * float(1 - 2 * self.target_dist)
            * hess_d
        )
        # put together distance and energy terms
        total_hess = energy_terms + distance_term

        return total_hess


class LinearInterp(NEB):
    """
    Generates a linear interpolation with a fixed number of
    images between the geometries of two species
    """

    # subclasses NEB to simply skip IDPP relaxation
    def _init_from_end_points(self, initial, final) -> None:
        """Only interpolation, no IDPP"""
        self.images[0].species = initial
        self.images[-1].species = final
        self.interpolate_geometries()

        return None


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
        dist_tol: Distance = Distance(1.0, "ang"),
        reduction_factor: float = 0.5,
        gtol: GradientRMS = GradientRMS(5.0e-4, "ha/ang"),
        init_trust: Distance = Distance(0.05, "ang"),
        max_trust: Distance = Distance(0.2, "ang"),
        min_trust: Distance = Distance(0.01, "ang"),
        constr_update_freq: int = 25,
    ):
        self.imgpair = BinaryImagePair(initial_species, final_species)
        self._dist_tol = Distance(dist_tol, "ang")
        self._gtol = GradientRMS(gtol, "ha/ang")

        self._engrad_method = None
        self._hess_method = None
        self._maxiter = abs(int(maxiter))
        self._reduction_fac = abs(float(reduction_factor))

        self._trust = float(Distance(init_trust, "ang"))
        self._max_tr = float(Distance(max_trust, "ang"))
        self._min_tr = float(Distance(min_trust, "ang"))

        # todo doc if negative turn off trust update
        if self._trust < 0:
            self._tr_upd = False
            self._trust = abs(self._trust)
        else:
            self._tr_upd = True

        self._constr_upd = int(abs(constr_update_freq))

    @property
    def converged(self) -> bool:
        if self.imgpair.dist < self._dist_tol:
            return True
        else:
            return False

    @property
    def _exceeded_maximum_iterations(self) -> bool:
        return True if self.imgpair.bitss_iters > self._maxiter else False

    def calculate(
        self,
        engrad_method: "autode.wrappers.methods.Method",
        hess_method: "autode.wrappers.methods.Method",
        n_cores: int,
    ):

        self.imgpair.set_method_and_n_cores(
            engrad_method=engrad_method,
            hess_method=hess_method,
            n_cores=n_cores,
        )
        logger.info("Starting BITSS optimisation to find transition state")

        while not self.converged:
            self._reduce_target_dist()
            if not self._bitss_minimise():
                logger.warning("Exceeded maximum num of iterations in BITSS")
                break

        logger.info(
            f"BITSS optimisation finished in "
            f"{self.imgpair.total_iters} geometry iteration steps."
        )

    def _reduce_target_dist(self) -> None:
        """
        Reduces the target distance for BITSS. (This can be
        considered as a BITSS macro-iteration step)
        """
        self.imgpair.target_dist = (
            1 - self._reduction_fac
        ) * self.imgpair.dist
        logger.info(
            f"BITSS macro-iteration: Setting target distance"
            f"to {self.imgpair.target_dist:.4f}; Current distance ="
            f" {self.imgpair.dist}"
        )
        return None

    @property
    def _microiter_converged(self) -> bool:
        """
        Whether optimiser "micro-iterations" are converged.
        Checks whether distance is close to the set value of
        target distance for current macro-iteration, and whether
        RMS gradient of BITSS energy is lower than global gtol

        Returns:
            (bool): True if converged, False otherwise
        """
        dist_criteria_met = np.isclose(
            self.imgpair.dist, self.imgpair.target_dist, atol=5.0e-4
        )
        grad_criteria_met = self.imgpair.rms_bitss_grad() < self._gtol
        if dist_criteria_met and grad_criteria_met:
            return True
        else:
            return False

    def _bitss_minimise(self) -> bool:
        """
        Minimises the BITSS potential with micro-iterations of
        a hybrid RFO/QA method

        Returns:
            (bool): True if micro-iterations converged otherwise False
        """
        self.imgpair.update_both_img_molecular_engrad()
        self.imgpair.update_both_img_molecular_hessian_by_calc()
        self.imgpair.update_bitss_constraints()
        micro_iter = 0
        # todo should I be calculating hessian after every macro-iter?
        while not self._microiter_converged:
            micro_iter += 1
            if self._exceeded_maximum_iterations:
                return False
            self._microiter_step()
            self.imgpair.update_both_img_molecular_engrad()
            # todo check stability is it better to update molecular hessian
            # or just the bitss hessian
            self.imgpair.update_both_img_molecular_hessian_by_formula()
            if self._microiter_converged:
                break
            if micro_iter % self._constr_upd == 0:
                self.imgpair.update_bitss_constraints()
            self._log_convergence()

        return True

    def _update_trust_radius(self):
        # todo
        pass

    def _microiter_step(self) -> None:
        hess = self.imgpair.bitss_hess()
        grad = self.imgpair.bitss_grad()

        rfo_step = _get_rfo_minimise_step(hess, grad)
        step_size = np.linalg.norm(rfo_step)

        # check if step is in trust radius
        if step_size <= self._trust:
            # take an RFO step
            logger.info("Taking a pure RFO step")
            step = rfo_step
        else:
            # try a QA step within trust radius
            try:
                logger.info("Taking a QA step within trust radius")
                step = _get_qa_minimise_step(hess, grad, self._trust)

            except OptimiserStepError:
                # if that didn't work, simply scale the rfo step
                logger.info("Taking a scaled RFO step")
                step = rfo_step * self._trust / step_size

        new_coords = self.imgpair.bitss_coords + step
        self.imgpair.bitss_coords = new_coords

        return None

    def _log_convergence(self):
        logger.info(
            f"BITSS micro-iter # {self.imgpair.total_iters}: initial "
            f"species E={self.imgpair.left_coord.e}, final "
            f"species E={self.imgpair.right_coord.e}, RMS of BITSS"
            f" grad={self.imgpair.rms_bitss_grad()}, Distance="
            f"{self.imgpair.dist}"
        )


def _get_rfo_minimise_step(
    hessian: np.ndarray, gradient: np.ndarray
) -> np.ndarray:
    """
    Using current Hessian and gradient, obtain an RFO (Rational
    Function Optimisation) minimising step

    Args:
        hessian (np.ndarray):
        gradient (np.ndarray):

    Returns:
        (np.ndarray): The step in flattened array
    """
    h_n = hessian.shape[0]

    # form the augmented Hessian
    aug_h = np.zeros(shape=(h_n + 1, h_n + 1), dtype=np.float64)
    aug_h[:h_n, :h_n] = hessian
    aug_h[-1, :h_n] = gradient
    aug_h[:h_n, -1] = gradient

    aug_h_lmda, aug_h_v = np.linalg.eigh(aug_h)
    # RF step uses the lowest non-zero eigenvalue
    mode = np.where(np.abs(aug_h_lmda) > 1.0e-10)[0][0]

    # step is scaled by the final element of eigenvector
    delta_s = aug_h_v[:-1, mode] / aug_h_v[-1, mode]

    return delta_s.flatten()


def _get_qa_minimise_step(
    hessian: np.ndarray, gradient: np.ndarray, trust: float
) -> np.ndarray:
    """
    Using current Hessian and gradient, get a minimising
    step, whose magnitude (norm) is equal to the trust
    radius (Quadratic Approximation step). Described in
    J. Golab, D. L. Yeager, Chem. Phys., 78, 1983, 175-199

    Args:
        hessian (np.ndarray):
        gradient (np.ndarray):
        trust (float):
    Returns:
        (np.ndarray): the step as a flat array
    """
    from scipy.optimize import root_scalar

    n = hessian.shape[0]
    h_eigvals = np.linalg.eigvalsh(hessian)
    first_mode = np.where(np.abs(h_eigvals) > 1.0e-10)[0][0]
    first_b = h_eigvals[first_mode]  # first non-zero eigenvalue of H

    def step_length_error(lmda):
        shifted_h = hessian - lmda * np.eye(n)  # level-shifted hessian
        inv_shifted_h = np.linalg.inv(shifted_h)
        step = -inv_shifted_h @ gradient.reshape(-1, 1)
        return np.linalg.norm(step) - trust

    # The value of shift parameter lambda must lie within (-infinity, first_b)
    # Need to find the roots of the 1D function step_length_error
    l_plus = 1.0
    for _ in range(1000):
        err = step_length_error(first_b - l_plus)
        if err > 0.0:  # found location where f(x) > 0
            break
        l_plus *= 0.5
    else:  # if loop didn't break
        raise OptimiserStepError("Unable to find lambda where error > 0")

    l_minus = l_plus - 1.0
    for _ in range(1000):
        err = step_length_error(first_b - l_minus)
        if err < 0.0:  # found location where f(x) < 0
            break
        l_minus -= 1.0  # reduce by 1.0
    else:
        raise OptimiserStepError("Unable to find lambda where error < 0")

    # Use scipy's root finder
    res = root_scalar(
        step_length_error,
        method="brentq",
        bracket=[l_minus, l_plus],
        maxiter=500,
    )

    if not res.converged:
        raise OptimiserStepError("Unable to find root of error function")

    lmda_final = res.root
    shift_h = lmda_final * np.eye(n) - hessian
    inv_shift_h = np.linalg.inv(shift_h)
    delta_s = -inv_shift_h @ gradient.reshape(-1, 1)

    return delta_s.flatten()
