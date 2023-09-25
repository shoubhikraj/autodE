"""
Freezing String Method to find transition states. This is
implemented along-side the bracket methods only due to
programmatic ease

References:
[1] A. Behn et al., J. Chem. Phys., 2011, 135, 224108 (original)
[2] S. Sharada et al., J. Chem. Theory Comput., 2012, 8, 5166-5174 (improved)
"""
from typing import Any, Tuple, Optional, Union, TYPE_CHECKING
import numpy as np

from autode.values import PotentialEnergy
from autode.neb import NEB
from autode.utils import ProcessPool
from autode.path.interpolation import CubicPathSpline
from autode.bracket.imagepair import EuclideanImagePair
from autode.bracket.base import BaseBracketMethod
from autode.opt.coordinates import CartesianCoordinates
from autode.opt.optimisers.rfo import RFOptimiser
from autode.opt.optimisers.hessian_update import BFGSSR1Update, BFGSPDUpdate

if TYPE_CHECKING:
    from autode.species.species import Species


class TangentQNROptimiser(RFOptimiser):
    def __init__(
        self,
        maxiter: int,
        tangent: np.ndarray,
        energy_eps: Union[PotentialEnergy, float],
        gtol=1e-3,
        etol=1e-4,
        max_step: float = 0.06,
        line_search_sigma: float = 0.7,
    ):
        super().__init__(
            init_alpha=max_step,
            maxiter=maxiter,
            gtol=gtol,
            etol=etol,
        )
        # todo remove translation rotation before generating tangent
        self._tau_hat = tangent / np.linalg.norm(tangent)
        # prefer BFGS as it is good for minimisation
        self._hessian_update_types = [BFGSPDUpdate, BFGSSR1Update]
        self._eps = PotentialEnergy(energy_eps).to("Ha")
        self._sigma = abs(float(line_search_sigma))

    def _update_gradient_and_energy(self) -> None:
        """
        Update the gradient and energy, removing the component along
        the tangent vector
        """
        super()._update_gradient_and_energy()
        assert self._coords is not None
        g_parall = np.dot(self._tau_hat, self._coords.g) * self._tau_hat
        self._coords.g = self._coords.g - g_parall

    def _initialise_run(self) -> None:
        assert self._species is not None
        self._coords = CartesianCoordinates(
            self._species.coordinates.to("ang")
        )
        self._coords.remove_tr = True
        assert len(self._tau_hat) == len(self._coords)
        self._coords.update_h_from_cart_h(self._low_level_cart_hessian)
        self._remove_tangent_from_hessian()
        self._update_gradient_and_energy()
        # TODO: make the hessian positive definite (min eigval option)
        # TODO: is RFO step enough

    def _remove_tangent_from_hessian(self) -> None:
        # Frank Jensen, Introduction to Computational Chemistry, 565
        # this is probably not the full transform, but that is ok as
        # an approximate hessian is sufficient
        assert self._coords is not None
        assert self._coords.h is not None
        x_k = self._tau_hat.reshape(-1, 1)
        p_k = np.eye(x_k.flatten().shape[0]) - np.matmul(x_k, x_k.T)
        self._coords.h = np.linalg.multi_dot([p_k.T, self._coords.h, p_k])
        return None

    def _step(self):
        """RFO step, ignoring the zeroed TR modes"""
        self._coords.h = self._updated_h()
        h_n, _ = self._coords.h.shape
        aug_H = np.zeros(shape=(h_n + 1, h_n + 1))

        aug_H[:h_n, :h_n] = self._coords.h
        aug_H[-1, :h_n] = self._coords.g
        aug_H[:h_n, -1] = self._coords.g

        aug_H_lmda = np.linalg.eigvalsh(aug_H)
        mode = np.where(np.abs(aug_H_lmda) > 1.0e-14)[0][0]
        shift_lmda = aug_H_lmda[mode]

        b, u = np.linalg.eigh(self._coords.h)
        f = u.T.dot(self._coords.g)
        delta_s = np.zeros_like(self._coords)

        for i in range(h_n):
            if b <= 1.0e-14:
                continue
                # todo check below formula
            delta_s -= f[i] * u[:, i] / (b[i] - shift_lmda)

        self._take_step_within_trust_radius(delta_s)

    def _get_adjusted_step(self, delta_s):
        s_hat = delta_s / np.linalg.norm(delta_s)
        # Determine the scaling factor from eqn (8) in ref [2]
        if self.iteration == 0:
            guess_alpha = self.alpha  # not clear from paper!
        else:
            e_delta = float(max(self.last_energy_change, -self._eps))
            guess_alpha = -2 * e_delta / np.dot(self._coords.g, s_hat)

        def sigma_estimate(alpha):
            return 1.0 - alpha / np.linalg.norm(
                np.matmul(np.linalg.inv(self._coords.h), self._coords.g)
            )

        if sigma_estimate(guess_alpha) <= self._sigma:
            return guess_alpha * s_hat
        else:
            for _ in range(50):
                guess_alpha = guess_alpha * 0.95
                if sigma_estimate(guess_alpha) <= self._sigma:
                    return guess_alpha * s_hat

            raise RuntimeError("Unable to find the correct step size")


def _optimise_get_coords(species, tau, method, n_cores, maxiter):
    opt = TangentQNROptimiser(maxiter=maxiter, tangent=tau)
    opt.run(species, method, n_cores)
    return CartesianCoordinates(species.coordinates), opt.iteration


def _parallel_optimise_tangent(
    new_nodes: tuple, taus: tuple, method, n_cores, maxiter: int
):
    # TODO: species list and tau list
    # todo check these formula
    n_procs = 2 if n_cores > 2 else 1
    n_cores_pp = max(int(n_cores // 2), 1)
    with ProcessPool(max_workers=n_procs) as pool:
        jobs = [
            pool.submit(
                _optimise_get_coords, mol, tau, method, n_cores_pp, maxiter
            )
            for mol, tau in zip(new_nodes, taus)
        ]
        result = [job.result() for job in jobs]

    assert len(result) == 2
    new_coords = (result[0][0], result[1][0])
    total_iters = result[0][1] + result[1][1]
    return new_coords, total_iters


def _align_coords_to_ref(
    align_coords: CartesianCoordinates, ref_coords: CartesianCoordinates
):
    from autode.geom import get_rot_mat_kabsch

    p_mat = align_coords.reshape(-1, 3)
    q_mat = ref_coords.reshape(-1, 3)

    q_mat_centroid = np.average(q_mat, axis=0)
    # translate to origin, then to ref's centroid
    p_mat -= np.average(p_mat, axis=0)
    p_mat += q_mat_centroid

    rot_mat = get_rot_mat_kabsch(p_mat, q_mat)
    rotated_p_mat = np.dot(rot_mat, p_mat.T).T

    align_coords[:] = rotated_p_mat.flatten()


class FSMPath(EuclideanImagePair):
    def __init__(
        self,
        left_image: "Species",
        right_image: "Species",
        step_size,
        maxiter_per_node: int,
        use_idpp: bool = True,
    ):
        super().__init__(left_image=left_image, right_image=right_image)

        self._step_size = step_size  # todo distance
        self._max_n = abs(int(maxiter_per_node))
        assert self._max_n > 0
        self._energy_eps = 0.0

    @property
    def ts_guess(self) -> Optional["Species"]:
        energies = [coords.e for coords in self._total_history]
        assert all(en is not None for en in energies), "Energy value missing"
        peak_idx = np.argmax(energies)
        assert peak_idx != 0 and peak_idx != len(self._total_history)
        tmp_spc = self._left_image.new_species(name="peak")
        peak_coords = self._total_history[peak_idx]
        tmp_spc.coordinates = peak_coords
        return tmp_spc

    @property
    def has_jumped_over_barrier(self) -> bool:
        return False

    def grow_string(self, use_idpp: bool = True):
        assert 0 < self._step_size < self.dist
        interp_density = max(int(self.dist / self._step_size), 1) * 10

        if not use_idpp:
            step = self.dist_vec * (self._step_size / self.dist)
            left_new = self.left_coords - step
            right_new = self.right_coords + step
            # Not clear what the tangents are for cartesian, so we
            # take the Cartesian direction, may be changed later
            left_tau = right_tau = self.dist_vec / self.dist
            result = _parallel_optimise_tangent(
                (left_new, right_new),
                (left_tau, right_tau),
                method=self._method,
                n_cores=self._n_cores,
                maxiter=self._max_n,
            )
            self._add_coordinates(result[0])
            # return result[1]
            # TODO: optimise here and return

        idpp = NEB.from_end_points(
            self._left_image, self._right_image, num=interp_density
        )
        spline = CubicPathSpline.from_species_list(idpp.images)
        # TODO: remove rotation, translation?
        left_dists = []
        right_dists = []
        for point in idpp.images:
            coords = CartesianCoordinates(point.coordinates)
            left_dists.append(np.linalg.norm(coords - self.left_coords))
            right_dists.append(np.linalg.norm(coords - self.right_coords))

        left_next_idx = np.argmin(np.array(left_dists) - self._step_size)
        right_next_idx = np.argmin(np.array(right_dists) - self._step_size)
        nodes = [
            idpp.images[left_next_idx],
            idpp.images[right_next_idx],
        ]
        tangents = [
            spline.tangent_at(spline.path_distances[idx])
            for idx in (left_next_idx, right_next_idx)
        ]
        # todo rename species list tau list etc.

    def _get_new_coords_tangents_cartesian(self) -> Tuple[tuple, tuple]:
        """
        Obtain the next set of new node coordinates and tangent,
        using linear Cartesian interpolation

        Returns:
            (tuple):
        """
        step = self.dist_vec * (self._step_size / self.dist)
        left_new = self.left_coords - step
        right_new = self.right_coords + step
        _align_coords_to_ref(left_new, self.left_coords)
        _align_coords_to_ref(right_new, self.right_coords)
        # NOTE: It is not clear from the paper what the tangents are
        # for cartesian, so we simply take the Cartesian direction
        left_tau = left_new - self.left_coords
        right_tau = right_new - self.right_coords
        return (left_new, right_new), (left_tau, right_tau)

    def _get_new_coords_tangents_idpp(self) -> Tuple[tuple, tuple]:
        """
        Obtain the next set of new node coordinates and tangent,
        using IDPP interpolation. Cubic spline is used for tangents

        Returns:
            (tuple):
        """
        # take a high density interpolation and choose closest to step size
        interp_density = max(int(self.dist / self._step_size), 1) * 10
        idpp = NEB.from_end_points(
            self._left_image, self._right_image, num=interp_density
        )
        assert len(idpp.images) > 2
        coords_list: list = []
        for point in idpp.images:
            coords = CartesianCoordinates(point.coordinates)
            if len(coords_list) > 0:
                _align_coords_to_ref(coords, coords_list[-1])
            coords_list.append(coords)
        assert np.all((coords_list[-1] - self.right_coords) < 1.0e-10)
        left_dists, right_dists = [], []
        for coords in coords_list:
            left_dists.append(np.linalg.norm(self.left_coords - coords))
            right_dists.append(np.linalg.norm(self.right_coords - coords))
        left_idx = np.argmin(np.array(left_dists) - self._step_size)
        right_idx = np.argmin(np.array(right_dists) - self._step_size)
        assert np.isclose(left_dists[left_idx], self._step_size, rtol=5e-2)
        assert np.isclose(right_dists[right_idx], self._step_size, rtol=5e-2)
        nodes = (coords_list[left_idx], coords_list[right_idx])
        spline = CubicPathSpline(coords_list)
        tangents = tuple(
            spline.tangent_at(spline.path_distances[idx])
            for idx in [left_idx, right_idx]
        )
        return nodes, tangents

    def _add_coordinates(self, new_nodes: tuple):
        # todo add new coords by removing translation and rotation
        pass

    def estimate_energy_epsilon(self):
        assert self.left_coords.e and self.right_coords.e
        delta_e = abs(self.left_coords.e - self.right_coords.e)
        upper_lim = PotentialEnergy(2.5, "kcal/mol")
        if delta_e < upper_lim:
            self._energy_eps = float(delta_e.to("Ha"))
        else:
            self._energy_eps = float(upper_lim.to("Ha"))
        return None


class FSM(BaseBracketMethod):
    def __init__(
        self,
        initial_species,
        final_species,
        step_size: float = 0.2,
        maxiter_per_node: int = 6,
        use_idpp: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            initial_species=initial_species, final_species=final_species
        )
        self.imgpair: FSMPath = FSMPath(
            left_image=initial_species,
            right_image=final_species,
            maxiter_per_node=maxiter_per_node,
            step_size=step_size,
            use_idpp=use_idpp,
        )
        self._current_microiters = 0

    @property
    def _macro_iter(self) -> int:
        return int(self.imgpair.total_iters / 2)

    @property
    def _micro_iter(self) -> int:
        return self._current_microiters

    @_micro_iter.setter
    def _micro_iter(self, value: int):
        self._current_microiters = value

    def _initialise_run(self) -> None:
        self.imgpair.update_both_img_engrad()
        self.imgpair.estimate_energy_epsilon()

    def _step(self) -> None:
        iters = self.imgpair.grow_string()
        self._micro_iter += iters

        pass
