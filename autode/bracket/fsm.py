"""
Freezing String Method to find transition states. This is
implemented along-side the bracket methods only due to
programmatic ease

References:
[1] A. Behn et al., J. Chem. Phys., 2011, 135, 224108 (original)
[2] S. Sharada et al., J. Chem. Theory Comput., 2012, 8, 5166-5174 (improved)
"""
from typing import Any, Optional, TYPE_CHECKING
import numpy as np

from autode.neb import NEB
from autode.utils import ProcessPool
from autode.bracket.imagepair import EuclideanImagePair
from autode.bracket.base import BaseBracketMethod
from autode.opt.coordinates import CartesianCoordinates
from autode.opt.optimisers.trm import HybridTRMOptimiser

if TYPE_CHECKING:
    from autode.species.species import Species


class TangentQNROptimiser(HybridTRMOptimiser):
    def __init__(
        self,
        maxiter: int,
        tangent: np.ndarray,
        gtol=1e-3,
        etol=1e-4,
        trust_radius: float = 0.1,
    ):
        super().__init__(
            init_trust=trust_radius,
            update_trust=False,
            damp=False,
            maxiter=maxiter,
            gtol=gtol,
            etol=etol,
        )
        self._tau_hat = tangent / np.linalg.norm(tangent)

    def _update_gradient_and_energy(self) -> None:
        super()._update_gradient_and_energy()
        assert self._coords is not None
        g_parall = np.dot(self._tau_hat, self._coords.g) * self._tau_hat
        self._coords.g = self._coords.g - g_parall

    def _initialise_run(self) -> None:
        assert self._species is not None
        self._coords = CartesianCoordinates(
            self._species.coordinates.to("ang")
        )
        assert len(self._tau_hat) == len(self._coords)
        self._coords.update_h_from_cart_h(self._low_level_cart_hessian)
        self._remove_tangent_from_hessian()
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
    n_cores_per_pp = max(int(n_cores // 2), 1)
    with ProcessPool(max_workers=n_procs) as pool:
        jobs = [
            pool.submit(
                _optimise_get_coords, mol, tau, method, n_cores, maxiter
            )
            for mol, tau in zip(new_nodes, taus)
        ]
        result = [job.result() for job in jobs]

    assert len(result) == 2
    new_coords = (result[0][0], result[1][0])
    total_iters = result[0][1] + result[1][1]
    return new_coords, total_iters


class FSMPath(EuclideanImagePair):
    def __init__(
        self,
        left_image: "Species",
        right_image: "Species",
        step_size,
        maxiter_per_node: int,
    ):
        super().__init__(left_image=left_image, right_image=right_image)

        self._step_size = step_size  # todo distance
        self._max_n = abs(int(maxiter_per_node))
        assert self._max_n > 0

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

    def grow_string(self, step_size: float, use_idpp: bool = True):
        assert 0 < step_size < self.dist
        interp_density = max(int(self.dist / step_size), 1) * 10

        if not use_idpp:
            step = self.dist_vec * (step_size / self.dist)
            left_new = self.left_coords - step
            right_new = self.right_coords + step
            # Not clear what the tangents are for cartesian, so we
            # take the Cartesian direction, may be changed later
            left_tau = right_tau = self.dist_vec / self.dist
            # TODO: optimise here and return

        idpp = NEB.from_end_points(
            self._left_image, self._right_image, num=interp_density
        )
        # TODO: remove rotation, translation?
        left_dists = []
        right_dists = []
        for point in idpp.images:
            coords = CartesianCoordinates(point.coordinates)
            left_dists.append(np.linalg.norm(coords - self.left_coords))
            right_dists.append(np.linalg.norm(coords - self.right_coords))

        left_next_idx = np.argmin(np.array(left_dists) - self._step_size)
        right_next_idx = np.argmin(np.array(right_dists) - self._step_size)
        species_list = [
            idpp.images[left_next_idx],
            idpp.images[right_next_idx],
        ]
        tau_list = [
            _get_tau_from_spline_at(idpp, idx)
            for idx in (left_next_idx, right_next_idx)
        ]
        # todo rename species list tau list etc.


def _get_tau_from_spline_at(images, idx):
    pass


class FSM(BaseBracketMethod):
    def __init__(self, initial_species, final_species):
        super().__init__(
            initial_species=initial_species, final_species=final_species
        )
