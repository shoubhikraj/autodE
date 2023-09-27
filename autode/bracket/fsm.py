"""
Freezing String Method to find transition states. This is
implemented along-side the bracket methods only due to
programmatic ease

References:
[1] A. Behn et al., J. Chem. Phys., 2011, 135, 224108 (original)
[2] S. Sharada et al., J. Chem. Theory Comput., 2012, 8, 5166-5174 (improved)
"""
from typing import Any, List, Tuple, Optional, Union, TYPE_CHECKING
import numpy as np

from autode.values import PotentialEnergy
from autode.neb import NEB
from autode.utils import ProcessPool
from autode.path.interpolation import CubicPathSpline
from autode.bracket.imagepair import EuclideanImagePair
from autode.bracket.base import BaseBracketMethod
from autode.opt.coordinates import CartesianCoordinates
from autode.opt.optimisers.rfo import RFOptimiser
from autode.opt.optimisers.hessian_update import BFGSSR1Update, BFGSUpdate
from autode.log import logger

if TYPE_CHECKING:
    from autode.species.species import Species


class TangentQNROptimiser(RFOptimiser):
    def __init__(
        self,
        maxiter: int,
        tangent: np.ndarray,
        gtol=1e-3,
        etol=1e-4,
        max_step: float = 0.06,
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
        self._hessian_update_types = [BFGSUpdate, BFGSSR1Update]

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
            if b[i] <= 1.0e-14:
                continue
                # todo check below formula
            delta_s -= f[i] * u[:, i] / (b[i] - shift_lmda)

        self._take_step_within_trust_radius(delta_s)


def _optimise_get_coords(species, tau, method, n_cores, maxiter):
    opt = TangentQNROptimiser(maxiter=maxiter, tangent=tau)
    opt.run(species, method, n_cores)
    return opt.final_coordinates, opt.iteration


def _parallel_optimise_tangent(
    new_nodes: tuple, species, taus: tuple, method, n_cores, maxiter: int
):
    # TODO: species list and tau list
    # todo check these formula
    mols = []
    for i, coords in enumerate(new_nodes):
        mol = species.new_species(name=species.name + f"{i}")
        mol.coordinates = coords
        mols.append(mol)

    n_procs = 2 if n_cores > 2 else 1
    n_cores_pp = max(int(n_cores // 2), 1)
    with ProcessPool(max_workers=n_procs) as pool:
        jobs = [
            pool.submit(
                _optimise_get_coords, mol, tau, method, n_cores_pp, maxiter
            )
            for mol, tau in zip(mols, taus)
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
        """
        The freezing string, contains the previous coordinates and the
        methods for growing the string inwards

        Args:
            left_image: The "reactant" species
            right_image: The "product" species
            step_size: Size of FSM interpolation step for growth
            maxiter_per_node: maximum optimiser steps for each new node
            use_idpp: whether to use IDPP interpolation or cartesian
        """
        super().__init__(left_image=left_image, right_image=right_image)

        self._step_size = step_size  # todo distance
        self._max_n = abs(int(maxiter_per_node))
        assert self._max_n > 0
        self._use_idpp = bool(use_idpp)

    @property
    def ts_guess(self) -> Optional["Species"]:
        """
        For FSM, the TS guess is the highest energy node

        Returns:
            (Species): The TS guess species
        """
        energies = [coords.e for coords in self._total_history]

        assert all(en is not None for en in energies), "Energy value missing"
        peak_idx = np.argmax(energies)
        if peak_idx == 0 or peak_idx == len(self._total_history):
            logger.warning("No peak found in FSM")
            return None

        tmp_spc = self._left_image.new_species(name="peak")
        peak_coords = self._total_history[peak_idx]
        tmp_spc.coordinates = peak_coords
        return tmp_spc

    @property
    def has_jumped_over_barrier(self) -> bool:
        """Barrier check is not needed for FSM"""
        return False

    def _add_coordinates(
        self, new_nodes: Tuple[CartesianCoordinates, CartesianCoordinates]
    ) -> None:
        """
        Add optimised nodes to the string, and remove any translation
        or rotation

        Args:
            new_nodes (tuple): The two new nodes in order (left and right)
        """
        left_new, right_new = new_nodes
        _align_coords_to_ref(left_new, self.left_coords)
        _align_coords_to_ref(right_new, self.right_coords)
        self.left_coords = left_new
        self.right_coords = right_new
        return None

    def grow_string(self):
        """
        Grow the string by adding two nodes: it performs Cartesian or
        IDPP interpolation, chooses guess coordinates and then optimises
        perpendicular to tangent.
        """
        if self._use_idpp:
            nodes, tangents = self._get_new_coords_tangents_idpp()
        else:
            nodes, tangents = self._get_new_coords_tangents_cartesian()

        result = _parallel_optimise_tangent(
            nodes,
            self._left_image.copy(),
            tangents,
            self._method,
            self._n_cores,
            self._max_n,
        )

        self._add_coordinates(result[0])
        return result[1]

    def _get_new_coords_tangents_cartesian(
        self,
    ) -> Tuple[
        Tuple[CartesianCoordinates, CartesianCoordinates],
        Tuple[np.ndarray, np.ndarray],
    ]:
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

    def _get_new_coords_tangents_idpp(
        self,
    ) -> Tuple[
        Tuple[CartesianCoordinates, CartesianCoordinates],
        Tuple[np.ndarray, np.ndarray],
    ]:
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
        coords_list: List[CartesianCoordinates] = []
        for point in idpp.images:
            coords = CartesianCoordinates(point.coordinates)
            if len(coords_list) > 0:
                _align_coords_to_ref(coords, coords_list[-1])
            coords_list.append(coords)

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
        tangents = (
            spline.tangent_at(spline.path_distances[left_idx]),
            spline.tangent_at(spline.path_distances[right_idx]),
        )
        return nodes, tangents


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
        # todo remove args, put maxiter here
        assert "cineb_at_conv" not in kwargs.keys()
        # todo warn if dist_tol set
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
        # self._dist_tol = step_size
        # todo step size as Distance

    def _log_convergence(self) -> None:
        # todo fix error with logging energy is None?
        return None

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
        return None

    def _step(self) -> None:
        iters = self.imgpair.grow_string()
        self._micro_iter += iters

        pass
