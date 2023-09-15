"""
Freezing String Method to find transition states. This is
implemented along-side the bracket methods only due to
programmatic ease

References:
[1] A. Behn et al., J. Chem. Phys., 2011, 135, 224108 (original)
[2] S. Sharada et al., J. Chem. Theory Comput., 2012, 8, 5166-5174 (improved)
"""
from typing import Any, TYPE_CHECKING
import numpy as np

from autode.neb import NEB
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
        gtol,
        etol,
        tangent: np.ndarray,
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

    def _remove_tangent_from_hessian(self) -> None:
        # Frank Jensen, Introduction to Computational Chemistry, 565
        # this is probably not the full transform, but that is ok as
        # an approximate hessian is sufficient
        assert self._coords is not None
        assert self._coords.h is not None
        x_k = self._tau_hat.reshape(-1, 1)
        q_k = np.matmul(x_k, x_k.T)
        p_k = np.ones_like(q_k) - q_k
        self._coords.h = np.linalg.multi_dot([p_k.T, self._coords.h, p_k])
        return None


class FSMPath(EuclideanImagePair):
    def __init__(
        self,
        left_image: "Species",
        right_image: "Species",
        maxiter_per_node: int,
    ):
        super().__init__(left_image=left_image, right_image=right_image)

        self._max_n = abs(int(maxiter_per_node))
        assert self._max_n > 0

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
