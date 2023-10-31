"""
Cartesian coordinate driving of bonds
"""
import numpy as np

from autode.opt.coordinates.primitives import PrimitiveDistance
from autode.opt.coordinates import CartesianComponent, CartesianCoordinates
from autode.opt.optimisers import RFOptimiser


class DrivenDistances:
    """
    Linear combination of distances that are being driven
    """

    def __init__(self, bonds, coefficients):
        self._check_bonds(bonds)
        assert all(isinstance(coeff, (float, int)) for coeff in coefficients)
        self._coeffs = list(coefficients)
        self._prims = []
        for bond in bonds:
            self._prims.append(PrimitiveDistance(*bond))

    def _check_bonds(self, bonds):
        pass

    def _check_ill_conditioned(self):
        """Is B-matrix linearly dependent"""

    def __call__(self, x):
        """Calculate the value of the combined coordinate"""
        val = 0.0
        for bond, coeff in zip(self._prims, self._coeffs):
            val += coeff * bond(x)
        return val

    def derivative(self, x):
        """Calculate the first derivative against Cartesian coordinates"""
        B = np.zeros_like(x)
        _x = x.reshape(-1, 3)
        n_atoms = _x.shape[0]
        for bond, coeff in zip(self._prims, self._coeffs):
            for j in range(n_atoms):
                B[3 * j + 0] += coeff * bond.derivative(
                    j, CartesianComponent.x, _x
                )
                B[3 * j + 1] += coeff * bond.derivative(
                    j, CartesianComponent.y, _x
                )
                B[3 * j + 2] += coeff * bond.derivative(
                    j, CartesianComponent.y, _x
                )

        return B

    def second_derivative(self, x):
        """Calculate the second derivative against Cartesian coordinates"""
        n = x.flatten().shape[0]
        C = np.zeros(shape=(n, n))

        _x = x.reshape(-1, 3)
        n_atoms = _x.shape[0]

        comps = [
            CartesianComponent.x,
            CartesianComponent.y,
            CartesianComponent.z,
        ]

        for bond, coeff in zip(self._prims, self._coeffs):
            for i in range(n_atoms):
                for j in range(n_atoms):
                    for comp_i in comps:
                        for comp_j in comps:
                            C[3 * i + comp_i, 3 * j + comp_j] = coeff * (
                                bond.second_derivative(
                                    i, comp_i, j, comp_j, _x
                                )
                            )
        return C


class CartesianDrivingOptimiser(RFOptimiser):
    def __init__(
        self,
        maxiter,
        gtol,
        etol,
        driving_coords: DrivenDistances,
        drive_step=0.2,
        trust_radius=0.08,
    ):
        super().__init__(maxiter, gtol, etol, init_alpha=trust_radius)

        self._drive_step = drive_step
        self._driven_coords = driving_coords
        self._lambda = 0.0  # Lagrangian multiplier
        self._target_dist = None

    @property
    def _g_norm(self):
        return np.sqrt(np.average(np.square(self._get_constrained_gradient())))

    def _initialise_run(self) -> None:
        """Initialise the run by generating self._coords"""
        assert self._species is not None

        self._coords = CartesianCoordinates(self._species.coordinates)
        self._coords.update_h_from_cart_h(self._low_level_cart_hessian)
        self._update_gradient_and_energy()

        current_dist = self._driven_coords(self._coords)
        self._target_dist = current_dist + self._drive_step

    def _get_constrained_gradient(self):
        A = self._driven_coords.derivative(self._coords)
        g_con = self._coords.g - self._lambda * A
        return g_con

    def _get_lagrangian_gradient(self):
        g_con = self._get_constrained_gradient()
        c_x = self._driven_coords(self._coords) - self._target_dist
        return np.append(g_con, [-c_x])

    def _get_lagrangian_hessian(self):
        W = self._coords.h - self._lambda * (
            self._driven_coords.second_derivative(self._coords)
        )
        A = self._driven_coords.derivative(self._coords)
        h_n = self._coords.h.shape[0]
        del2_L = np.zeros(shape=(h_n + 1, h_n + 1))
        del2_L[:h_n, :h_n] = W
        del2_L[-1, :h_n] = -A
        del2_L[:h_n, -1] = -A

        return del2_L

    def _step(self) -> None:
        """Take an RFO step"""
        assert self._coords is not None
        self._coords.h = self._updated_h()

        # get delta2 L and symmetrize
        h = self._get_lagrangian_hessian()
        h = (h + h.T) / 2
        b, u = np.linalg.eigh(h)
        f = u.T.dot(self._get_lagrangian_gradient())

        # choose the constraint mode from the lagrangian eigenvalues
        constr_idx = np.argmax(u[-1])

        # build the paritioned rfo step
        h_n, _ = h.shape
        step = np.zeros(shape=h_n)

        min_b = np.delete(b, constr_idx)
        min_f = np.delete(f, constr_idx)
        min_u = np.delete(u, constr_idx, axis=1)
        min_aug_h = np.zeros((h_n, h_n))
        min_aug_h[: h_n - 1, : h_n - 1] = np.diag(min_b)
        min_aug_h[-1, : h_n - 1] = min_f
        min_aug_h[: h_n - 1, 1] = min_f
        lmda_ns = np.linalg.eigvalsh(min_aug_h)
        lmda_n = lmda_ns[np.where(np.abs(lmda_ns) > 1e-15)[0][0]]

        for i in range(len(min_f)):
            step += min_f[i] * min_u[:, i] / (min_b[i] - lmda_n)

        max_b = b[constr_idx]
        max_f = b[constr_idx]
        max_u = u[:, constr_idx]
        max_aug_h = np.zeros((2, 2))
        max_aug_h[0, 0] = max_b
        max_aug_h[-1, 0] = max_f
        max_aug_h[0, -1] = max_f
        lmda_p = np.linalg.eigvalsh(max_aug_h)[-1]
        step += max_f * max_u / (max_b - lmda_p)

        # take step within trust radius
        max_step = np.max(step[:-1])
        if max_step > self.alpha:
            step = step * self.alpha / max_step

        self._coords = self._coords + step[:-1]
        self._lambda = step[-1]
        return None
