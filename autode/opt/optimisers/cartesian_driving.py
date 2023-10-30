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
        assert all(isinstance(coeff, float) for coeff in coefficients)
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

    def _initialise_run(self) -> None:
        """Initialise the run by generating self._coords"""
        assert self._species is not None

        self._coords = CartesianCoordinates(self._species.coordinates)
        self._coords.update_h_from_cart_h(self._low_level_cart_hessian)
        self._update_gradient_and_energy()

        current_dist = self._driven_coords(self._coords)
        self._target_dist = current_dist - self._drive_step

    def _get_lagrangian_gradient(self):
        A = self._driven_coords.derivative(self._coords)
        g_con = self._coords.g - self._lambda * A
        c_x = self._driven_coords(self._coords) - self._target_dist
        return np.append(g_con, [-c_x])

    def _get_lagrangian_hessian(self):
        W = self._coords.h - self._lambda * (
            self._driven_coords.second_derivative(self._coords)
        )
        A = self._driven_coords.derivative(self._coords)
        h_n = self._coords.h.shape[0]
        del2_L = np.zeros(h_n + 1, h_n + 1)
        del2_L[:h_n, :h_n] = W
        del2_L[-1, :h_n] = -A
        del2_L[:h_n, -1] = -A

        return del2_L
