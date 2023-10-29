"""
Cartesian coordinate driving of bonds
"""
import numpy as np

from autode.opt.coordinates.primitives import PrimitiveDistance
from autode.opt.coordinates import CartesianComponent
from autode.opt.optimisers import RFOptimiser


class DrivenDistances:
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
        val = 0.0
        for bond, coeff in zip(self._prims, self._coeffs):
            val += coeff * bond(x)
        return val

    def derivative(self, x):
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
        n = x.flatten().shape[0]
        C = np.zeros(shape=(n, n))

        _x = x.reshape(-1, 3)
        n_atoms = _x.shape[0]

        for bond, coeff in zip(self._prims, self._coeffs):
            for i in range(n_atoms):
                for j in range(n_atoms):
                    for comp_i in [0, 1, 2]:
                        for comp_j in [0, 1, 2]:
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
        driving_coords,
        drive_step=0.2,
        trust_radius=0.08,
    ):
        super().__init__(maxiter, gtol, etol, init_alpha=trust_radius)

        self._drive_step = drive_step
        self._lambda = 0.0  # Lagrangian multiplier
        self._target_dist = None

    def _get_lagrangian_gradient(self):
        pass

    def _get_lagrangian_hessian(self):
        pass
