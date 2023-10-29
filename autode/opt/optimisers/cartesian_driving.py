"""
Cartesian coordinate driving of bonds
"""
import numpy as np

from autode.opt.coordinates.primitives import PrimitiveDistance
from autode.opt.coordinates import CartesianComponent


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
