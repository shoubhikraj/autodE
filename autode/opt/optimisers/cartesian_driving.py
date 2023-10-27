from autode.opt.coordinates.primitives import PrimitiveDistance


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
        pass
