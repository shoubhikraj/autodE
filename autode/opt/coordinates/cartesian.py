import numpy as np
from typing import Optional, TYPE_CHECKING

from autode.log import logger
from autode.values import ValueArray
from autode.opt.coordinates.base import OptCoordinates
from autode.opt.coordinates.dic import DIC

if TYPE_CHECKING:
    from autode.values import Gradient
    from autode.hessians import Hessian


class CartesianCoordinates(OptCoordinates):
    """Flat Cartesian coordinates shape = (3 × n_atoms, )"""

    def __repr__(self):
        return f"Cartesian Coordinates({np.ndarray.__str__(self)} {self.units.name})"

    def __new__(cls, input_array, units="Å") -> "CartesianCoordinates":
        """New instance of these coordinates"""
        return super().__new__(
            cls, np.array(input_array).flatten(), units=units
        )

    def __array_finalize__(self, obj) -> None:
        """See https://numpy.org/doc/stable/user/basics.subclassing.html"""
        return None if obj is None else super().__array_finalize__(obj)

    def _str_is_valid_unit(self, string) -> bool:
        """Is a string a valid unit for these coordinates e.g. nm"""
        return any(string in unit.aliases for unit in self.implemented_units)

    def _update_g_from_cart_g(self, arr: Optional["Gradient"]) -> None:
        """
        Updates the gradient from a calculated Cartesian gradient, which for
        Cartesian coordinates there is nothing to be done for.

        -----------------------------------------------------------------------
        Arguments:
            arr: Gradient array
        """
        self.g = None if arr is None else np.array(arr).flatten()

    def _update_h_from_cart_h(self, arr: Optional["Hessian"]) -> None:
        """
        Update the Hessian from a Cartesian Hessian matrix with shape
        3N x 3N for a species with N atoms.


        -----------------------------------------------------------------------
        Arguments:
            arr: Hessian matrix
        """
        self.h = None if arr is None else np.array(arr)

    def iadd(self, value: np.ndarray) -> OptCoordinates:
        return np.ndarray.__iadd__(self, value)

    def to(self, value: str) -> OptCoordinates:
        """
        Transform between cartesian and internal coordinates e.g. delocalised
        internal coordinates or other units

        -----------------------------------------------------------------------
        Arguments:
            value (str): Intended conversion

        Returns:
            (autode.opt.coordinates.OptCoordinates): Transformed coordinates

        Raises:
            (ValueError): If the conversion cannot be performed
        """
        logger.info(f"Transforming Cartesian coordinates to {value}")

        if value.lower() in ("cart", "cartesian", "cartesiancoordinates"):
            return self

        elif value.lower() in ("dic", "delocalised internal coordinates"):
            return DIC.from_cartesian(self)

        # ---------- Implement other internal transformations here -----------

        elif self._str_is_valid_unit(value):
            return CartesianCoordinates(
                ValueArray.to(self, units=value), units=value
            )
        else:
            raise ValueError(
                f"Cannot convert Cartesian coordinates to {value}"
            )

    @property
    def expected_number_of_dof(self) -> int:
        """Expected number of degrees of freedom for the system"""
        n_atoms = len(self.flatten()) // 3
        return 3 * n_atoms - 6


class CartTRCoordinates(CartesianCoordinates):
    """
    Cartesian coordinates with translation and rotation removed from
    the gradient and hessian.

    Reference: Page, McIver, J. Chem. Phys., 1988, 88(2), 922
    """

    def _get_tr_vectors(self) -> np.ndarray:
        """
        Obtain translation and rotation vectors which may or may not
        be orthogonal, and may contain linear dependencies.

        Returns:
            (np.ndarray): The translation rotation vectors
        """
        assert len(self) != 0 and len(self.shape) == 1
        assert len(self) % 3 == 0
        n_atoms = int(self.shape[0] / 3)

        # Translation vectors
        b_1 = np.tile([1.0, 0.0, 0.0], reps=n_atoms)
        b_2 = np.tile([0.0, 1.0, 0.0], reps=n_atoms)
        b_3 = np.tile([0.0, 0.0, 1.0], reps=n_atoms)
        # Rotation vectors
        b_4, b_5, b_6 = [np.zeros_like(b_1) for _ in range(3)]
        for idx in range(n_atoms):
            coord_x = self[3 * idx]
            coord_y = self[3 * idx + 1]
            coord_z = self[3 * idx + 2]
            b_4[3 * idx : 3 * idx + 3] = [0.0, coord_z, -coord_y]
            b_5[3 * idx : 3 * idx + 3] = [-coord_z, 0.0, coord_x]
            b_6[3 * idx : 3 * idx + 3] = [coord_y, -coord_x, 0.0]

        b = np.zeros(shape=(3 * n_atoms, 6))
        for i, arr in enumerate([b_1, b_2, b_3, b_4, b_5, b_6]):
            b[:, i] = arr

        return b

    def _calculate_projector(self):
        """
        Calculate the projector that will remove the translation and
        rotation from the gradient and Hessian

        Returns:
            (np.ndarray): The projector matrix
        """
        b = self._get_tr_vectors()

        # get overlap matrix, and check if it is singular
        s = np.matmul(b.T, b)
        if np.linalg.matrix_rank(s) < 6:
            # todo change this to handle linear cases
            raise Exception("Matrix singular!!")
        s_inv = np.linalg.inv(s)
        r = np.linalg.multi_dot([b, s_inv, b.T])
        assert r.shape[0] == r.shape[1]
        return np.eye(r.shape[0]) - r

    def _update_g_from_cart_g(self, arr: Optional["Gradient"]) -> None:
        """
        Updates the gradient from Cartesian gradient, removing the components
        along rotational or translational modes

        Args:
            arr: Gradient array
        """
        if arr is None:
            self.g = None
            return None

        p = self._calculate_projector()
        # todo check formula frank jensen
        self.g = np.matmul(p, arr)

    def _update_h_from_cart_h(self, arr: Optional["Hessian"]) -> None:
        """
        Updates the hessian from Cartesian Hessian, removing the components
        along rotational and translational modes

        Args:
            arr: Hessian array
        """
        # NOTE: Unless at a minima, rot. modes cannot be formally
        # projected out as there is some coupling to vib. modes
        if arr is None:
            self.h = None
            return None

        p = self._calculate_projector()
        # todo check formula with frank jensen
        self.h = np.multi_dot([p, arr, p.T])


def _stabilised_gram_schmidt_orthonormalise(matrix):
    """
    Orthogonalise a set of vectors that are the columns of
    the matrix provided, and return a matrix of the same
    shape with orthonormal columns

    Args:
        matrix:

    Returns:
        (np.ndarray):
    """

    def proj_u_v(u, v):
        """projection of v on u"""
        return u * np.dot(u, v) / np.dot(u, u)

    def is_zero_vec(v):
        """is this a zero vector"""
        return np.allclose(v, np.zeros_like(v))

    vecs = [np.array(matrix[:, i]).flatten() for i in range(matrix.shape[1])]
    assert not any(is_zero_vec(v) for v in vecs)

    for i in range(1, len(vecs)):
        try:
            pass
        except IndexError:
            pass
