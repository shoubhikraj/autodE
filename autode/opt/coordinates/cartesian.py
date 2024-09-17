import numpy as np
import scipy
from typing import Optional, List, TYPE_CHECKING

from autode.log import logger
from autode.values import ValueArray
from autode.opt.coordinates.base import OptCoordinates
from autode.opt.coordinates.dic import DIC

if TYPE_CHECKING:
    from autode.values import Gradient
    from autode.hessians import Hessian


TR_SHIFT = 100.0  # shift factor for trans. rot. modes


class CartesianCoordinates(OptCoordinates):
    """Flat Cartesian coordinates shape = (3 × n_atoms, )"""

    def __repr__(self):
        return f"Cartesian Coordinates({np.ndarray.__str__(self)} {self.units.name})"

    def __new__(cls, input_array, units="Å") -> "CartesianCoordinates":
        """New instance of these coordinates"""

        # if it has units cast into current units
        if isinstance(input_array, ValueArray):
            input_array = ValueArray.to(input_array, units=units)

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
        self._g = None if arr is None else np.array(arr).flatten()

    def _update_h_from_cart_h(self, arr: Optional["Hessian"]) -> None:
        """
        Update the Hessian from a Cartesian Hessian matrix with shape
        3N x 3N for a species with N atoms.


        -----------------------------------------------------------------------
        Arguments:
            arr: Hessian matrix
        """
        assert self.h_or_h_inv_has_correct_shape(arr)
        self._h = None if arr is None else np.array(arr)

    @property
    def n_constraints(self) -> int:
        return 0

    @property
    def n_satisfied_constraints(self) -> int:
        return 0

    @property
    def active_indexes(self) -> List[int]:
        return list(range(len(self)))

    @property
    def inactive_indexes(self) -> List[int]:
        return []

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
    def cart_proj_g(self) -> Optional[np.ndarray]:
        return self.g

    @property
    def expected_number_of_dof(self) -> int:
        """Expected number of degrees of freedom for the system"""
        n_atoms = len(self.flatten()) // 3
        return 3 * n_atoms - 6


class CartesianTRCoordinates(CartesianCoordinates):
    """
    Cartesian coordinates that removes the translational and
    rotational components from the gradient and Hessian
    """

    @property
    def tr_vecs(self):
        """
        Obtain the translation and rotational vectors in orthonormal
        format - returns 6 vectors for non-linear and 5 for linear
        molecules

        Returns:

        """
        # Ref.: Page, McIver, J. Chem. Phys., 1988, 88(2), 922
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

        # get orthonormal basis from SVD, removes one mode if linear
        v = scipy.linalg.orth(b, rcond=1.0e-5)
        assert v.shape[1] in (5, 6)
        return v

    @property
    def h(self):
        """
        Return the Hessian matrix with trans. and rot. modes frozen
        """
        if self._h is None:
            return None

        arr = self._h.copy()
        tr_vecs = self.tr_vecs
        for idx in range(tr_vecs.shape[1]):
            vec = tr_vecs[:, idx].flatten()
            arr += TR_SHIFT * np.outer(vec, vec)

        return arr

    @h.setter
    def h(self, value):
        """Set the value of h"""
        raise NotImplementedError

    @property
    def g(self):
        """
        Obtain the gradient with trans. and rot. directions zeroed
        """
        if self._g is None:
            return None

        arr = self._g.copy()
        # obtain the projection matrix
        tr_vecs = self.tr_vecs
        r_mat = np.matmul(tr_vecs, tr_vecs.T)
        assert r_mat.shape[0] == r_mat.shape[1]
        proj = np.eye(r_mat.shape[0]) - r_mat

        arr = arr.ravel().reshape(-1, 1)
        return np.matmul(proj, arr).flatten()

    @g.setter
    def g(self, value):
        raise NotImplementedError
