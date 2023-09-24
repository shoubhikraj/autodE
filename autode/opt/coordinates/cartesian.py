import numpy as np
import scipy
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
        arr = super().__new__(
            cls, np.array(input_array).flatten(), units=units
        )

        arr.remove_tr = False  # whether to remove trans. and rot. dof
        return arr

    def __array_finalize__(self, obj) -> None:
        """See https://numpy.org/doc/stable/user/basics.subclassing.html"""
        super().__array_finalize__(obj)

        for attr in ["remove_tr"]:
            setattr(self, attr, getattr(obj, attr, None))
        return

    def _str_is_valid_unit(self, string) -> bool:
        """Is a string a valid unit for these coordinates e.g. nm"""
        return any(string in unit.aliases for unit in self.implemented_units)

    def _update_g_from_cart_g(self, arr: Optional["Gradient"]) -> None:
        """
        Updates the gradient from a calculated Cartesian gradient, optionally
        removing the rotation and translational components

        -----------------------------------------------------------------------
        Arguments:
            arr: Gradient array
        """
        if arr is None:
            self.g = None
            return

        arr = np.array(arr).flatten()

        if self.remove_tr:
            arr = arr.reshape(-1, 1)  # type: ignore
            p = self._calculate_projector()
            self.g = np.matmul(p, arr).flatten()
        else:
            self.g = arr

    def _update_h_from_cart_h(self, arr: Optional["Hessian"]) -> None:
        """
        Update the Hessian from a Cartesian Hessian matrix with shape
        3N x 3N for a species with N atoms. Optionally project out the
        rotational and vibrational degrees of freedom

        -----------------------------------------------------------------------
        Arguments:
            arr: Hessian matrix
        """
        if arr is None:
            self.h = None
            return

        if self.remove_tr:
            p = self._calculate_projector()
            self.h = np.linalg.multi_dot([p.T, arr, p])
        else:
            self.h = np.array(arr)

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

    def _get_tr_vecs(self) -> np.ndarray:
        """
        Obtain translation and rotation vectors and then orthonormalise
        them, removing linear dependencies for linear molecules

        Returns:
            (np.ndarray): The orthonormal trans. and rot. vectors
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

        # get orthonormal basis from SVD, removes one mode if linear
        v = scipy.linalg.orth(b, rcond=1.0e-5)
        assert v.shape[1] in (5, 6)
        return v

    def _calculate_projector(self):
        """
        Calculate the projector that will remove the translation and
        rotation from the gradient and Hessian

        Returns:
            (np.ndarray): The projector matrix
        """
        v = self._get_tr_vecs()
        r = np.matmul(v, v.T)

        assert r.shape[0] == r.shape[1]
        return np.eye(r.shape[0]) - r

    @property
    def expected_number_of_dof(self) -> int:
        """Expected number of degrees of freedom for the system"""
        # todo check this
        n_tr = self._get_tr_vecs().shape[1]
        n_atoms = len(self.flatten()) // 3
        return 3 * n_atoms - n_tr
