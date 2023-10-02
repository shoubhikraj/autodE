"""
Internal coordinates. Notation follows:


x : Cartesian coordinates
B : Wilson B matrix
q : Primitive internal coordinates
G : Spectroscopic G matrix
"""
import itertools
import numpy as np

from typing import Any, Optional, Type, List, TYPE_CHECKING
from abc import ABC, abstractmethod
from autode.values import Angle
from autode.opt.coordinates.base import OptCoordinates, CartesianComponent
from autode.opt.coordinates.primitives import (
    InverseDistance,
    Primitive,
    Distance,
    ConstrainedDistance,
    BondAngle,
    DihedralAngle,
)

if TYPE_CHECKING:
    from autode.opt.coordinates.cartesian import CartesianCoordinates
    from autode.opt.coordinates.primitives import (
        ConstrainedPrimitive,
        _DistanceFunction,
    )


class InternalCoordinates(OptCoordinates, ABC):  # lgtm [py/missing-equals]
    def __new__(cls, input_array) -> "InternalCoordinates":
        """New instance of these internal coordinates"""

        arr = super().__new__(cls, input_array, units="Å")

        for attr in ("_x", "primitives"):
            setattr(arr, attr, getattr(input_array, attr, None))

        return arr

    def __array_finalize__(self, obj: "OptCoordinates") -> None:
        """See https://numpy.org/doc/stable/user/basics.subclassing.html"""
        OptCoordinates.__array_finalize__(self, obj)

        for attr in ("_x", "primitives"):
            setattr(self, attr, getattr(obj, attr, None))

        return

    @property
    def n_constraints(self) -> int:
        """Number of constraints in these coordinates"""
        return self.primitives.n_constrained

    @property
    def constrained_primitives(self) -> List["ConstrainedPrimitive"]:
        return [p for p in self.primitives if p.is_constrained]

    @property
    def n_satisfied_constraints(self) -> int:
        """Number of constraints that are satisfied in these coordinates"""
        x = self.to("cartesian")
        return sum(p.is_satisfied(x) for p in self.constrained_primitives)


class PIC(list, ABC):
    """Primitive internal coordinates"""

    def __init__(self, *args: Any):
        """
        List of primitive internal coordinates with a Wilson B matrix.
        If there are no arguments then all possible primitive coordinates
        will be generated
        """
        super().__init__(args)

        self._B: Optional[np.ndarray] = None

        if not self._are_all_primitive_coordinates(args):
            raise ValueError(
                "Cannot construct primitive internal coordinates "
                f"from {args}. Must be primitive internals"
            )

    @property
    def B(self) -> np.ndarray:
        """Wilson B matrix"""

        if self._B is None:
            raise AttributeError(
                f"{self} had no B matrix. Please calculate "
                f"the value of the primitives to determine B"
            )

        return self._B

    @property
    def G(self) -> np.ndarray:
        """Spectroscopic G matrix as the symmetrised Wilson B matrix"""
        return np.dot(self.B, self.B.T)

    @classmethod
    def from_cartesian(
        cls,
        x: "CartesianCoordinates",
    ) -> "PIC":
        """Construct a complete set of primitive internal coordinates from
        a set of Cartesian coordinates"""

        pic = cls()
        pic._populate_all(x=x)

        return pic

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Populate Primitive-s used in the construction of set"""

        q = self._calc_q(x)
        self._calc_B(x)

        return q

    def close_to(self, x: np.ndarray, other: np.ndarray) -> np.ndarray:
        """
        Calculate a set of primitive internal coordinates (PIC) that are
        'close to' another set. This means that the restriction on dihedral
        angles being in the range (-π, π] is relaxed in favour of the smallest
        ∆q possible (where q is a value of a primitive coordinate).
        """
        assert len(self) == len(other) and isinstance(other, np.ndarray)

        q = self._calc_q(x)
        self._calc_B(x)

        for i, primitive in enumerate(self):
            if isinstance(primitive, DihedralAngle):

                dq = q[i] - other[i]

                if np.abs(dq) > np.pi:  # Ensure |dq| < π
                    q[i] -= np.sign(dq) * 2 * np.pi

        return q

    def __eq__(self, other: Any):
        """Comparison of two PIC sets"""

        is_equal = (
            isinstance(other, PIC)
            and len(other) == len(self)
            and all(p0 == p1 for p0, p1 in zip(self, other))
        )

        return is_equal

    def _calc_q(self, x: np.ndarray) -> np.ndarray:
        """Calculate the value of the internals"""

        if len(self) == 0:
            self._populate_all(x)

        return np.array([q(x) for q in self])

    @abstractmethod
    def _populate_all(self, x: np.ndarray) -> None:
        """Populate primitives from an array of cartesian coordinates"""

    def _calc_B(self, x: np.ndarray) -> None:
        """Calculate the Wilson B matrix"""

        if len(self) == 0:
            raise ValueError(
                "Cannot calculate the Wilson B matrix, no "
                "primitive internal coordinates"
            )

        cart_coords = x.reshape((-1, 3))

        n_atoms, _ = cart_coords.shape
        B = np.zeros(shape=(len(self), 3 * n_atoms))

        for i, primitive in enumerate(self):
            for j in range(n_atoms):

                B[i, 3 * j + 0] = primitive.derivative(
                    j, CartesianComponent.x, x=cart_coords
                )
                B[i, 3 * j + 1] = primitive.derivative(
                    j, CartesianComponent.y, x=cart_coords
                )
                B[i, 3 * j + 2] = primitive.derivative(
                    j, CartesianComponent.z, x=cart_coords
                )

        self._B = B
        return None

    @staticmethod
    def _are_all_primitive_coordinates(args: tuple) -> bool:
        return all(isinstance(arg, Primitive) for arg in args)

    @property
    def n_constrained(self) -> int:
        """Number of constrained primitive internal coordinates"""
        return sum(p.is_constrained for p in self)


class _FunctionOfDistances(PIC):
    @property
    @abstractmethod
    def _primitive_type(self) -> Type["_DistanceFunction"]:
        """Type of primitive coordinate defining f(r_ij)"""

    def _populate_all(self, x: np.ndarray):

        n_atoms = len(x.flatten()) // 3

        # Add all the unique inverse distances (i < j)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                self.append(self._primitive_type(i, j))

        return None


class InverseDistances(_FunctionOfDistances):
    """1 / r_ij for all unique pairs i,j. Will be redundant"""

    @property
    def _primitive_type(self):
        return InverseDistance


class Distances(_FunctionOfDistances):
    """r_ij for all unique pairs i,j. Will be redundant"""

    @property
    def _primitive_type(self):
        return Distance


class AnyPIC(PIC):
    def _populate_all(self, x: np.ndarray) -> None:
        raise RuntimeError("Cannot populate all on an AnyPIC instance")


def build_redundant_pic(species):
    # start with the original graph
    graph = species.graph.copy()
    # todo add constraints to graph here

    pass


def build_pic_from_graph(
    species, graph, aux_bonds=False, linear_bends=False, robust_dihedrals=False
):
    # first put the bonds into the list
    pic = AnyPIC()

    # add the constraints to the list
    if species.constraints.distance is not None:
        for (i, j) in species.constraints.distance:
            graph.add_edge(i, j)

    pass


def _add_distances_from_species(pic, species, core_graph):
    n = 0
    for (i, j) in sorted(core_graph.edges):
        if (
            species.constraints.distance is not None
            and (i, j) in species.constraints.distance
        ):
            r = species.constraints.distance[(i, j)]
            pic.append(ConstrainedDistance(i, j, r))
            n += 1
        else:
            pic.append(Distance(i, j))
    assert n == species.constraints.n_distance


def _add_bends_from_species(pic, species, core_graph, linear_bends=False):
    for o in range(species.n_atoms):
        for (n, m) in itertools.combinations(core_graph.neighbors(o), r=2):
            if species.angle(m, o, n) < Angle(175, "deg"):
                pic.append(BondAngle(o=o, m=m, n=n))
            elif linear_bends:
                pass  # todo linear bends


def _add_dihedrals_from_species(
    pic, species, core_graph, robust_dihedrals=False
):
    # no dihedrals possible with less than 4 atoms
    if species.n_atoms < 4:
        return

    for (o, p) in core_graph.edges:
        for m in species.graph.neighbors(o):
            if m == p:
                continue

            if species.angle(m, o, p) > Angle(175, "deg"):
                continue

            for n in species.graph.neighbors(p):
                if n == o:
                    continue

                is_linear_1 = species.angle(m, o, p) > Angle(175, "deg")
                is_linear_2 = species.angle(o, p, n) > Angle(175, "deg")

                # don't add when both angles are linear
                if is_linear_1 and is_linear_2:
                    continue

                # if only one angle almost linear, add robust dihedral
                if (is_linear_1 or is_linear_2) and robust_dihedrals:
                    pass  # todo robust dihedrals
                else:
                    pic.append(DihedralAngle(m, o, p, n))


def minimise_primitive_lstsq(
    current_x: "CartesianCoordinates",
    pic: PIC,
    target_q: np.ndarray,
    q_weights: np.ndarray,
):
    """
    Impose a desired primitive internal coordinate on a Cartesian
    coordinate by performing weighted least squares minimisation

    Args:
        current_x (CartesianCoordinates): Current set of Cartesian coords
        pic (PIC): Primitive internal coordinates set
        target_q (np.ndarray) : The target value of primitives
        q_weights (np.ndarray): Least square weights for each primitive

    Returns:
        (CartesianCoordinates): The Cartesian coordinate that is closest to
                                the desired value of primitives
    """
    target_q = target_q.flatten()
    q_weights = q_weights.flatten()
    assert isinstance(pic, PIC) and isinstance(target_q, np.ndarray)
    assert len(target_q) == len(q_weights)

    def squared_error_and_deriv(x):
        q = pic.close_to(x, target_q)
        # todo raise exception if angle becomes 180 degrees?
        assert len(q) == len(target_q)
        wt_err = np.sum(np.square(q - target_q) * q_weights)
        # todo check this formula
        wt_err_der = np.sum((pic.B * q - pic.B * target_q) * q_weights * 2)
        return wt_err, wt_err_der

    from scipy.optimize import minimize

    res = minimize(
        squared_error_and_deriv,
        x0=current_x,
        jac=True,
        method="CG",
        options={"maxiter": 7000, "gtol": 1e-8},
    )
    assert res.success

    return CartesianCoordinates(res.x)
