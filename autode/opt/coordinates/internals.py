"""
Internal coordinates. Notation follows:


x : Cartesian coordinates
B : Wilson B matrix
q : Primitive internal coordinates
G : Spectroscopic G matrix
"""
import itertools
import numpy as np

from typing import Any, Optional, Tuple, Type, List, TYPE_CHECKING
from abc import ABC, abstractmethod
from autode.values import Angle, Distance
from autode.mol_graphs import make_graph
from autode.constraints import DistanceConstraints
from autode.opt.coordinates.base import OptCoordinates, CartesianComponent
from autode.opt.coordinates.primitives import (
    PrimitiveInverseDistance,
    Primitive,
    PrimitiveDistance,
    ConstrainedPrimitiveDistance,
    PrimitiveBondAngle,
    PrimitiveDihedralAngle,
)

if TYPE_CHECKING:
    from autode.species import Species
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
            if isinstance(primitive, PrimitiveDihedralAngle):

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


class PrimitiveInverseDistances(_FunctionOfDistances):
    """1 / r_ij for all unique pairs i,j. Will be redundant"""

    @property
    def _primitive_type(self):
        return PrimitiveInverseDistance


class PrimitiveDistances(_FunctionOfDistances):
    """r_ij for all unique pairs i,j. Will be redundant"""

    @property
    def _primitive_type(self):
        return PrimitiveDistance


class AnyPIC(PIC):
    def _populate_all(self, x: np.ndarray) -> None:
        raise RuntimeError("Cannot populate all on an AnyPIC instance")


def build_redundant_pic_from_species(species):
    species = species.copy()
    n_dof = 3 * species.n_atoms - (5 if species.is_linear() else 6)

    if species.graph is not None:
        make_graph(species, allow_invalid_valancies=True)

    pic, dof = get_pic_and_dof_from_species(species)
    if dof > n_dof:
        return pic

    # removing bonds due to valency consideration may have cause problems
    make_graph(species, allow_invalid_valancies=True)
    pic, dof = get_pic_and_dof_from_species(species, True, True)
    if dof > n_dof:
        return pic
    else:
        raise RuntimeError("Failed to build redundant internal coordinates")


def get_pic_and_dof_from_species(
    species: "Species",
    aux_bonds: bool = False,
    aux_interfrag: bool = False,
) -> Tuple[AnyPIC, float]:
    # take a copy so that modifications do not affect original
    species = species.copy()
    assert species.graph is not None
    pic = AnyPIC()
    _handle_fragments_constraints(pic, species, aux_interfrag=aux_interfrag)
    _add_distances_from_species(pic, species, aux_bonds=aux_bonds)
    _add_bends_from_species(pic, species)
    _add_dihedrals_from_species(pic, species)
    x = CartesianCoordinates(species.coordinates)
    _ = pic(x)
    dof = np.linalg.matrix_rank(pic.B, tol=1e-6)
    return pic, dof


def _handle_fragments_constraints(
    pic, species: "Species", aux_interfrag=False
) -> None:
    """
    Update the "core" graph of the molecule, by adding the distance
    constraints into the graph, and by joining the fragments.
    Also adds auxiliary interfragment bonds, which are not part
    of the "core" graph. The AnyPIC instance and the species are
    modified in-place.

    Args:
        species (Species):
    """
    # todo join fragments and constraints
    assert species.graph is not None
    frags = list(species.graph.connected_fragments())
    # add constraints into core graph
    constraints = species.constraints.distance
    if constraints is None:
        constraints = DistanceConstraints()
    for (i, j) in constraints:
        species.graph.add_edge(i, j, pi=False, active=False)

    # handle fragments
    if len(frags) == 1:
        return None
    for (frag1, frag2) in itertools.combinations(frags, r=2):
        distances = []
        atom_pairs = []
        for (i, j) in itertools.product(frag1, frag2):
            atom_pairs.append((i, j))
            distances.append(species.distance(i, j))
        # min interfragment distances is a "core" bond
        min_pair = atom_pairs[np.argmin(distances)]
        species.graph.add_edge(*min_pair, pi=False, active=False)
        # add auxiliary interfragment bonds
        if aux_interfrag:
            min_dist = min(distances)
            for idx, (i, j) in enumerate(atom_pairs):
                if (
                    distances[idx] < 1.3 * min_dist
                    or distances[idx] < Distance(2.0, "ang")
                ) and not species.graph.has_edge(i, j):
                    pic.append(PrimitiveDistance(i, j))
    return None


def _add_distances_from_species(
    pic: "AnyPIC", species: "Species", aux_bonds: bool = True
) -> None:
    """
    Add distances from a species using the graph of core bonds
    provided. Optionally, add auxiliary extra-redundant bonds.
    The AnyPIC instance is modified in-place.

    Args:
        pic (AnyPIC): An AnyPIC instance
        species (Species): The species
        aux_bonds (bool): Whether to add auxiliary bonds
    """
    # now we iterate over the "core" bonds
    core_graph = species.graph
    assert core_graph is not None
    constraints = species.constraints.distance
    if constraints is None:
        constraints = DistanceConstraints()

    for (i, j) in sorted(core_graph.edges):
        if (i, j) in constraints:
            r = constraints[(i, j)]
            pic.append(ConstrainedPrimitiveDistance(i, j, r))
        else:
            pic.append(PrimitiveDistance(i, j))
    assert len(constraints) == pic.n_constrained

    if not aux_bonds:
        return None

    # add auxiliary "extra-redundant" bonds
    for (i, j) in itertools.combinations(range(species.n_atoms), r=2):
        if core_graph.has_edge(i, j):
            continue
        # avoid adding if already covered by auxiliary interfragment
        if PrimitiveDistance(i, j) in pic:
            continue
        if species.distance(i, j) < 2.5 * species.eqm_bond_distance(i, j):
            pic.append(PrimitiveDistance(i, j))
    return None


def _add_bends_from_species(pic, species) -> None:
    """
    Add all bond angle (i.e. valence angles) from a species and
    its "core" graph. The AnyPIC instance is modified in-place.

    Args:
        pic (AnyPIC): An AnyPIC instance
        species (Species): The species
    """
    core_graph = species.graph
    for o in range(species.n_atoms):
        for (n, m) in itertools.combinations(core_graph.neighbors(o), r=2):
            pic.append(PrimitiveBondAngle(o=o, m=m, n=n))
            # TODO: implement linear angles
    return None


def _add_dihedrals_from_species(pic, species) -> None:
    """
    Add all dihedral angles (i.e. torsions) from a species and
    its graph, avoiding cases where any one of the adjacent pair of
    bonds having an angle close to 180 degrees. The AnyPIC instance
    is modified in-place.

    Args:
        pic (AnyPIC): An AnyPIC instance
        species (Species): The species
    """
    core_graph = species.graph
    # no dihedral possible with less than 4 atoms
    if species.n_atoms < 4:
        return None

    for (o, p) in core_graph.edges:
        for m in core_graph.neighbors(o):
            if m == p:
                continue

            for n in core_graph.neighbors(p):
                if n == o:
                    continue

                # avoid triangle rings like cyclopropane
                if n == m:
                    continue

                is_linear_1 = species.angle(m, o, p) > Angle(175, "deg")
                is_linear_2 = species.angle(o, p, n) > Angle(175, "deg")

                if is_linear_2 or is_linear_1:
                    # TODO: implement robust dihedrals
                    continue
                else:
                    pic.append(PrimitiveDihedralAngle(m, o, p, n))

    return None
