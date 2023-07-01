import itertools
from copy import deepcopy
import numpy as np
from typing import Optional, Union, List, Sequence, Tuple, TYPE_CHECKING
from autode.atoms import Atom, Atoms
from itertools import product as iterprod
from scipy.spatial import distance_matrix
from autode.log import logger
from autode.geom import (
    get_points_on_sphere,
    calc_rmsd_on_atom_indices,
    align_species,
    calc_rmsd,
)
from autode.solvent.solvents import get_solvent
from autode.mol_graphs import union
from autode.species.species import Species
from autode.utils import requires_atoms, work_in
from autode.config import Config
from autode.methods import get_lmethod
from autode.conformers import Conformer
from autode.exceptions import MethodUnavailable

if TYPE_CHECKING:
    from autode.bond_rearrangement import BondRearrangement


def get_complex_conformer_atoms(molecules, rotations, points):
    """
    Generate a conformer of a complex given a set of molecules, rotations for
    each and points on which to shift

    -----------------------------------------------------------------------
    Arguments:
        molecules (list(autode.species.Species)):

        rotations (list(np.ndarray)): List of len 4 np arrays containing the
                                    [theta, x, y, z] defining the rotation
                                        amount and axis

        points: (list(np.ndarray)): List of length 3 np arrays containing the
                                    point to add the molecule with index i

    Returns:
        (list(autode.atoms.Atom))
    """
    assert len(molecules) - 1 == len(rotations) == len(points) > 0

    # First molecule is static so start with those atoms
    atoms = deepcopy(molecules[0].atoms)

    # For each molecule add it to the current set of atoms with the centroid
    # ~ COM located at the origin
    for i, molecule in enumerate(molecules[1:]):

        centroid = np.average(np.array([atom.coord for atom in atoms]), axis=0)

        # Shift to the origin and rotate randomly, by the same amount
        theta, axis = np.random.uniform(-np.pi, np.pi), np.random.uniform(
            -1, 1, size=3
        )
        for atom in atoms:
            atom.translate(vec=-centroid)
            atom.rotate(axis, theta)

        coords = np.array([atom.coord for atom in atoms])

        mol_centroid = np.average(molecule.coordinates, axis=0)
        shifted_mol_atoms = deepcopy(molecule.atoms)

        # Shift the molecule to the origin then rotate randomly
        theta, axis = rotations[i][0], rotations[i][1:]
        for atom in shifted_mol_atoms:
            atom.translate(vec=-mol_centroid)
            atom.rotate(axis, theta)

        # Shift until the current molecules don't overlap with the current
        #  atoms, i.e. aren't far enough apart
        far_enough_apart = False

        # Shift the molecule by 0.1 Å in the direction of the point
        # (which has length 1) until the
        # minimum distance to the rest of the complex is 2.0 Å
        while not far_enough_apart:

            for atom in shifted_mol_atoms:
                atom.coord += points[i] * 0.1

            mol_coords = np.array([atom.coord for atom in shifted_mol_atoms])

            if np.min(distance_matrix(coords, mol_coords)) > 2.0:
                far_enough_apart = True

        atoms += shifted_mol_atoms

    return atoms


class Complex(Species):
    def __init__(
        self,
        *args: Species,
        name: str = "complex",
        do_init_translation: bool = False,
        copy: bool = True,
        solvent_name: Optional[str] = None,
    ):
        """
        Molecular complex e.g. VdW complex of one or more Molecules

        -----------------------------------------------------------------------
        Arguments:
            *args (autode.species.Species):

        Keyword Arguments:
            name (str):

            do_init_translation (bool): Translate molecules initially such
                                        that they donot overlap

            copy (bool): Should the molecules be copied into this complex?

            solvent_name (str | None): Name of the solvent, if None then select
                                       the first solvent from the constituent
                                       molecules
        """
        super().__init__(
            name=name,
            atoms=sum(
                (deepcopy(mol.atoms) if copy else mol.atoms for mol in args),
                None,
            ),
            charge=sum(mol.charge for mol in args),
            mult=sum(m.mult for m in args) - (len(args) - 1),
        )

        self._molecules = args

        if do_init_translation:
            self._init_translation()

        self.solvent = self._init_solvent(solvent_name)
        self.graph = union(graphs=[mol.graph for mol in self._molecules])

    def __repr__(self):
        return self._repr(prefix="Complex")

    def __eq__(self, other):
        """Equality of two complexes"""
        return isinstance(other, self.__class__) and all(
            a == b for (a, b) in zip(self._molecules, other._molecules)
        )

    @Species.atoms.setter
    def atoms(self, value: Union[List[Atom], Atoms, None]):

        if value is None:
            self.graph = None
            self._molecules = []

        elif self.n_atoms != len(value):
            raise ValueError(
                f"Cannot set atoms in {self.name} with a "
                "different number of atoms. Molecular composition"
                " must have changed."
            )

        logger.warning(
            f"Modifying the atoms of {self.name} - assuming the "
            f"same molecular composition"
        )
        return super(Complex, type(self)).atoms.fset(self, value)

    @property
    def n_molecules(self) -> int:
        """Number of molecules in this molecular complex"""
        return len(self._molecules)

    def atom_indexes(self, mol_index: int):
        """
        List of atom indexes of a molecule withibn a Complex

        -----------------------------------------------------------------------
        Arguments:
            mol_index (int): Index of the molecule
        """
        if mol_index not in set(range(self.n_molecules)):
            raise AssertionError(
                f"Could not get idxs for molecule {mol_index}"
                f". Not present in this complex"
            )

        first_index = sum([mol.n_atoms for mol in self._molecules[:mol_index]])
        last_index = sum(
            [mol.n_atoms for mol in self._molecules[: mol_index + 1]]
        )

        return list(range(first_index, last_index))

    def reorder_atoms(self, mapping: dict) -> None:
        """
        Reorder the atoms in this complex using a dictionary keyed with current
        atom indexes and values as their new positions

        -----------------------------------------------------------------------
        Arguments:
            mapping (dict):
        """
        logger.warning(
            f"Reordering the atoms in a complex ({self.name}) will"
            f" not preserve the molecular composition"
        )

        return super().reorder_atoms(mapping)

    def _generate_conformers(self):
        """
        Generate rigid body conformers of a complex by (1) Fixing the first m
        olecule, (2) initialising the second molecule's COM evenly on the points
        of a sphere around the first with a random rotation and (3) iterating
        until all molecules in the complex have been added
        """
        n = self.n_molecules

        if n < 2:
            # Single (or zero) molecule complex only has a single *rigid body*
            # conformer
            self.conformers = [Conformer(name=self.name, species=self)]
            return None

        self.conformers = []
        m = 0  # Current conformer number

        points_on_sphere = get_points_on_sphere(
            n_points=Config.num_complex_sphere_points
        )

        for _ in iterprod(
            range(Config.num_complex_random_rotations), repeat=n - 1
        ):
            # Generate the rotation thetas and axes
            rotations = [
                np.random.uniform(-np.pi, np.pi, size=4) for _ in range(n - 1)
            ]

            for points in iterprod(points_on_sphere, repeat=n - 1):

                conf = Conformer(
                    name=f"{self.name}_conf{m}",
                    charge=self.charge,
                    mult=self.mult,
                )
                conf.solvent = self.solvent
                conf.atoms = get_complex_conformer_atoms(
                    self._molecules, rotations, points
                )
                self.conformers.append(conf)
                m += 1

                if m == Config.max_num_complex_conformers:
                    logger.warning(
                        f"Generated the maximum number of complex "
                        f"conformers ({m})"
                    )
                    return None

        logger.info(f"Generated {m} conformers")
        return None

    @work_in("conformers")
    def populate_conformers(self):
        r"""
        Generate and optimise with a low level method a set of conformers, the
        number of which is::

        Config.num_complex_sphere_points ×  Config.num_complex_random_rotations
         ^ (n molecules in complex - 1)

        This will not be exact as get_points_on_sphere does not return quite
        the desired number of points for small N.
        """
        n_confs = (
            Config.num_complex_sphere_points
            * Config.num_complex_random_rotations
            * (self.n_molecules - 1)
        )
        logger.info(
            f"Generating and optimising {n_confs} conformers of "
            f"{self.name} with a low-level method"
        )

        self._generate_conformers()

        try:
            lmethod = get_lmethod()
            for conformer in self.conformers:
                conformer.optimise(method=lmethod)
                conformer.print_xyz_file()

        except MethodUnavailable:
            logger.error("Could not optimise complex conformers")

        return None

    def translate_mol(self, vec: Sequence[float], mol_index: int):
        """
        Translate a molecule within a complex by a vector

        -----------------------------------------------------------------------
        Arguments:
            vec (np.ndarray | list(float)): Length 3 vector

            mol_index (int): Index of the molecule to translate. e.g. 2 will
                             translate molecule 1 in the complex
                             they are indexed from 0
        """
        logger.info(
            f"Translating molecule {mol_index} by {vec} in {self.name}"
        )

        if mol_index not in set(range(self.n_molecules)):
            raise ValueError(
                f"Could not translate molecule {mol_index} "
                "not present in this complex"
            )

        for atom_idx in self.atom_indexes(mol_index):
            self.atoms[atom_idx].translate(vec)

        return None

    def rotate_mol(
        self,
        axis: Union[np.ndarray, Sequence],
        theta: Union["autode.values.Angle", float],
        mol_index: int,
        origin: Union[np.ndarray, Sequence, None] = None,
    ):
        """
        Rotate a molecule within a complex an angle theta about an axis given
        an origin

        -----------------------------------------------------------------------
        Arguments:
            axis (np.ndarray | list): Length 3 vector

            theta (float | autode.values.Angle):

            origin (np.ndarray | list): Length 3 vector

            mol_index (int): Index of the molecule to translate. e.g. 2 will
                            translate molecule 1 in the complex
                             they are indexed from 0
        """
        logger.info(
            f"Rotating molecule {mol_index} by {theta:.4f} radians "
            f"in {self.name}"
        )

        if mol_index not in set(range(self.n_molecules)):
            raise ValueError(
                f"Could not rotate molecule {mol_index} "
                "not present in this complex"
            )

        for atom_idx in self.atom_indexes(mol_index):
            self.atoms[atom_idx].rotate(axis, theta, origin)

        return None

    @requires_atoms
    def calc_repulsion(self, mol_index: int):
        """Calculate the repulsion between a molecule and the rest of the
        complex"""

        coords = self.coordinates

        mol_indexes = self.atom_indexes(mol_index)
        mol_coords = [coords[i] for i in mol_indexes]
        other_coords = [
            coords[i] for i in range(self.n_atoms) if i not in mol_indexes
        ]

        # Repulsion is the sum over all pairs 1/r^4
        distance_mat = distance_matrix(mol_coords, other_coords)
        repulsion = 0.5 * np.sum(np.power(distance_mat, -4))

        return repulsion

    def _init_translation(self):
        """Translate all molecules initially to avoid overlaps"""

        if self.n_molecules < 2:
            return  # No need to translate 0 or 1 molecule

        # Points on the unit sphere maximally displaced from one another
        points = get_points_on_sphere(n_points=self.n_molecules)

        # Shift along the vector defined on the unit sphere by the molecule's
        # radius + 4Å, which should generate a somewhat reasonable geometry
        for i in range(self.n_molecules):
            self.translate_mol(
                vec=(self._molecules[i].radius + 4) * points[i], mol_index=i
            )
        return None

    def _init_solvent(self, solvent_name: str):
        """Initial solvent"""

        if solvent_name is not None:
            return get_solvent(solvent_name, kind="implicit")

        if self.n_molecules > 0:
            solvent = self._molecules[0].solvent
            if any(solvent != mol.solvent for mol in self._molecules):
                raise AssertionError(
                    "Cannot form a complex with molecules in "
                    "different solvents"
                )

            return solvent

        return None


class ReactantComplex(Complex):
    # NOTE: Methods must be identical to ProductComplex

    def to_product_complex(self):
        """Return a product complex from this reactant complex"""

        prod_complex = self.copy()
        prod_complex.__class__ = ProductComplex

        return prod_complex

    def __init__(self, *args, name="reac_complex", **kwargs):
        """
        Reactant complex

        -----------------------------------------------------------------------
        Arguments:
            *args (autode.species.Reactant):

        Keyword Arguments:
            name (str):
        """
        super().__init__(*args, name=name, **kwargs)


class ProductComplex(Complex):
    # NOTE: Methods must be identical to ReactantComplex

    def to_reactant_complex(self):
        """Return a reactant complex from this product complex"""

        reac_complex = self.copy()
        reac_complex.__class__ = ReactantComplex

        return reac_complex

    def __init__(self, *args, name="prod_complex", **kwargs):
        """
        Product complex

        -----------------------------------------------------------------------
        Arguments:
            *args (autode.species.Product):

        Keyword Arguments:
            name (str):
        """
        super().__init__(*args, name=name, **kwargs)


class NCIComplex(Complex):
    """Non covalent interaction complex"""


def align_product_to_reactant_by_symmetry_rmsd(
    product_complex: Complex,
    reactant_complex: Complex,
    bond_rearr: "BondRearrangement",
    max_maps: int = 50,
) -> Tuple["Species", "Species"]:
    """
    Aligns a product complex against a reactant complex for a given
    bond rearrangement by iterating through all possible graph
    isomorphisms for the heavy atoms (and any active H), and then
    checking the RMSD on those selected atoms only. Assumes that the
    product complex and reactant complex have at least one conformer
    (in case there is only one molecule, the conformer must be a copy
    of the complex geometry)

    Args:
        product_complex: The product complex (must have at least one conf)
        reactant_complex: The reactant complex (must have at least one conf)
        bond_rearr: The bond rearrangement
        max_maps: Maximum number of heavy-atom isomorphism maps to check against

    Returns:
        (tuple[Species, Species]): The aligned reactant and product geometries
    """
    from networkx.algorithms import isomorphism
    from autode.mol_graphs import reac_graph_to_prod_graph
    from autode.exceptions import NoMapping

    assert len(reactant_complex.conformers) > 0
    assert len(product_complex.conformers) > 0

    # extract only heavy atoms and any hydrogens involved in reaction
    fit_atom_idxs = _get_heavy_and_active_h_atom_indices(
        reactant_complex, bond_rearr
    )

    # NOTE: multiple reaction mappings possible due to symmetry, e.g., 3 hydrogens
    # in -CH3 are symmetric/equivalent in 2D. We choose unique mappings where the
    # mapping of heavy atoms (or hydrogens involved in reaction) change, to reduce
    # the complexity of the problem.
    node_match = isomorphism.categorical_node_match("atom_label", "C")
    gm = isomorphism.GraphMatcher(
        reac_graph_to_prod_graph(reactant_complex.graph, bond_rearr),
        product_complex.graph,
        node_match=node_match,
    )
    assert len(product_complex.conformers) > 0, "Must have conformer(s)"

    unique_mappings = []

    def is_mapping_unique(full_map):
        for this_map in unique_mappings:
            if all(this_map[i] == full_map[i] for i in fit_atom_idxs):
                return False
        return True

    # collect at most <max_trials> number of unique mappings
    for mapping in gm.isomorphisms_iter():
        if is_mapping_unique(mapping):
            unique_mappings.append(mapping)
        if len(unique_mappings) > max_maps:
            break

    logger.info(
        f"Obtained {len(unique_mappings)} possible mappings "
        f"based on heavy and active atoms"
    )
    # todo check the logic of this function
    lowest_rmsd = None
    aligned_rct: Optional["Species"] = None
    aligned_prod: Optional["Species"] = None

    for mapping in unique_mappings:

        sorted_mapping = {i: mapping[i] for i in sorted(mapping)}

        for rct_conf in reactant_complex.conformers:
            # take copy to avoid permanent reordering
            rct_tmp = rct_conf.copy()
            rct_tmp.reorder_atoms(sorted_mapping)
            mapped_atom_idxs = [sorted_mapping[i] for i in fit_atom_idxs]

            for conf in product_complex.conformers:
                rmsd = calc_rmsd_on_atom_indices(
                    conf, rct_tmp, mapped_atom_idxs
                )
                if lowest_rmsd is None or rmsd < lowest_rmsd:
                    lowest_rmsd = rmsd
                    aligned_rct, aligned_prod = rct_tmp, conf.copy()

    logger.info(f"Lowest heavy-atom RMSD of fit = {lowest_rmsd}")

    if aligned_rct is None or aligned_prod is None:
        raise NoMapping("Unable to obtain isomorphism from bond rearrangment")

    align_species(aligned_rct, aligned_prod, fit_atom_idxs)
    # TODO: align the hydrogens by fragments => below map is wrong as changed
    h_atoms_idxs = set(range(aligned_rct.n_atoms)).difference(fit_atom_idxs)

    return aligned_rct, aligned_prod


def _get_heavy_and_active_h_atom_indices(
    reactant: Species, bond_rearr: "BondRearrangement"
) -> List[int]:
    """
    Obtain all the heavy atoms, and only those hydrogens that are
    involved in the reaction, informed by the bond rearrangement

    Args:
        reactant (Species): The reactant species
        bond_rearr (BondRearrangement): The bond rearrangement from reactant
                                        to form product

    Returns:
        (list[int]): The indices of the heavy atoms, and the active hydrogens
    """
    selected_atoms = set()
    for idx, atom in enumerate(reactant.atoms):
        if atom.label != "H":
            selected_atoms.add(idx)

    for pairs in bond_rearr.all:
        selected_atoms.update(pairs)

    return list(selected_atoms)


class _FragmentGroup:
    """
    Denotes a fragment of a molecule consisting of a central atom,
    some hydrogens attached to it, and non-hydrogen neighbours of
    the central atom
    """

    def __init__(self, centre, hydrogens, neighbours):
        self.centre = int(centre)
        self.hydrogens = list(hydrogens)
        self.neighbours = list(neighbours)

    @property
    def n_hs(self) -> int:
        return len(self.hydrogens)

    @property
    def n_neighbours(self) -> int:
        return len(self.neighbours)

    def permutate_hs(self) -> list:
        """Return all indices of atoms, but with permuting hydrogens"""
        for perm in itertools.permutations(self.hydrogens):
            frag_idxs = [self.centre] + list(perm) + self.neighbours
            yield frag_idxs

    def get_fragment(self, mol, perm=None) -> Atoms:
        if perm is None:
            perm = [self.centre] + list(self.hydrogens) + self.neighbours

        frag = Atoms([mol.atoms[i] for i in perm])
        return frag


def get_hydrogen_groups(
    first_species: Species, active_idxs: List[int]
) -> List[_FragmentGroup]:

    fragments = []
    # all the hydrogen atoms that are not active
    h_idxs = [
        i
        for i in range(first_species.n_atoms)
        if first_species.atoms[i].label == "H"
    ]
    h_idxs = list(set(h_idxs).difference(active_idxs))
    used_h_idxs = []
    for idx in h_idxs:
        if idx in used_h_idxs:
            continue
        n_bonds_to_h = first_species.graph.degree[idx]
        if n_bonds_to_h == 0:
            # zero bonds, isolated H atom, no need to permute
            continue
        elif n_bonds_to_h > 1:
            logger.warning(
                f"Unusual geometry - H{idx} has more than one bonds"
                f" skipping alignment"
            )
            continue
        # now there should be only one bond to H
        assert len(list(first_species.graph.neighbors(idx))) == 1
        centre = list(first_species.graph.neighbors(idx))[0]
        centre_neighbours = list(first_species.graph.neighbors(centre))
        # collect all H's attached to centre
        all_hs = [x for x in centre_neighbours if x in h_idxs]
        all_hs.append(idx)
        used_h_idxs.extend(all_hs)
        # get other heavy, non-active atoms
        non_hs = [x for x in centre_neighbours if x not in h_idxs]
        non_hs = list(set(non_hs).difference(active_idxs))
        fragments.append(_FragmentGroup(centre, all_hs, non_hs))

    return fragments


def align_h_groups_by_permutation(
    first_species, second_species, h_groups: List[_FragmentGroup]
):
    for frag_group in h_groups:
        if frag_group.n_hs == 1:
            # only one-H, nothing to be done
            continue
        elif frag_group.n_hs >= 2 or frag_group.n_neighbours == 0:
            fragment = frag_group.get_fragment(first_species)
            if fragment.are_planar() or fragment.are_linear():
                logger.warning(
                    f"Unable to align H{frag_group.hydrogens} due to symmetry"
                )
                continue
        best_rmsd, best_mapping = None, None
        for perm in frag_group.permutate_hs():
            frag1 = frag_group.get_fragment(first_species, perm)
            frag2 = frag_group.get_fragment(second_species)
            rmsd = calc_rmsd(frag1.coordinates, frag2.coordinates)
            if best_rmsd is None or rmsd < best_rmsd:
                best_rmsd = rmsd
                best_mapping = dict(zip(frag_group.hydrogens, perm))

        full_mapping = {i for i in range(first_species.n_atoms)}
        full_mapping.update(best_mapping)
        first_species.reorder_atoms(full_mapping)

    return None
