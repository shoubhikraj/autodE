"""
Code for relative orientation of molecules in a reactive
complex or product complex, based on bond rearrangement
"""
import numpy as np
from typing import TYPE_CHECKING
from autode.bond_rearrangement import BondRearrangement
from autode.exceptions import NoMapping
from autode.mol_graphs import graph_matcher, reac_graph_to_prod_graph

if TYPE_CHECKING:
    from autode.mol_graphs import MolecularGraph
    from autode.species import Species


def get_inv_bond_rearr(
    reac_graph: "MolecularGraph",
    prod_graph: "MolecularGraph",
    bond_rearr: BondRearrangement,
) -> BondRearrangement:
    """
    Invert the bond rearrangement, taking into account the graph isomorphism.
    Only returns one possible inverse in case there is symmetry.

    Args:
        reac_graph (MolecularGraph): Reactant graph
        prod_graph (MolecularGraph): Product graph
        bond_rearr (BondRearrangement): The bond rearrangement
    """
    gm = graph_matcher(
        graph1=reac_graph_to_prod_graph(reac_graph, bond_rearr),
        graph2=prod_graph,
    )
    try:
        init_mapping = next(gm.isomorphisms_iter())
    except StopIteration:
        raise NoMapping

    return BondRearrangement(
        breaking_bonds=[
            (init_mapping[i], init_mapping[j]) for i, j in bond_rearr.fbonds
        ],
        forming_bonds=[
            (init_mapping[i], init_mapping[j]) for i, j in bond_rearr.bbonds
        ],
    )


class AlignmentPenalty:
    """
    Class for calculation of penalty function used to align molecules.
    It puts 1/r^4 repulsion between all atom pairs between different
    molecules, and (r-r0)**4 for the atom-pairs forming bonds
    """

    def __init__(
        self,
        cmplx: Complex,
        bond_rearr: BondRearrangement,
    ):
        """
        Create an object for calculating penalty with 1/r^4 repulsion
        and (r-r0)^4 attraction. r0 is the sum of van der Waals radii
        of the two atoms which will form a bond

        Args:
            cmplx: The complex with more than one molecule
            bond_rearr: The bond rearrangement - only take into account
                        the forming bonds
        """
        self.orig_coords = cmplx.coordinates.reshape(-1, 3)
        self.n_molecules = cmplx.n_molecules
        assert self.n_molecules > 1
        fbonds = bond_rearr.fbonds
        assert all(
            isinstance(fbond[0], int) and isinstance(fbond[1], int)
            for fbond in fbonds
        )
        self.fbonds = fbonds
        self.vdw_sums = [
            cmplx.atoms[i].vdw_radius + cmplx.atoms[j].vdw_radius
            for i, j in self.fbonds
        ]
        self.idxs_list = [
            np.array(cmplx.atom_indexes(i)) for i in range(self.n_molecules)
        ]
        self.subst_centres = self.find_subst_centres(bond_rearr)
        self._anti_subs = False

    @property
    def has_subs_centres(self) -> bool:
        """Are there substitution centres?"""
        return len(self.subst_centres) > 0

    @property
    def using_anti_subs(self) -> bool:
        """Get whether anti-substitution terms are in use"""
        return self._anti_subs

    @using_anti_subs.setter
    def using_anti_subs(self, value: bool):
        """Set whether anti-substitution terms are to be used"""
        self._anti_subs = bool(value)

    @staticmethod
    def find_subst_centres(bond_rearr) -> list[tuple[int, int, int]]:
        """
        Find all substitution centres of type A-C--X

        Args:
            bond_rearr:
        """
        all_subst_centres = []
        for fbond in bond_rearr.fbonds:
            for bbond in bond_rearr.bbonds:
                if len(set(fbond + bbond)) != 3:
                    continue

                c_atom = list(set(fbond).intersection(bbond))[0]
                x_atom = bbond[0] if bbond[1] == c_atom else bbond[1]
                a_atom = fbond[0] if fbond[1] == c_atom else fbond[1]

                all_subst_centres.append((a_atom, c_atom, x_atom))

        return all_subst_centres

    def get_rotated_translated_coords(self, x: np.ndarray) -> np.ndarray:
        """
        Return new coordinates by rotating and translating
        molecules. We keep the first (0-th) molecule fixed,
        and move all other molecules.

        Args:
            x: A numpy array with 6 * (n_molecules - 1) i.e.
               6 numbers for each movable molecule. The first 3
               numbers indicate movement vector for the centre of
               geometry and the last 3 indicate rotation around
               x, y and z axes

        Returns:
            (np.ndarray): New array of coordinates
        """
        new_coords = self.orig_coords.copy()
        x = np.asarray(x)
        assert x.shape == (6 * (self.n_molecules - 1),)

        for i in range(1, self.n_molecules):
            mol_coords = new_coords[self.idxs_list[i]]
            old_origin = mol_coords.mean(axis=0)
            # move to origin, rotate around x, y and z axes
            mol_coords = mol_coords - old_origin
            rot_mat = get_rot_mat_euler(
                axis=np.array([1.0, 0.0, 0.0]), theta=x[3 * i]
            )
            mol_coords = np.matmul(rot_mat, mol_coords.T).T
            rot_mat = get_rot_mat_euler(
                axis=np.array([0.0, 1.0, 0.0]), theta=x[3 * i + 1]
            )
            mol_coords = np.matmul(rot_mat, mol_coords.T).T
            rot_mat = get_rot_mat_euler(
                axis=np.array([0.0, 0.0, 1.0]), theta=x[3 * i + 2]
            )
            mol_coords = np.matmul(rot_mat, mol_coords.T).T
            # translate to old origin + the movement vector
            mol_coords += old_origin + x[3 * (i - 1) : 3 * i]
            new_coords[self.idxs_list[i]] = mol_coords

        return new_coords

    def penalty_rotate_translate(self, x: np.ndarray) -> float:
        """
        Obtain the penalty function for a known rotation and translation
        from the original coordinates. The first (zeroth) molecule
        is kept fixed.

        Args:
            x (np.ndarray): 6 * (n_molecules - 1) array with six
                    numbers for translation and rotation of each
                    moveable molecule.

        Returns:
            (float): The penalty value
        """
        k1 = 0.1
        k2 = 0.1
        penalty = 0.0
        new_coords = self.get_rotated_translated_coords(x)
        for i, j in itertools.combinations(range(self.n_molecules), 2):
            mol_i_coords = new_coords[self.idxs_list[i]]
            mol_j_coords = new_coords[self.idxs_list[j]]
            dist_mat = distance_matrix(mol_i_coords, mol_j_coords)
            penalty += 0.5 * np.sum(np.power(dist_mat, -4))
        # add 2nd order attractive force
        for idx, (i, j) in enumerate(self.fbonds):
            r = np.linalg.norm(new_coords[i] - new_coords[j])
            r0 = self.vdw_sums[idx]
            penalty += k1 * (r - r0) ** 2

        if not self.using_anti_subs:
            return penalty

        for a, c, x in self.subst_centres:
            v_ac = new_coords[c] - new_coords[a]
            v_cx = new_coords[x] - new_coords[c]
            cos_angle = (
                v_ac.dot(v_cx) / np.linalg.norm(v_ac) / np.linalg.norm(v_cx)
            )
            penalty += k2 * (1 - cos_angle) ** 1.5

        return penalty


def point_to_point_path_obstruction(
    pt_a: np.ndarray,
    pt_b: np.ndarray,
    centres: list[np.ndarray],
    radii: list[float],
):
    """
    Obtain the total length of the line segment between point_a and
    point_b which is occluded by spheres with given centres and radii

    Args:
        pt_a (np.ndarray): 3D position vector of first point
        pt_b (np.ndarray): 3D position vector of second point
        centres (list[np.ndarray]): List of 3D position vectors of the
                centres of the spheres
        radii (list[float]): List of radii of the spheres

    Returns:
        (float): The path obstruction length
    """
    intervals = []
    d_ij = pt_b - pt_a
    for centre, radius in zip(centres, radii):
        c_to_a = pt_a - centre
        a_fac = np.dot(d_ij, d_ij)
        b_fac = 2 * np.dot(d_ij, c_to_a)
        c_fac = np.dot(c_to_a, c_to_a) - radius**2
        discr = b_fac**2 - 4 * a_fac * c_fac
        if discr < 0:
            continue
        elif discr < 1e-8:
            continue

        t1 = (-b_fac + np.sqrt(discr)) / (2 * a_fac)
        t2 = (-b_fac - np.sqrt(discr)) / (2 * a_fac)
        t1 = np.clip(t1, 0, 1)
        t2 = np.clip(t2, 0, 1)
        if t1 == t2:
            continue
        assert t2 < t1
        intervals.append((t2, t1))
    if len(intervals) == 0:
        return 0.0
    intervals = sorted(intervals, key=lambda x: x[0])
    total = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    total += cur_end - cur_start
    return total * np.linalg.norm(d_ij)


def calculate_bond_path_obstruction(mol, i, j):
    """
    Estimate how much of the cylinder between atoms i and j in
    molecule mol are obstructed by other atoms

    Args:
        mol:
        i:
        j:

    Returns:
        (float): Average path obstruction
    """

    def get_perp_vector(vec) -> np.ndarray:
        """Return a perpendicular vector to vec using cross products"""
        dot_prods = [
            np.abs(np.dot(vec, np.array([1, 0, 0]))),
            np.abs(np.dot(vec, np.array([0, 1, 0]))),
            np.abs(np.dot(vec, np.array([0, 0, 1]))),
        ]
        idx = np.argmin(dot_prods)
        if idx == 0:
            return np.cross(vec, np.array([1, 0, 0]))
        elif idx == 1:
            return np.cross(vec, np.array([0, 1, 0]))
        else:
            return np.cross(vec, np.array([0, 0, 1]))

    coords_i = mol.coordinates[i]
    coords_j = mol.coordinates[j]
    line = coords_j - coords_i
    axis_1 = get_perp_vector(line)
    axis_2 = np.cross(line, axis_1)
    axis_1 = axis_1 / np.linalg.norm(axis_1)
    axis_2 = axis_2 / np.linalg.norm(axis_2)
    # radius of cylinder is average of the two radii
    c_r = (mol.atoms[i].covalent_radius + mol.atoms[j].covalent_radius) / 2
    # approximate cylinder with 6 'rays' along the sides and central line
    points_i = [coords_i]
    points_j = [coords_j]
    for k in range(6):
        angle = 2 * np.pi * k / 6
        dx_plus_dy = (
            c_r * np.cos(angle) * axis_1 + c_r * np.sin(angle) * axis_2
        )
        points_i.append(coords_i + dx_plus_dy)
        points_j.append(coords_j + dx_plus_dy)

    other_atoms, other_vdw_rs = [], []
    for idx in range(mol.n_atoms):
        if idx in [i, j]:
            continue
        other_atoms.append(mol.coordinates[idx])
        other_vdw_rs.append(mol.atoms[idx].covalent_radius)

    if len(other_atoms) == 0:
        return 0.0

    obstruction_lens = []
    for point_i, point_j in zip(points_i, points_j):
        obstruction_lens.append(
            get_length_obstruction(point_i, point_j, other_atoms, other_vdw_rs)
        )
    return np.average(obstruction_lens)
