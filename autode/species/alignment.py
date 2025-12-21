import itertools
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix

from networkx import weisfeiler_lehman_subgraph_hashes
from autode.mol_graphs import MolecularGraph, is_isomorphic
from autode.species import (
    Complex,
    ReactantComplex,
    ProductComplex,
    Reactant,
    Product,
)
from autode.conformers import Conformers, Conformer
from autode.bond_rearrangement import BondRearrangement, get_bond_rearrangs
from autode.substitution import get_substc_and_add_dummy_atoms
from autode.geom import (
    calc_rmsd,
    get_rot_mat_euler,
    calc_heavy_atom_rmsd,
    get_rot_mat_kabsch,
)
from autode.mol_graphs import (
    get_mapping,
    reac_graph_to_prod_graph,
    graph_matcher,
)
from autode.log import logger
from autode.neb.idpp import IDPP


_NUM_INTERP_IMAGES = 40


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
            force_anti_subs: Add additional angle terms to enforce anti-substitution
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
        self.anti_subs = False

    @property
    def has_subs_centres(self):
        """Are there substitution centres?"""
        return len(self.subst_centres) > 0

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

        if not self.anti_subs:
            return penalty

        for a, c, x in self.subst_centres:
            v_ac = new_coords[c] - new_coords[a]
            v_cx = new_coords[x] - new_coords[c]
            cos_angle = (
                v_ac.dot(v_cx) / np.linalg.norm(v_ac) / np.linalg.norm(v_cx)
            )
            penalty += k2 * (1 - cos_angle) ** 1.5

        return penalty


def get_oriented_complexes(
    reactive_complex: Complex,
    bond_rearr: BondRearrangement,
    rmsd_prune_tol: float = 0.2,
):
    """
    Create oriented conformers of a complex composed of one-or-more
    species by adding forces along the forming bonds so that their
    van der Waals spheres touch.

    Args:
        reactive_complex:
        bond_rearr:
        rmsd_prune_tol:

    Returns:
        (list[Complex]): A list of Complex objects
    """
    if reactive_complex.n_molecules < 2:
        return [reactive_complex]

    if len(bond_rearr.fbonds) == 0:
        raise RuntimeError(
            "Complex alignment requested, but no forming bonds!"
        )

    # Check that there is at least one active atom in each molecule in complex
    for mol_idx in range(reactive_complex.n_molecules):
        mol_atom_idxs = reactive_complex.atom_indexes(mol_idx)
        assert any(
            atom_idx in fbond
            for fbond in bond_rearr.fbonds
            for atom_idx in mol_atom_idxs
        )
    # TODO: Add a test for this part later
    # work on a copy to avoid modifying the original complex
    cmplx = reactive_complex.copy()
    cmplx.conformers = Conformers()
    cmplx._generate_conformers()

    complex_orientations: list[Complex] = []

    def put_unique_conf_into_list(new_conf):
        """Put only unique conformations (by RMSD) into the list"""
        min_rmsd = np.inf
        for c in complex_orientations:
            rmsd = calc_heavy_atom_rmsd(new_conf.atoms, c.atoms)
            if rmsd < min_rmsd:
                min_rmsd = rmsd
        if min_rmsd > rmsd_prune_tol:
            complex_orientations.append(new_conf)

    tmp_cmplx = reactive_complex.copy()
    tmp_cmplx.conformers = Conformers()
    for conf in cmplx.conformers:
        tmp_cmplx.coordinates = conf.coordinates
        # First without angle terms
        penalty_func = AlignmentPenalty(tmp_cmplx, bond_rearr)
        x0 = np.zeros((6 * (cmplx.n_molecules - 1),))
        res = minimize(
            fun=penalty_func.penalty_rotate_translate,
            x0=x0,
            method="l-bfgs-b",
        )
        tmp_cmplx.coordinates = penalty_func.get_rotated_translated_coords(
            res.x
        )
        put_unique_conf_into_list(tmp_cmplx.copy())
        tmp_cmplx.coordinates = conf.coordinates

        if not penalty_func.has_subs_centres:
            continue
        penalty_func.anti_subs = True
        x0 = np.zeros((6 * (cmplx.n_molecules - 1),))
        res = minimize(
            fun=penalty_func.penalty_rotate_translate,
            x0=x0,
            method="l-bfgs-b",
        )
        tmp_cmplx.coordinates = penalty_func.get_rotated_translated_coords(
            res.x
        )
        put_unique_conf_into_list(tmp_cmplx.copy())

    prune_complexes_by_fbond_feasibility(complex_orientations, bond_rearr)

    return complex_orientations


def prune_complexes_by_fbond_feasibility(
    complexes,
    bond_rearr,
    fbond_obstr_thresh: float = 0.5,
    fbond_collision_thresh: float = 0.2,
):
    """
    Remove complexes based on which conformations are too strained for a
    reaction to take place

    Args:
        complexes:
        bond_rearr:
        fbond_obstr_thresh:
        fbond_collision_thresh:

    Returns:
        (list):
    """
    fbond_obstructions = []
    for cmplx in complexes:
        fbond_obstr_vals = [
            calculate_bond_path_obstruction(cmplx, *fbond)
            for fbond in bond_rearr.fbonds
        ]
        fbond_obstructions.append(
            np.sqrt(np.mean(np.square(fbond_obstr_vals)))
        )

    # Remove all complexes which have large fbond obstruction, but keep
    # at least one if all are removed!
    pruned_complexes = [
        cmplx
        for cmplx, fbond_obstr in zip(complexes, fbond_obstructions)
        if fbond_obstr < fbond_obstr_thresh
    ]
    if len(pruned_complexes) == 0:
        min_fbond_obstr_idx = np.argmin(fbond_obstructions)
        pruned_complexes = [complexes[min_fbond_obstr_idx]]

    # similar treatment for fbond collision
    fbond_collisions = [
        calculate_fbond_collision_parameter(cmplx, bond_rearr)
        for cmplx in pruned_complexes
    ]
    final_complexes = [
        cmplx
        for cmplx, fbond_coll in zip(pruned_complexes, fbond_collisions)
        if fbond_coll > fbond_collision_thresh
    ]
    if len(final_complexes) == 0:
        min_fbond_coll_idx = np.argmin(fbond_collisions)
        final_complexes = [pruned_complexes[min_fbond_coll_idx]]

    print(f"Obtained *{len(final_complexes)}* complexes after pruning")
    return final_complexes


def calculate_fbond_collision_parameter(mol, bond_rearr):
    """
    Obtain the

    Args:
        mol:
        bond_rearr:

    Returns:

    """
    min_dists = []

    for fb1, fb2 in itertools.combinations(bond_rearr.fbonds, 2):
        if len(set(fb1 + fb2)) == 3:
            continue
        coord1, coord2 = mol.coordinates[list(fb1)]
        coord3, coord4 = mol.coordinates[list(fb2)]
        distances = []
        for k in range(0, 101):
            pt = coord1 + (coord2 - coord1) * k / 100.0
            v = coord4 - coord3
            assert np.linalg.norm(v) > 1e-4, "Line segment too short"
            w = pt - coord3
            t = np.clip(w.dot(v) / v.dot(v), 0.0, 1.0)
            q = coord3 + t * v
            distances.append(np.linalg.norm(pt - q))
        min_dists.append(min(distances))

    if len(min_dists) == 0:
        return np.inf

    return min(min_dists)


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

    def get_length_obstruction(pt_a, pt_b, centres, radii) -> float:
        """
        Get the total length of the line between pt_a and pt_b obstructed
        by spheres with specified radii and centres
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

    coords_i = mol.coordinates[i]
    coords_j = mol.coordinates[j]
    line = coords_j - coords_i
    axis_1 = get_perp_vector(line)
    axis_2 = np.cross(line, axis_1)
    axis_1 = axis_1 / np.linalg.norm(axis_1)
    axis_2 = axis_2 / np.linalg.norm(axis_2)
    # radius of cylinder if average of the two radii
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


def create_oriented_mapped_complexes(
    *args, print_interp_geometries: bool = True
):
    """
    Start from a set of reactants and products and then return pairs
    of reactant complexes and product complexes that are aligned as
    well as atom-mapped.

    Args:
        *args:
        print_interp_geometries: Whether to print the path between
                aligned, mapped reactants and products
    """
    reactants = []
    products = []
    for mol in args:
        if isinstance(mol, Reactant):
            reactants.append(mol)
        elif isinstance(mol, Product):
            products.append(mol)
        else:
            raise ValueError(f"Must be Reactant/Product but got {type(mol)}")

    rct_complex = ReactantComplex(*reactants)
    prd_complex = ProductComplex(*products)

    all_brs = get_bond_rearrangs(
        rct_complex, prd_complex, name="dbl", save=False
    )
    print(f"Found *{len(all_brs)}* bond rearrangements")

    for k, bond_rearr in enumerate(all_brs):
        print(f"Bond rearrangement: {repr(bond_rearr)}")
        reactant = rct_complex.copy()
        product = prd_complex.copy()
        # initial mapping sets correct connectivity but not geometry
        init_mapping = get_mapping(
            graph1=product.graph,
            graph2=reac_graph_to_prod_graph(reactant.graph, bond_rearr),
        )
        init_mapping_inv = {v: k for k, v in init_mapping.items()}
        bond_rearr_inv = BondRearrangement(
            breaking_bonds=[
                (init_mapping_inv[i], init_mapping_inv[j])
                for i, j in bond_rearr.fbonds
            ],
            forming_bonds=[
                (init_mapping_inv[i], init_mapping_inv[j])
                for i, j in bond_rearr.bbonds
            ],
        )
        oriented_rcts = get_oriented_complexes(reactant, bond_rearr)
        oriented_prds = get_oriented_complexes(product, bond_rearr_inv)
        for prod in oriented_prds:
            prod.reorder_atoms(init_mapping)
        # TODO: Remove after DEBUG
        for mol in oriented_rcts:
            mol.print_xyz_file(filename=f"rct_{k}.xyz", append=True)
        for mol in oriented_prds:
            mol.print_xyz_file(filename=f"prd_{k}.xyz", append=True)


def write_idpp_path(coords1, coords2):
    """DEBUG"""
    idpp = IDPP(n_images=_NUM_INTERP_IMAGES)
    path = idpp.get_path(coords1, coords2)
    path_coords = [coords1.ravel()]
    counter = 0
    for i in range(_NUM_INTERP_IMAGES - 2):
        path_coords.append(path[counter : counter + coords1.ravel().shape[0]])
        counter += coords1.ravel().shape[0]
    path_coords.append(coords2.ravel())
    with open("path.xyz", "w") as f:
        for coords in path_coords:
            f.write(f"{len(coords)}\n\n")
            for i in range(len(coords) // 3):
                f.write(
                    f"H {coords[3*i]:.8f} {coords[3*i+1]:.8f} {coords[3*i+2]:.8f}\n"
                )


# Convenience function to align a set of coordinates by superposition
def get_aligned_centred_coords(
    coords_a: np.ndarray, coords_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform a Kabsch alignment of two sets of coordinates.
    Will also translate them both to the origin

    Args:
        coords_a: (N x 3) array
        coords_b: (N x 3) array

    Returns:
        (tuple[np.ndarray, np.ndarray]): The aligned coordinates
    """
    coords_a = coords_a.reshape(-1, 3)
    coords_b = coords_b.reshape(-1, 3)
    coords_a = coords_a - np.average(coords_a, axis=0)
    coords_b = coords_b - np.average(coords_b, axis=0)

    rot_mat = get_rot_mat_kabsch(coords_a, coords_b)
    coords_a = np.dot(rot_mat, coords_a.T).T
    return coords_a, coords_b


def get_inv_bond_rearr(
    reac_graph, prod_graph, bond_rearr
) -> BondRearrangement:
    """
    Invert the bond rearrangement, taking into account the graph isomorphism
    """
    gm = graph_matcher(
        graph1=reac_graph_to_prod_graph(reac_graph, bond_rearr),
        graph2=prod_graph,
    )
    # TODO: clean except
    init_mapping = next(gm.isomorphisms_iter())
    return BondRearrangement(
        breaking_bonds=[
            (init_mapping[i], init_mapping[j]) for i, j in bond_rearr.fbonds
        ],
        forming_bonds=[
            (init_mapping[i], init_mapping[j]) for i, j in bond_rearr.bbonds
        ],
    )


def choose_best_bond_rearr_from_equivs(
    mol: Complex, br_set: list[BondRearrangement]
):
    """
    From a set of equivalent bond rearrangements find the option
    that provides the shortest distances between atoms forming
    bonds as well as smallest steric clash in the pathway between
    those atoms

    Args:
        mol: The reactant (or product) complex
        br_set: List of bond rearrangements

    Returns:
        (tuple[BondRearrangement,list[Complex]]): The 'best' possible option and
                    complexes for that bond rearrangement
    """
    best_complexes = None
    best_fbond_attack = np.inf
    for br in br_set:
        complexes = get_oriented_complexes(mol, br)
        # Find the min of (RMS fbond lengths + RMS bond path obstruction)
        all_fbond_costs = [
            np.sqrt(
                np.mean(
                    np.square(
                        cmplx.distance(i, j)
                        + calculate_bond_path_obstruction(cmplx, i, j)
                        for i, j in br.fbonds
                    )
                )
            )
            for cmplx in complexes
        ]
        if min(all_fbond_costs) < best_fbond_attack:
            best_complexes = complexes
            best_fbond_attack = min(all_fbond_costs)

    assert best_complexes is not None
    return best_complexes


def get_equiv_bond_rearrs(
    graph: MolecularGraph, bond_rearr: BondRearrangement
) -> list[BondRearrangement]:
    """
    Given a graph and a bond rearrangement, find all possible equivalent bond
    rearrangements arising from symmetry (e.g. C-H activation for methane can
    have four possible options)

    Args:
        graph:
        bond_rearr:

    Returns:
        (list): List of equivalent bond rearrangements
    """

    def does_br_match_orig(new_br):
        """Check if new bond rearr is truly equivalent to original"""
        if any(graph.has_edge(i, j) for i, j in new_br.fbonds):
            return False
        if not all(graph.has_edge(i, j) for i, j in new_br.bbonds):
            return False
        if is_isomorphic(
            reac_graph_to_prod_graph(graph, new_br),
            reac_graph_to_prod_graph(graph, bond_rearr),
        ):
            return True

    node_hashes = weisfeiler_lehman_subgraph_hashes(
        graph, node_attr="atom_label", iterations=6
    )
    node_hashes = {k: v[-1] for k, v in node_hashes.items()}

    active_idxs = bond_rearr.active_atoms
    assert all(idx in node_hashes.keys() for idx in active_idxs)
    # Get all nodes in graph that have matching hashes to active idx
    active_idx_options = [
        [k for k, v in node_hashes.items() if v == node_hashes[idx]]
        for idx in active_idxs
    ]
    equiv_brs = []
    for comb in itertools.product(*active_idx_options):
        if len(set(comb)) != len(active_idxs):
            continue
        remap = dict(zip(active_idxs, comb))
        new_bond_rearr = BondRearrangement(
            breaking_bonds=[
                (remap[i], remap[j]) for i, j in bond_rearr.bbonds
            ],
            forming_bonds=[(remap[i], remap[j]) for i, j in bond_rearr.fbonds],
        )
        if does_br_match_orig(new_bond_rearr):
            equiv_brs.append(new_bond_rearr)

    return equiv_brs


class AutomorphInterpAMapper:
    """
    Refine atom-mapping based on graph automorphism for all pairs of coordinates
    """

    def __init__(
        self,
        coords_pairs: list[tuple[np.ndarray, np.ndarray]],
        reac_graph: MolecularGraph,
        prod_graph: MolecularGraph,
        bond_rearr: BondRearrangement,
    ):
        """
        Create an atom-mappper object

        Args:
            coords_pairs (list[tuple[np.ndarray, np.ndarray]]): A list of pairs of
                        reactant and product coordinates
            reac_graph: Reactant graph
            prod_graph: Product graph
            bond_rearr: Bond rearrangement
        """
        # reshape to (-1, 3)
        self.coords_pairs = [
            (coords_a.reshape((-1, 3)), coords_b.reshape((-1, 3)))
            for coords_a, coords_b in coords_pairs
        ]
        assert isinstance(reac_graph, MolecularGraph) and isinstance(
            prod_graph, MolecularGraph
        )
        self.reac_conv_graph = reac_graph_to_prod_graph(reac_graph, bond_rearr)
        self.prod_graph = prod_graph
        self.bond_rearr = bond_rearr
        self.inv_bond_rearr = get_inv_bond_rearr(
            reac_graph, prod_graph, bond_rearr
        )
        self.idpp_obj = IDPP(
            n_images=_NUM_INTERP_IMAGES, sequential=False
        )  # TODO: NUM_INTERP_IMAGES

    def _get_xhn_groups_and_capped_graph(self, for_product=False):
        """
        Find all -XHn groups and cap them on the graph, either for the
        reactant or the product
        """
        # If there are X-H bonds forming from the reactant to product (i.e. X-H bonds
        # breaking in inverse bond rearr) then do not remove those hydrogens
        xs_avoid_pruning = []
        if for_product:
            graph = self.prod_graph.copy()
            bonds_to_check = self.inv_bond_rearr.bbonds
        else:
            graph = self.reac_conv_graph.copy()
            bonds_to_check = self.bond_rearr.fbonds

        for i, j in bonds_to_check:
            atom_i, atom_j = (
                graph.nodes[i]["atom_label"],
                graph.nodes[j]["atom_label"],
            )
            if atom_i == "H" and atom_j == "H":
                continue
            if atom_i != "H" and atom_j != "H":
                continue
            h_idx = i if atom_i == "H" else j
            x_idx = j if atom_i == "H" else i
            if graph.degree[h_idx] != 1:
                continue
            xs_avoid_pruning.append(x_idx)

        idxs = list(graph.nodes)
        assert idxs == list(range(max(idxs) + 1))
        nodes_and_hs = {}
        for i in idxs:
            if graph.nodes[i]["atom_label"] == "H":
                continue
            if i in xs_avoid_pruning:
                continue
            neighbours = list(graph.neighbors(i))
            h_neighbours = [
                k
                for k in neighbours
                if graph.nodes[k]["atom_label"] == "H"
                and len(list(graph.neighbors(k))) == 1  # H bonded to only one
                and k not in self.bond_rearr.active_atoms  # H not in reaction
            ]
            if len(h_neighbours) == 0:
                continue

            if len(h_neighbours) > 0:
                nodes_and_hs[i] = h_neighbours

        for node, hs in nodes_and_hs.items():
            graph.remove_nodes_from(hs)
            prev_label = graph.nodes[node]["atom_label"]
            graph.nodes[node]["atom_label"] = prev_label + f"H{len(hs)}"

        return graph, nodes_and_hs

    def map_core_atoms(self) -> list[dict[int, int]]:
        """
        Map the core atoms for the reactant and product complexes

        Returns:
            (list[dict[int, int]]): A tuple of the best
                            core mappings for each pair of coordinates
        """
        reac_conv_core = self._get_xhn_groups_and_capped_graph(
            for_product=False
        )[0]
        prod_core = self._get_xhn_groups_and_capped_graph(for_product=True)[0]
        gm = graph_matcher(reac_conv_core, prod_core)
        best_core_mappings: list[dict[int, int]] = [
            dict() for _ in self.coords_pairs
        ]
        best_path_lens = [np.inf for _ in self.coords_pairs]
        for mapping in gm.isomorphisms_iter():
            reac_idxs, prod_idxs = zip(*mapping.items())
            for k, (reac_coords, prod_coords) in enumerate(self.coords_pairs):
                path_len = self.idpp_obj.get_path_length(
                    *get_aligned_centred_coords(
                        reac_coords[list(reac_idxs)],
                        prod_coords[list(prod_idxs)],
                    )
                )
                if path_len < best_path_lens[k]:
                    best_path_lens[k] = path_len
                    best_core_mappings[k] = mapping
        print("Finished mapping core atoms...")

        reac_idxs, prod_idxs = zip(*best_core_mappings[0].items())
        path_coords = self.idpp_obj.get_path(
            *get_aligned_centred_coords(
                reac_coords[list(reac_idxs)],
                prod_coords[list(prod_idxs)],
            )
        )
        assert path_coords.shape[0] % (_NUM_INTERP_IMAGES - 2) == 0
        floats_per_image = path_coords.shape[0] // (_NUM_INTERP_IMAGES - 2)
        n_atoms = floats_per_image // 3

        coords = path_coords.reshape(_NUM_INTERP_IMAGES - 2, n_atoms, 3)

        # with open("idp_path_viz.trj.xyz", "w") as f:
        #    for img in range(_NUM_INTERP_IMAGES-2):
        #        f.write(f"{n_atoms}\n\n")
        #        for x, y, z in coords[img]:
        #           f.write(f"H {x:.8f} {y:.8f} {z:.8f}\n")
        #
        return best_core_mappings

    def map_non_core_hs(self, coords_pair, core_map):
        """
        Map non-core hydrogens for a pair of coordinates and a given
        core_map
        """
        reac_nodes_hs = self._get_xhn_groups_and_capped_graph(
            for_product=False
        )[1]
        prod_nodes_hs = self._get_xhn_groups_and_capped_graph(
            for_product=True
        )[1]
        all_mappings = core_map.copy()
        reac_coords, prod_coords = coords_pair

        # Map -XHn groups to each other
        for reac_node, reac_hs in reac_nodes_hs.items():
            prod_node = core_map[reac_node]
            prod_hs = prod_nodes_hs[prod_node]
            assert len(prod_hs) == len(reac_hs)
            if len(reac_hs) == 1:
                all_mappings[reac_hs[0]] = prod_hs[0]
            else:
                print(f"Aligning H group: {reac_hs} <-> {prod_hs}")
                best_len = np.inf
                best_perm = None
                for h_perm in itertools.permutations(prod_hs):
                    tmp_mappings = all_mappings.copy()
                    tmp_mappings.update(dict(zip(reac_hs, h_perm)))
                    reac_idxs, prod_idxs = zip(*tmp_mappings.items())
                    path_len = self.idpp_obj.get_path_length(
                        *get_aligned_centred_coords(
                            reac_coords[list(reac_idxs)],
                            prod_coords[list(prod_idxs)],
                        )
                    )
                    if path_len < best_len:
                        best_len = path_len
                        best_perm = h_perm
                all_mappings.update(dict(zip(reac_hs, best_perm)))

        final_reac_idxs, final_prod_idxs = zip(*all_mappings.items())
        final_len = self.idpp_obj.get_path_length(
            *get_aligned_centred_coords(
                reac_coords[list(final_reac_idxs)],
                prod_coords[list(final_prod_idxs)],
            )
        )

        path_coords = self.idpp_obj.get_path(
            *get_aligned_centred_coords(
                reac_coords[list(final_reac_idxs)],
                prod_coords[list(final_prod_idxs)],
            )
        )
        assert path_coords.shape[0] % (_NUM_INTERP_IMAGES - 2) == 0
        floats_per_image = path_coords.shape[0] // (_NUM_INTERP_IMAGES - 2)
        n_atoms = floats_per_image // 3

        coords = path_coords.reshape(_NUM_INTERP_IMAGES - 2, n_atoms, 3)

        with open("idp_path_viz.trj.xyz", "w") as f:
            for img in range(_NUM_INTERP_IMAGES - 2):
                f.write(f"{n_atoms}\n\n")
                for x, y, z in coords[img]:
                    f.write(f"H {x:.8f} {y:.8f} {z:.8f}\n")

        print("After H_mapping", all_mappings, "final_len=", final_len)
        return all_mappings, final_len

    def get_best_mappings(self):
        best_core_maps = self.map_core_atoms()
        final_maps, final_path_lens = [], []
        for k, coords_pair in enumerate(self.coords_pairs):
            total_map, total_len = self.map_non_core_hs(
                coords_pair, best_core_maps[k]
            )
            final_maps.append(total_map)
            final_path_lens.append(total_len)
        return final_maps, final_path_lens
