import itertools
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix

from autode.exceptions import NoMapping
from autode.mol_graphs import MolecularGraph
from autode.species import (
    Complex,
    ReactantComplex,
    ProductComplex,
    Reactant,
    Product,
)
from autode.conformers import Conformers, Conformer
from autode.bond_rearrangement import BondRearrangement, get_bond_rearrangs
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

    def __init__(self, cmplx: Complex, bond_rearr: BondRearrangement):
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
        _k = 0.1
        penalty = 0.0
        new_coords = self.get_rotated_translated_coords(x)
        for i, j in itertools.combinations(range(self.n_molecules), 2):
            mol_i_coords = new_coords[self.idxs_list[i]]
            mol_j_coords = new_coords[self.idxs_list[j]]
            dist_mat = distance_matrix(mol_i_coords, mol_j_coords)
            penalty += 0.5 * np.sum(np.power(dist_mat, -4))
        # add 4th order attractive force
        for idx, (i, j) in enumerate(self.fbonds):
            r = np.linalg.norm(new_coords[i] - new_coords[j])
            r0 = self.vdw_sums[idx]
            penalty += _k * (r - r0) ** 2
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

    for conf in cmplx.conformers:
        tmp_cmplx = reactive_complex.copy()
        tmp_cmplx.conformers = Conformers()
        tmp_cmplx.coordinates = conf.coordinates
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
        put_unique_conf_into_list(tmp_cmplx)

    return prune_complexes_by_fbond_feasibility(
        complex_orientations, bond_rearr
    )


def prune_complexes_by_fbond_feasibility(
    complexes,
    bond_rearr,
):
    fbond_obstructions = []
    fbond_collisions = []
    for cmplx in complexes:
        fbond_obstr_vals = [
            calculate_bond_path_obstruction(cmplx, *fbond)
            for fbond in bond_rearr.fbonds
        ]
        fbond_obstructions.append(
            np.sqrt(np.mean(np.square(fbond_obstr_vals)))
        )
        fbond_collisions.append(
            calculate_fbond_collision_parameter(cmplx, bond_rearr)
        )
    print("Fbond obstructions", fbond_obstructions)
    print("Fbond collisions", fbond_collisions)
    return complexes


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


class InterpAtomMapper:
    """
    Refine atom-mapping based on the TS-like graph for all pairs of coordinates
    """

    def __init__(
        self,
        coords_pairs: list[tuple[np.ndarray, np.ndarray]],
        ts_graph: MolecularGraph,
    ):
        """
        Create an atom-mappper object

        Args:
            coords_pairs (list[tuple[np.ndarray, np.ndarray]]): A list of pairs of
                        reactant and product coordinates
            ts_graph:
        """
        # reshape to (-1, 3)
        self.coords_pairs = [
            (coords_a.reshape((-1, 3)), coords_b.reshape((-1, 3)))
            for coords_a, coords_b in coords_pairs
        ]
        assert isinstance(ts_graph, MolecularGraph)
        self.ts_graph = ts_graph
        self.idpp_obj = IDPP(n_images=_NUM_INTERP_IMAGES, sequential=False)

    @property
    def _core_and_other_idxs(self) -> tuple[list[int], list[int]]:
        """
        Obtain the core indices for the TS-like graph, which include all
        the heavy atoms and any H atom which is involved in the reaction.
        Also returns the non-core indices

        Returns:
            (list[int]): A list of indices of the core atoms
        """
        idxs = list(self.ts_graph.nodes)
        assert idxs == list(range(max(idxs) + 1))
        active_bonds = self.ts_graph.active_bonds
        active_idxs = list(set().union(*active_bonds))
        # NOTE: Only heavy atoms, active atoms and H atoms which are attached
        # to the active atoms, and any H atom with multiple bonds are included
        # in the core part. We do NOT include free, non-participating H2 (rare?)
        core_idxs = set()
        for i in idxs:
            if i in active_idxs:
                core_idxs.add(i)
            elif self.ts_graph.nodes[i]["atom_label"] != "H":
                core_idxs.add(i)
            elif self.ts_graph.nodes[i]["atom_label"] == "H":
                if any(self.ts_graph.has_edge(i, j) for j in active_idxs):
                    core_idxs.add(i)
                if self.ts_graph.degree[i] > 1:
                    core_idxs.add(i)
        return list(core_idxs), list(set(idxs) - set(core_idxs))

    @staticmethod
    def _get_aligned_centred_coords(
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
        coords_a = coords_a - np.average(coords_a, axis=0)
        coords_b = coords_b - np.average(coords_b, axis=0)

        rot_mat = get_rot_mat_kabsch(coords_a, coords_b)
        coords_a = np.dot(rot_mat, coords_a.T).T
        return coords_a, coords_b

    def map_core_atoms(self) -> list[dict[int, int]]:
        """
        Map the core atoms for the reactant and product complexes

        Returns:
            (list[dict[int, int]]): A tuple of the best
                            core mappings and path lengths for those maps
        """
        core_graph = self.ts_graph.subgraph(self._core_and_other_idxs[0])
        gm = graph_matcher(core_graph, core_graph)
        best_core_mappings: list[dict[int, int]] = [
            dict() for _ in self.coords_pairs
        ]
        best_path_lens = [math.inf for _ in self.coords_pairs]
        for mapping in gm.isomorphisms_iter():
            rct_idxs, prod_idxs = zip(*mapping.items())
            for k, (rct_coords, prod_coords) in enumerate(self.coords_pairs):
                # TODO: Remove the get_aligned_idpp_path_len function
                path_len = self.idpp_obj.get_path_length(
                    *self._get_aligned_centred_coords(
                        rct_coords[list(rct_idxs)],
                        prod_coords[list(prod_idxs)],
                    )
                )
                if path_len < best_path_lens[k]:
                    best_path_lens[k] = path_len
                    best_core_mappings[k] = mapping
        logger.info(f"Finished mapping core atoms...")
        return best_core_mappings

    def get_best_mappings(self):
        best_core_maps, best_path_lens = self.map_core_atoms()
        final_maps, final_path_lens = [], []
        for k, coords_pair in enumerate(self.coords_pairs):
            total_map, total_len = self.map_hydrogens(
                coords_pair, best_core_maps[k]
            )
            final_maps.append(total_map)
            final_path_lens.append(total_len)
        return final_maps, final_path_lens

    def map_hydrogens(self, coords_pair, core_map):
        h_idxs = self._core_and_other_idxs[1]
        rct_coords, prod_coords = coords_pair
        # start with core and add hydrogens
        all_mappings = core_map.copy()

        # get h atom groups -XHn (also H2)
        all_h_groups = []
        for idx in h_idxs:
            if any(idx in group for group in all_h_groups):
                continue

            n_bonds_to_h = self.ts_graph.degree[idx]
            # detached H, unusual but may happen(?), add that as a group
            if n_bonds_to_h == 0:
                all_h_groups.append([idx])
            elif n_bonds_to_h == 1:
                centre = list(self.ts_graph.neighbors(idx))[0]
                if self.ts_graph.nodes[centre]["atom_label"] != "H":
                    centre_attached = set(
                        list(self.ts_graph.neighbors(centre))
                    )
                    centre_hs = list(centre_attached.intersection(h_idxs))
                    # put all those Hs into a group
                    all_h_groups.append(centre_hs)
                else:
                    # here we have a free H-H attachment (unusual!)
                    all_h_groups.append([idx, centre])
            else:
                raise RuntimeError(
                    "Something went wrong in counting hydrogens"
                )

        # now map the hydrogen groups, in order of the number of H in groups
        all_h_groups.sort(key=len)
        for h_group in all_h_groups:
            if len(h_group) == 1:
                all_mappings[h_group[0]] = h_group[0]
            elif len(h_group) >= 2:
                print(f"Aligning H group:", h_group)
                best_len = math.inf
                best_perm = None
                for perm in itertools.permutations(h_group):
                    tmp_mappings = all_mappings.copy()
                    tmp_mappings.update(dict(zip(h_group, perm)))
                    rct_idxs, prod_idxs = zip(*tmp_mappings.items())
                    path_len = self.idpp_obj.get_path_length(
                        *self._get_aligned_centred_coords(
                            rct_coords[list(rct_idxs)],
                            prod_coords[list(prod_idxs)],
                        )
                    )
                    print("For permutation:", perm, "length=", path_len)
                    if path_len < best_len:
                        best_len = path_len
                        best_perm = perm
                print("Best length =", best_len)
                all_mappings.update(dict(zip(h_group, best_perm)))

        final_rct_idxs, final_prod_idxs = zip(*all_mappings.items())
        final_len = self.idpp_obj.get_path_length(
            *self._get_aligned_centred_coords(
                rct_coords[list(final_rct_idxs)],
                prod_coords[list(final_prod_idxs)],
            )
        )

        return all_mappings, final_len


def create_aligned_complex_conformers(
    cmplx: Complex, bond_rearr: BondRearrangement
):
    """
    Create conformers of a complex and then align them by adding forces
    along the forming bonds. Modifies the conformer geometries in place.

    Args:
        cmplx: The complex with more than one species
        bond_rearr: The bond rearrangement on the complex
    """
    if cmplx.n_molecules < 2:
        return None

    if len(bond_rearr.fbonds) == 0:
        raise RuntimeError("Something has gone wrong in bond rearrangement")

    cmplx._generate_conformers()

    n_dof = (cmplx.n_molecules - 1) * 6
    for conf in cmplx.conformers:
        print("Minimized one conformer")
        cmplx.coordinates = conf.coordinates
        penalty_func = AlignmentPenalty(cmplx, bond_rearr)
        x0 = np.zeros(n_dof)
        res = minimize(
            fun=penalty_func.penalty_rotate_translate,
            x0=x0,
            method="l-bfgs-b",
        )
        conf.coordinates = penalty_func.get_rotated_translated_coords(res.x)

    return None


def prune_best_aligned_complex_conformers(
    cmplx: Complex,
    bond_rearr: BondRearrangement,
    dtol_fac: float = 1.3,
    rmsd_tol: float = 0.2,
):
    """
    Prune the complexes based on distance criteria, and also
    on RMSD criteria

    Args:
        cmplx:
        bond_rearr:
        dtol_fac:
        rmsd_tol:

    Returns:

    """
    # TODO have to decide some better distance criteria here
    fbonds = bond_rearr.fbonds

    if len(fbonds) == 0 or cmplx.n_conformers == 0:
        return None

    # any complex with more than 30% higher RMS(bond) is removed
    all_rms_bond_ls = []
    for i, conf in enumerate(cmplx.conformers):
        rms_bond_l = math.sqrt(
            sum(conf.distance(*fb) ** 2 for fb in fbonds) / len(fbonds)
        )
        all_rms_bond_ls.append(rms_bond_l)

    min_rms_bond_l = min(all_rms_bond_ls)
    new_conf_list = []
    for i, conf in enumerate(cmplx.conformers):
        if all_rms_bond_ls[i] < min_rms_bond_l * dtol_fac:
            new_conf_list.append(conf)
    print(f"Pruned to {len(new_conf_list)} conformers based on bond lengths")

    rmsd_conf_list: list = []
    for conf in new_conf_list:
        if len(rmsd_conf_list) == 0:
            rmsd_conf_list.append(conf)
        elif all(
            calc_heavy_atom_rmsd(conf.atoms, other.atoms) > rmsd_tol
            for other in rmsd_conf_list
        ):
            rmsd_conf_list.append(conf)
    print(f"Pruned to {len(rmsd_conf_list)} conformers based on " f"RMSD")
    cmplx.conformers = Conformers(rmsd_conf_list)
    return None


def get_pairs_of_reactive_confs(
    rct_cmplx: Complex, prod_cmplx: Complex, bond_rearr: BondRearrangement
):
    """
    Get several pairs of reactant and product conformers for
    further atom-mapping refinement

    Args:
        rct_cmplx:
        prod_cmplx:
        bond_rearr:

    Returns:

    """
    # TODO: simplify this part of code
    if rct_cmplx.n_conformers == 0:
        rct_cmplx.conformers = Conformers(
            [Conformer(name=rct_cmplx.name, species=rct_cmplx)]
        )
    if prod_cmplx.n_conformers == 0:
        prod_cmplx.conformers = Conformers(
            [Conformer(name=prod_cmplx.name, species=prod_cmplx)]
        )

    pairs = []
    if len(prod_cmplx.conformers) > len(rct_cmplx.conformers):
        ref_confs = rct_cmplx.conformers
        other_confs = prod_cmplx.conformers
    else:
        ref_confs = prod_cmplx.conformers
        other_confs = rct_cmplx.conformers

    heavy_idxs = [
        i for i in range(rct_cmplx.n_atoms) if rct_cmplx.atoms[i] != "H"
    ]
    active_idxs = bond_rearr.active_atoms
    heavy_active_idxs = list(set(heavy_idxs).union(set(active_idxs)))

    for ref_conf in ref_confs:
        best_rmsd = math.inf
        best_conf = None
        for other_conf in other_confs:
            rmsd = calc_rmsd(
                ref_conf.coordinates[heavy_active_idxs],
                other_conf.coordinates[heavy_active_idxs],
            )
            print("RMSD", rmsd)
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_conf = other_conf

        pairs.append((ref_conf, best_conf))

    return pairs


def get_rxn_core_indices(ts_graph: "MolecularGraph") -> list[int]:
    """
    Obtain the 'core' for a TS graph, which contains all non-H atoms
    and any H atom which is involved in the reaction and any H with
    non-standard bonding pattern (e.g. attached to two or more atoms)

    Args:
        graph: The TS graph
        bond_rearr: Bond rearrangement for the reaction

    Returns:
        (list[int]): A list of indices of the core atoms
    """
    idxs = list(ts_graph.nodes)
    active_bonds = ts_graph.active_bonds
    active_idxs = list(set().union(*active_bonds))

    core_idxs = set()
    for i in idxs:
        if i in active_idxs:
            core_idxs.add(i)
        elif ts_graph[i]["atom_label"] != "H":
            core_idxs.add(i)
        elif ts_graph[i]["atom_label"] == "H" and ts_graph.degree[i] > 1:
            core_idxs.add(i)
    return list(core_idxs)


def align_map_rct_prod_complexes(
    rct_coords: np.ndarray,
    prod_coords: np.ndarray,
    ts_graph: "MolecularGraph",
):
    """
    Align a pair of reactant and product complexes (coordinates)
    based on automorphisms of TS graphs

    Args:
        rct_coords:
        prod_coords:
        ts_graph:

    Returns:

    """
    core_idxs = get_rxn_core_indices(ts_graph)
    core_graph = ts_graph.subgraph(core_idxs)

    idpp = IDPP(_NUM_INTERP_IMAGES)
    gm = graph_matcher(core_graph, core_graph)
    best_core_mapping = None
    best_core_path_len = math.inf
    for mapping in gm.isomorphisms_iter():
        # TODO: check that the ordering is correct
        rct_idxs, prod_idxs = zip(*mapping.items())
        path_len = idpp.get_path_length(
            rct_coords[rct_idxs], prod_coords[prod_idxs]
        )
        if path_len < best_core_path_len:
            best_core_path_len = path_len
            best_core_mapping = mapping

    if best_core_mapping is None:
        raise NoMapping

    other_idxs = list(set(list(ts_graph)).difference(set(core_idxs)))
    dummy_map = best_core_mapping.copy()
    for k in other_idxs:
        dummy_map[k] = k

    rct_idxs, prod_idxs = zip(*dummy_map.items())
    rct_coords = rct_coords[rct_idxs]
    prod_coords = prod_coords[prod_idxs]

    # perform rigid body alignment on core_idxs
    p_mat = np.array(rct_coords[core_idxs])
    p_mat = p_mat - np.average(p_mat, axis=0)

    q_mat = np.array(prod_coords[core_idxs])
    q_mat = q_mat - np.average(q_mat, axis=0)

    rot_mat = get_rot_mat_kabsch(p_mat, q_mat)
    rct_coords = np.dot(rot_mat, rct_coords.T).T

    # now align the other atoms (i.e. hydrogens)


def match_non_core_hs(
    rct_coords: np.ndarray,
    prod_coords: np.ndarray,
    h_idxs: list[int],
    core_mapping: dict[int, int],
    ts_graph: "MolecularGraph",
):
    """
    Match the non-core H atoms

    Args:
        rct_coords:
        prod_coords:
        h_idxs:
        ts_graph:

    Returns:

    """
    hs_nodes = {}
    for idx in h_idxs:
        assert ts_graph.degree[idx] == 1
        node = list(ts_graph.neighbors(idx))[0]
        assert node not in h_idxs
        if node in hs_nodes:
            continue

        node_neighbours = list(ts_graph.neighbors(node))
        all_node_hs = [k for k in node_neighbours if k in h_idxs]
        hs_nodes[node] = all_node_hs

    idpp = IDPP(_NUM_INTERP_IMAGES)
    total_mapping = None
    for node, h_idxs in hs_nodes.items():
        best_mapping = None
        best_path_len = math.inf
        for perm in itertools.permutations(h_idxs):
            new_map = core_mapping.copy()
            for k, v in zip(h_idxs, perm):
                new_map[k] = v

            rct_idxs, prod_idxs = zip(*new_map.items())
            path_len = idpp.get_path_length(
                rct_coords[rct_idxs], prod_coords[prod_idxs]
            )
            if path_len < best_path_len:
                best_path_len = path_len
                best_mapping = new_map
        assert best_mapping is not None


def align_species(
    reactant_mols: list,
    product_mols: list,
):
    rct_cmplx = ReactantComplex(*reactant_mols)
    prod_cmplx = ProductComplex(*product_mols)
    # TODO: have get_bond_rearrangs ignore symmetry
    bond_rearrs = get_bond_rearrangs(
        rct_cmplx, prod_cmplx, name="alignment", save=False
    )

    for bond_rearr in bond_rearrs:
        rct_copy = rct_cmplx.copy()
        prod_copy = prod_cmplx.copy()
        mapping = get_mapping(
            prod_copy.graph,
            reac_graph_to_prod_graph(rct_copy.graph, bond_rearr),
        )
        prod_copy.reorder_atoms(mapping)
