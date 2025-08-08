import itertools
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
from autode.species import Complex
from autode.conformers import Conformers, Conformer
from autode.bond_rearrangement import BondRearrangement
from autode.geom import calc_rmsd, get_rot_mat_euler


class R4Penalty:
    """Calculate the penalty for a set of rotations and translations"""

    def __init__(self, cmplx: Complex, bond_rearr):
        self.orig_coords = cmplx.coordinates.reshape(-1, 3)
        self.n_molecules = cmplx.n_molecules
        fbonds = bond_rearr.fbonds
        assert all(
            isinstance(fbond[0], int) and isinstance(fbond[1], int)
            for fbond in fbonds
        )
        self.fbonds = fbonds
        self.vdw_radii = [
            cmplx.atoms[i].vdw_radius + cmplx.atoms[j].vdw_radius
            for i, j in self.fbonds
        ]
        self.sigmas = [r / (2 ** (1 / 6)) for r in self.vdw_radii]
        self.idxs_list = [
            np.array(cmplx.atom_indexes(i)) for i in range(self.n_molecules)
        ]

    def get_rotated_translated_coords(self, x):
        new_coords = self.orig_coords.copy()
        x = np.asarray(x)
        assert x.shape == (6 * (self.n_molecules - 1),)
        for i in range(1, self.n_molecules):
            mol_coords = new_coords[self.idxs_list[i]]
            old_origin = mol_coords.mean(axis=0)
            mol_coords = mol_coords - old_origin
            rot_mat = get_rot_mat_euler(axis=[1, 0, 0], theta=x[3 * i])
            mol_coords = np.matmul(rot_mat, mol_coords.T).T
            rot_mat = get_rot_mat_euler(axis=[0, 1, 0], theta=x[3 * i + 1])
            mol_coords = np.matmul(rot_mat, mol_coords.T).T
            rot_mat = get_rot_mat_euler(axis=[0, 0, 1], theta=x[3 * i + 2])
            mol_coords = np.matmul(rot_mat, mol_coords.T).T
            mol_coords += old_origin + x[3 * (i - 1) : 3 * i]
            new_coords[self.idxs_list[i]] = mol_coords
        return new_coords

    def vdw_r4_penalty_rotate_translate(self, x):
        """x is numpy array with 6 * (cmplx.n_molecules - 1)"""
        _k = 0.3
        penalty = 0.0
        new_coords = self.get_rotated_translated_coords(x)
        for i, j in itertools.combinations(range(self.n_molecules), 2):
            mol_i_coords = new_coords[self.idxs_list[i]]
            mol_j_coords = new_coords[self.idxs_list[j]]
            dist_mat = distance_matrix(mol_i_coords, mol_j_coords)
            penalty += 0.5 * np.sum(np.power(dist_mat, -4))
        # add van der Waals terms, remove r4 repulsion
        for idx, (i, j) in enumerate(self.fbonds):
            r = np.linalg.norm(new_coords[i] - new_coords[j])
            r0 = self.vdw_radii[idx]
            penalty += _k * (r - r0) ** 4
        return penalty


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
        penalty_func = R4Penalty(cmplx, bond_rearr)
        x0 = np.zeros(n_dof)
        res = minimize(
            fun=penalty_func.vdw_r4_penalty_rotate_translate,
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
    print(
        f"Pruned to {len(new_conf_list)} conformers based on " f"bond lengths"
    )

    rmsd_conf_list: list = []
    for conf in new_conf_list:
        if len(rmsd_conf_list) == 0:
            rmsd_conf_list.append(conf)
        elif all(
            calc_rmsd(conf.coordinates, other.coordinates) > rmsd_tol
            for other in rmsd_conf_list
        ):
            rmsd_conf_list.append(conf)
    print(f"Pruned to {len(rmsd_conf_list)} conformers based on " f"RMSD")
    cmplx.conformers = Conformers(rmsd_conf_list)
    return None


def get_best_pair_alignment(rct_cmplx: Complex, prod_cmplx: Complex):
    """
    Obtain a pair of conformers based on 3D geometrical similarity

    Args:
        rct_cmplx:
        prod_cmplx:

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

    def get_cme_mol(mol):
        """Get Coulomb matrix eigenvalues"""
        arr = np.zeros(shape=(mol.n_atoms, mol.n_atoms))
        for i in range(mol.n_atoms):
            for j in range(mol.n_atoms):
                if i == j:
                    arr[i, j] = 0.5 * mol.atoms[i].atomic_number ** 2.4
                else:
                    arr[i, j] = (
                        mol.atoms[i].atomic_number * mol.atoms[j].atomic_number
                    ) / mol.distance(i, j)
        return np.linalg.eigvalsh(arr)

    rct_cme_list = []
    prod_cme_list = []
    for conf in rct_cmplx.conformers:
        rct_cme_list.append(get_cme_mol(conf))
    for conf in prod_cmplx.conformers:
        prod_cme_list.append(get_cme_mol(conf))

    lowest_similarity = None
    best_pair: tuple = tuple()
    for rct_i, prod_j in itertools.product(
        range(rct_cmplx.n_conformers), range(prod_cmplx.n_conformers)
    ):
        cme_1 = rct_cme_list[rct_i]
        cme_2 = prod_cme_list[prod_j]
        similarity = np.linalg.norm(cme_1 - cme_2)
        if lowest_similarity is None or similarity < lowest_similarity:
            lowest_similarity = similarity
            best_pair = rct_i, prod_j

    return (
        rct_cmplx.conformers[best_pair[0]],
        prod_cmplx.conformers[best_pair[1]],
    )
