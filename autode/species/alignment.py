import itertools
import math

import numpy as np
from scipy.optimize import minimize
from autode.species import Complex
from autode.conformers import Conformers, Conformer
from autode.bond_rearrangement import BondRearrangement


def forming_bonds_vdw_force_term(
    cmplx: Complex, bond_rearr: BondRearrangement
):
    """
    Return force terms on the bonds that are forming: k (x-x0)**2.
    Here x0 is the ideal distance which is the sum of van der Waals
    radii of the atoms.

    Args:
        cmplx:
        bond_rearr:

    Returns:
        (float): The value of the force term
    """
    e_sum = 0.0
    k = 1.0
    for i, j in bond_rearr.fbonds:
        r_i_vdw = cmplx.atoms[i].vdw_radius
        r_j_vdw = cmplx.atoms[j].vdw_radius
        r0 = r_i_vdw + r_j_vdw
        r = cmplx.distance(i, j)
        s = r0 / 1.414
        e_sum -= 1 / r**4  # remove the repulsion term
        e_sum += k * ((s / r) ** 4 - (s / r) ** 2)

    return e_sum


def get_energy_rotate_translate(
    x: np.ndarray,
    cmplx: Complex,
    bond_rearr: BondRearrangement,
    return_coords: bool = False,
):
    """
    Get the 'energy' based on hard sphere r^4 repulsion, and
    r^6 attraction on the forming bonds. The first species
    in the complex is always considered stationary.


    Args:
        x:
        cmplx:
        bond_rearr:
        return_coords:

    Returns:

    """
    # Must have 6 DOF variables for each molecule
    _x = x.ravel()
    assert len(_x) == (cmplx.n_molecules - 1) * 6
    assert cmplx.n_molecules > 1

    edited_cmplx = cmplx.copy()
    cmplx_coords = edited_cmplx.coordinates
    for i in range(edited_cmplx.n_molecules):
        # do not move first molecule
        if i == 0:
            continue
        mol_idxs = edited_cmplx.atom_indexes(i)
        mol_cog = cmplx_coords[mol_idxs].reshape(-1, 3).mean(axis=0)
        theta_x, theta_y, theta_z = _x[i + 2 : i + 5]
        edited_cmplx.rotate_mol([1, 0, 0], theta_x, i, mol_cog)
        edited_cmplx.rotate_mol([0, 1, 0], theta_y, i, mol_cog)
        edited_cmplx.rotate_mol([0, 0, 1], theta_z, i, mol_cog)
        edited_cmplx.translate_mol(_x[i - 1 : i + 2], i)

    total_en = 0.0
    for i in range(edited_cmplx.n_molecules):
        total_en += edited_cmplx.calc_repulsion(i)
    total_en = total_en / edited_cmplx.n_molecules

    bond_term = forming_bonds_vdw_force_term(edited_cmplx, bond_rearr)
    if not return_coords:
        return total_en + bond_term
    else:
        return edited_cmplx.coordinates


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
        x0 = np.zeros(n_dof)
        res = minimize(
            fun=get_energy_rotate_translate,
            x0=x0,
            method="l-bfgs-b",
            args=(
                cmplx,
                bond_rearr,
            ),
        )
        conf.coordinates = get_energy_rotate_translate(
            res.x, cmplx, bond_rearr, return_coords=True
        )

    return None


def prune_best_aligned_complex_conformers(
    cmplx: Complex, bond_rearr: BondRearrangement
):
    """
    Prune the complexes based on distance criteria

    Args:
        cmplx:
        bond_rearr:

    Returns:

    """
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
        if all_rms_bond_ls[i] < min_rms_bond_l * 1.3:
            new_conf_list.append(conf)
    print(f"Pruned to {len(new_conf_list)} conformers")
    cmplx.conformers = Conformers(new_conf_list)
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
