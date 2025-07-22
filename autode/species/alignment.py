import numpy as np
from scipy.optimize import minimize
from autode.species import Complex
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
        e_sum += k * (r - r0) ** 6

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
