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
        e_sum += k * (r - r0) ** 2

    return e_sum


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
    cmplx._generate_conformers()
