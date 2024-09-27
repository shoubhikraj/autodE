"""
Align reactants and products to each other using graph isomorphism
"""
from autode.transition_states.locate_tss import translate_rotate_reactant
from autode.mol_graphs import reac_graph_to_prod_graph
from autode.bond_rearrangement import BondRearrangement
from autode.species.complex import ReactantComplex


def get_heavy_and_active_h_indices(
    rct: ReactantComplex, prod, bond_rearr: BondRearrangement
):
    """Obtain indices of the heavy atoms, and any active H atoms (which
    are included in the set for convenience of mapping)"""
    assert rct.n_atoms == prod.n_atoms
    rct_idxs = []
    prod_idxs = []
    for i in range(rct.n_atoms):
        if rct.atoms[i].label != "H":
            rct_idxs.append(i)
        if prod.atoms[i].label != "H":
            prod_idxs.append(i)

    active_hs = []
    for i in bond_rearr.active_atoms:
        if rct.atoms[i].label == "H":
            active_hs.append(i)

    if len(active_hs) == 0:
        return rct_idxs, prod_idxs
    # TODO: remove terminal hydrogens only and keep active Hs


def align_rct_prod_with_rmsd(rct, prod, bond_rearr: BondRearrangement):
    """Align with graph"""

    transformed_graph = reac_graph_to_prod_graph(rct.graph, bond_rearr)
