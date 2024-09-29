"""
Align reactants and products to each other using graph isomorphism
"""
from typing import TYPE_CHECKING, List
from autode.transition_states.locate_tss import translate_rotate_reactant
from autode.mol_graphs import reac_graph_to_prod_graph
from autode.bond_rearrangement import BondRearrangement
from autode.geom import calc_rmsd
from autode.species.complex import ReactantComplex
from networkx import isomorphism

if TYPE_CHECKING:
    from autode.species import Species


def get_rxn_core_indices(
    rct: "Species", bond_rearr: "BondRearrangement"
) -> List[int]:
    """
    Obtain the 'core' for a reactant and a set of bond rearrangements
    which includes all non-hydrogen atoms, any H-atoms involved in the
    reaction, and any H-atoms that have non-standard bonding pattern
    (i.e. attached to more than one neighbour)

    Args:
        rct: The reactant (can be molecule or complex)
        bond_rearr: Bond rearrangement for the reaction

    Returns:
        (list[int]): A list of indices of the core atoms
    """
    assert rct.graph is not None
    idxs = set()
    for i in range(rct.n_atoms):
        if rct.atoms[i].label != "H":
            idxs.add(i)
        elif rct.atoms[i].label == "H" and rct.graph.degree[i] > 1:
            idxs.add(i)
        if i in bond_rearr.active_atoms:
            idxs.add(i)
    return list(idxs)


def align_rct_prod(
    reactant: "Species", product: "Species", bond_rearr: "BondRearrangement"
):
    """
    Align the reactant and product using their graphs, and the given
    bond rearrangement. First aligns the 'core' atoms (excluding terminal
    hydrogen atoms) and then the remaining H atoms. Modifies the reactant
    and product objects in-place with the best mapping.

    Args:
        reactant: The reactant molecule/complex
        product: The product molecule/complex
        bond_rearr: Bond rearrangement for reaction
    """

    rct_rearr_graph = reac_graph_to_prod_graph(reactant.graph, bond_rearr)
    node_match = isomorphism.categorical_node_match("atom_label", "C")
    gm = isomorphism.GraphMatcher(
        rct_rearr_graph, product.graph, node_match=node_match
    )
    # Initial mapping will ensure the correct connectivity
    init_map = next(gm.isomorphisms_iter())
    product.reorder_atoms(mapping={u: v for v, u in init_map.items()})

    core_idxs = get_rxn_core_indices(reactant, bond_rearr)
    core_mapping = get_aligned_mapping_on_core(
        reactant, product, rct_rearr_graph, core_idxs
    )

    return None


MAX_MAPS = 50


def get_aligned_mapping_on_core(rct, prod, mol_graph, core_idxs):
    """
    Align the reactant and product using the 'core' part of the
    graph. Assumes reactant and product have been initially mapped
    once. Checks all possible isomorphism mappings on the core
    part, and returns the one with lowest RMSD.

    Args:
        rct:
        prod:
        mol_graph:
        core_idxs:

    Returns:

    """
    # TODO automorphism over rct and also prod
    rct_coords = rct.coordinates
    prod_coords = prod.coordinates
    subgraph = mol_graph.subgraph(core_idxs)

    node_match = isomorphism.categorical_node_match("atom_label", "C")
    gm = isomorphism.GraphMatcher(subgraph, subgraph, node_match=node_match)

    best_mapping = None
    best_rmsd = float("inf")
    for counter, mapping in enumerate(gm.isomorphisms_iter()):
        # TODO: check if the order is correct! probably wrong!!!
        rct_idxs, prod_idxs = zip(*mapping.items())
        rmsd = calc_rmsd(rct_coords[rct_idxs], prod_coords[prod_idxs])
        if rmsd < best_rmsd:
            best_mapping = mapping
            best_rmsd = rmsd
        if counter > MAX_MAPS:
            break

    if best_mapping is None:
        raise Exception

    return best_mapping
