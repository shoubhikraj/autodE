"""
Align reactants and products to each other using graph isomorphism
"""
from autode.transition_states.locate_tss import translate_rotate_reactant
from autode.mol_graphs import reac_graph_to_prod_graph
from autode.bond_rearrangement import BondRearrangement
from autode.geom import calc_rmsd
from autode.species.complex import ReactantComplex
from networkx import isomorphism


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
    # TODO: remove Hs not attached to rxn centre


def get_heavy_active_h_indices(rct, bond_rearr):
    """Obtain the indices of the heavy atoms and any active H atoms"""
    # TODO: non-terminal H atoms?
    idxs = set()
    for i in range(rct.n_atoms):
        if rct.atoms[i].label != "H":
            idxs.add(i)
        if i in bond_rearr.active_atoms:
            idxs.add(i)
    return list(idxs)


def align_rct_prod(
    rct: ReactantComplex, prod: ReactantComplex, bond_rearr: BondRearrangement
):
    """Align with graph"""

    rct_rearr_graph = reac_graph_to_prod_graph(rct.graph, bond_rearr)
    node_match = isomorphism.categorical_node_match("atom_label", "C")
    gm = isomorphism.GraphMatcher(
        rct_rearr_graph, prod.graph, node_match=node_match
    )
    # Initial mapping will ensure the correct connectivity
    init_map = next(gm.isomorphisms_iter())
    prod.reorder_atoms(mapping={u: v for v, u in init_map.items()})
    rxn_graph = rct_rearr_graph.copy()
    for i, j in bond_rearr.all:
        rxn_graph.add_edge(i, j, pi=False, active=True)

    heavy_idxs = get_heavy_active_h_indices(rct, bond_rearr)
    align_on_idxs(rct, prod, rxn_graph, heavy_idxs)
    return rct, prod


def align_on_idxs(rct, prod, rxn_graph, heavy_idxs):
    """Graph automorphism to align on a set of indices"""
    rct_coords = rct.coordinates
    prod_coords = prod.coordinates
    subgraph = rxn_graph.subgraph(heavy_idxs)

    node_match = isomorphism.categorical_node_match("atom_label", "C")
    edge_match = isomorphism.categorical_edge_match("active", False)
    gm = isomorphism.GraphMatcher(
        subgraph, subgraph, node_match=node_match, edge_match=edge_match
    )

    best_mapping = None
    best_rmsd = float("inf")
    for mapping in gm.isomorphisms_iter():
        # TODO: check if the order is correct! probably wrong!!!
        rct_idxs, prod_idxs = zip(*mapping.items())
        rmsd = calc_rmsd(rct_coords[rct_idxs], prod_coords[prod_idxs])
        if rmsd < best_rmsd:
            best_mapping = mapping
            best_rmsd = rmsd
        # TODO add maximum number of isomorphisms returned

    if best_mapping is None:
        raise Exception

    pass
