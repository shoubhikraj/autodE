"""
Align reactants and products to each other using graph isomorphism
"""
import itertools
import numpy as np
from typing import TYPE_CHECKING, List
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from autode.transition_states.locate_tss import translate_rotate_reactant
from autode.mol_graphs import reac_graph_to_prod_graph
from autode.bond_rearrangement import BondRearrangement
from autode.geom import calc_rmsd
from autode.species.complex import ReactantComplex
from networkx import isomorphism

if TYPE_CHECKING:
    from autode.species import Species


def get_rxn_core_indices(
    rct: "Species", prod: "Species", bond_rearr: "BondRearrangement"
) -> List[int]:
    """
    Obtain the 'core' for a reactant and a set of bond rearrangements
    which includes all non-hydrogen atoms, any H-atoms involved in the
    reaction, and any H-atoms that have non-standard bonding pattern
    (i.e. attached to more than one neighbour)

    Args:
        rct: The reactant molecule/complex
        prod: The product molecule/complex
        bond_rearr: Bond rearrangement for the reaction

    Returns:
        (list[int]): A list of indices of the core atoms
    """
    assert rct.graph is not None and prod.graph is not None
    idxs = set()
    for i in range(rct.n_atoms):
        if rct.atoms[i].label != "H":
            idxs.add(i)
        elif rct.atoms[i].label == "H" and (
            rct.graph.degree[i] > 1 or prod.graph.degree[i] > 1
        ):
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

    core_idxs = get_rxn_core_indices(reactant, product, bond_rearr)
    core_map = get_aligned_mapping_on_core(reactant, product, core_idxs)
    other_idxs = set(range(reactant.n_atoms)).difference(set(core_idxs))

    # the dummy map is needed for reordering atoms
    dummy_full_map = core_map.copy()
    for k in other_idxs:
        dummy_full_map[k] = k
    product.reorder_atoms(mapping={u: v for v, u in dummy_full_map.items()})
    # TODO: have to RMSD minimise on core mapping

    return None


MAX_MAPS = 50


def match_non_core_h(rct, prod, h_idxs):
    """
    Match the non-core H atoms to each other by using the
    connectivity and the Hungarian algorithm

    Args:
        rct:
        prod:
        h_idxs:
    """
    rct_nodes = {}
    prod_nodes = {}
    for idx in h_idxs:
        assert rct.graph.degree[idx] == 1
        node = list(rct.graph.neighbors(idx))[0]
        assert node not in h_idxs
        if node in rct_nodes:
            continue

        rct_neighbors = list(rct.graph.neighbors(node))
        all_rct_hs = [k for k in rct_neighbors if k in h_idxs]
        rct_nodes[node] = all_rct_hs
        prod_neighbors = list(prod.graph.neighbors(node))
        all_prod_hs = [k for k in prod_neighbors if k in h_idxs]
        prod_nodes[node] = all_prod_hs

    # sanity checks
    reconstr_rct_idxs = list(itertools.chain(*rct_nodes.values()))
    reconstr_prod_idxs = list(itertools.chain(*prod_nodes.values()))
    assert len(reconstr_prod_idxs) == len(reconstr_rct_idxs) == len(h_idxs)
    assert set(reconstr_rct_idxs) == set(reconstr_prod_idxs)

    assigned_map = {}
    for node in rct_nodes.keys():
        row_idxs = rct_nodes[node]
        col_idxs = prod_nodes[node]
        # TODO: check once by debugging
        dist_matrix = cdist(
            rct.coordinates[row_idxs], prod.coordinates[col_idxs]
        )
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        node_h_map = zip(
            list(np.array(row_idxs)[row_ind]),
            list(np.array(col_idxs)[col_ind]),
        )
        assigned_map.update(dict(node_h_map))

    return assigned_map


def get_aligned_mapping_on_core(rct, prod, core_idxs):
    """
    Align the reactant and product using the 'core' part of the
    graph. Assumes reactant and product have been initially mapped
    once. Checks all possible isomorphism mappings on the core
    part, and returns the one with lowest RMSD.

    Args:
        rct:
        prod:
        core_idxs:

    Returns:
        (dict): Dictionary of best reactant -> product mapping
    """
    rct_coords = rct.coordinates
    prod_coords = prod.coordinates
    rct_core = rct.graph.subgraph(core_idxs)
    prod_core = prod.graph.subgraph(core_idxs)

    # automorphism with reactant graph as well as product graph
    node_match = isomorphism.categorical_node_match("atom_label", "C")
    gm1 = isomorphism.GraphMatcher(rct_core, rct_core, node_match=node_match)
    gm2 = isomorphism.GraphMatcher(prod_core, prod_core, node_match=node_match)
    all_possible_maps = itertools.chain(
        gm1.isomorphisms_iter(), gm2.isomorphisms_iter()
    )

    previous_maps = []
    best_mapping = None
    best_rmsd = float("inf")
    for mapping in all_possible_maps:
        if mapping not in previous_maps:
            previous_maps.append(mapping)
        if len(previous_maps) > MAX_MAPS:
            break
        # TODO: check if the order is correct!
        rct_idxs, prod_idxs = zip(*mapping.items())
        rmsd = calc_rmsd(rct_coords[rct_idxs], prod_coords[prod_idxs])
        if rmsd < best_rmsd:
            best_mapping = mapping
            best_rmsd = rmsd

    if best_mapping is None:
        raise Exception

    return best_mapping
