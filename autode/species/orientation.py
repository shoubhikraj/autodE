"""
Code for relative orientation of molecules in a reactive
complex or product complex, based on bond rearrangement
"""
from typing import TYPE_CHECKING
from autode.bond_rearrangement import BondRearrangement
from autode.exceptions import NoMapping
from autode.mol_graphs import graph_matcher, reac_graph_to_prod_graph

if TYPE_CHECKING:
    from autode.mol_graphs import MolecularGraph


def get_inv_bond_rearr(
    reac_graph: "MolecularGraph",
    prod_graph: "MolecularGraph",
    bond_rearr: BondRearrangement,
) -> BondRearrangement:
    """
    Invert the bond rearrangement, taking into account the graph isomorphism.
    Only returns one possible inverse in case there is symmetry.

    Args:
        reac_graph (MolecularGraph): Reactant graph
        prod_graph (MolecularGraph): Product graph
        bond_rearr (BondRearrangement): The bond rearrangement
    """
    gm = graph_matcher(
        graph1=reac_graph_to_prod_graph(reac_graph, bond_rearr),
        graph2=prod_graph,
    )
    try:
        init_mapping = next(gm.isomorphisms_iter())
    except StopIteration:
        raise NoMapping

    return BondRearrangement(
        breaking_bonds=[
            (init_mapping[i], init_mapping[j]) for i, j in bond_rearr.fbonds
        ],
        forming_bonds=[
            (init_mapping[i], init_mapping[j]) for i, j in bond_rearr.bbonds
        ],
    )
