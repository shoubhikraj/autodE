"""
Alignment to obtain reactant and product complexes
"""
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from autode.bond_rearrangement import BondRearrangement
    from autode.species import Complex


def align_map_reactant_and_product(
    rct_complex: "Complex",
    prod_complex: "Complex",
    bond_rearr: "BondRearrangement",
) -> None:
    """
    Align the reactant and product complexes based on the provided
    bond rearrangment, and then obtain the best mapping. Suitable for
    starting double-ended TS search runs

    Args:
        rct_complex:
        prod_complex:
        bond_rearr:
    """
