"""
Single-ended Growing String Method (SE-GSM) implementation. Does not
relax the final path with a double-ende method, and uses slightly
different internal coordinate definitions.
"""
import numpy as np
from typing import TYPE_CHECKING
from autode.path import Path
from autode.opt.coordinates.primitives import ConstrainedCompositeBonds
from autode.opt.optimisers.qa import QAOptimiser
from autode.config import Config


if TYPE_CHECKING:
    from autode.species import Species
    from autode.bond_rearrangement import BondRearrangement
    from autode.bonds import ScannedBond


class SEGSM(Path):
    def __init__(
        self, bonds, method, initial_species, final_species, step_size=0.1
    ):
        """
        SE-GSM path from initial to final species.

        -----------------------------------------------------------------------
        Args:
            bonds (list[ScannedBond]):
            method:
            initial_species (Species):
            final_species:
        """
        super().__init__()

        self.method = method
        self.bonds = bonds
        self.final_species = final_species
        self.step_size = step_size

        # Add the first point after unconstrained minimization
        assert initial_species is not None
        point = initial_species.new_species(with_constraints=False)
        # TODO: append point should be made aware of constraints

    def append(self, point):
        """
        Add a new point to the path and take a step along the reaction
        coordinate

        Args:
            point:

        Returns:

        """
        idx = len(self) - 1

        constr = self._get_driving_coordinate(point)
        opt = QAOptimiser(
            maxiter=50, gtol=5e-4, etol=1e-4, extra_prims=[constr]
        )
        opt.run(
            point,
            method=self.method,
            n_cores=Config.n_cores,
            name=f"gsm_path{idx}",
        )
        point.reset_graph()

        return super().append(point)

    def generate(self):
        """Generate a growing string path"""

    def _get_driving_coordinate(self, point: "Species"):
        """
        Obtain the GSM driving coordinate for a point

        Args:
            point (Species):

        Returns:
            (ConstrainedCompositeBonds): The constrained primitive
        """
        bonds = []
        coeffs = []

        for bond in self.bonds:
            i, j = bond.atom_indexes
            # enforce the sign
            c = np.sign(bond.dr) * np.abs(
                point.distance(i, j) - bond.final_dist
            )

            # stop moving along almost formed/broken bonds
            if np.abs(c) > 1e-4:
                bonds.append((i, j))
                coeffs.append(c)

        driven_coord = ConstrainedCompositeBonds(bonds, coeffs, value=0)

        # take a step along the coordinate
        curr_val = driven_coord(point.coordinates.flatten())
        next_val = curr_val + self.step_size

        return ConstrainedCompositeBonds(bonds, coeffs, value=next_val)
