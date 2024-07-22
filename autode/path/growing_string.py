"""
Single-ended Growing String Method (SE-GSM) implementation. Does not
relax the final path with a double-ende method, and uses slightly
different internal coordinate definitions.
"""
import numpy as np
from typing import TYPE_CHECKING
from autode.path import Path
from autode.opt.coordinates.cartesian import CartesianCoordinates
from autode.opt.coordinates.primitives import (
    ConstrainedCompositeBonds,
    PrimitiveDistance,
)
from autode.opt.coordinates.internals import AnyPIC
from autode.opt.optimisers.qa import QAOptimiser
from autode.config import Config
from autode.log import logger


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
        self.bond_grad_history = []

        # Add the first point after unconstrained minimization
        assert initial_species is not None
        point = initial_species.new_species(with_constraints=False)
        self.add(point, unconstrained=True)
        # TODO: append point should be made aware of constraints

    def add(self, point, unconstrained=False):
        """
        Add a new point to the path, take a step along the reaction
        coordinate, and optimise

        Args:
            point:
            unconstrained:

        Returns:

        """
        idx = len(self) - 1

        constr = None
        if not unconstrained:
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

    def generate(self, name="gsm_path"):
        """
        Generate a growing string path from the starting point. Can be called
        only once.

        ---------------------------------------------------------------------

        Args:
            name (str): Prefix for plots and geometries
        """
        logger.info("Generating SE-GSM path from initial point")

        def reached_final_point():
            idx = self.product_idx(product=self.final_species)
            if idx is not None:
                return True

        while not reached_final_point():
            point = self[-1].new_species()
            self.append(point)

    def _get_gradients_across_ics(self, point):
        """
        Project the Cartesian gradient onto the redundant internal
        coordinate space to obtain the gradients across bonds for
        the point given

        Returns:
            (list[float]): Gradients along the bonds
        """
        assert point.gradient is not None, "Must have gradients!"
        pic = AnyPIC.from_species(point)

        # find indices of specified bonds in PIC
        bond_positions = []
        for bond in self.bonds:
            i, j = bond.atom_indexes
            ic = PrimitiveDistance(i, j)
            if ic not in pic:
                pic.add(ic)
            bond_positions.append(pic.index(ic))

        # project to redundant internals
        x = CartesianCoordinates(point.coordinates)
        g_x = np.array(point.gradient).flatten()
        B = pic.get_B(x)
        B_inv = np.linalg.pinv(B)
        g_q = np.matmul(B_inv.T, g_x).flatten()

        return list(g_q[bond_positions])

    def _have_bonds_crossed_barrier(self):
        """
        Detect whether the bonds have crossed the barrier so far
        or not by checking the sign of the gradient

        Returns:
            (list[bool]):
        """
        from autode.bonds import FormingBond, BreakingBond

        hist = zip(*self.bond_grad_history)

        has_crossed = []
        for idx, bond in enumerate(self.bonds):
            grads = hist[idx]
            for k in range(len(hist) - 2):
                a, b, c = grads[k : k + 3]
                if a < b < c and a < 0 < c:
                    increasing = True
                if a > b > c and a > 0 > c:
                    decreasing = True

            # forming bond sign -ve -> +ve

    def _get_driving_coordinate(self, point: "Species"):
        """
        Obtain the GSM driving coordinate for a point. Should
        have gradients defined.

        Args:
            point (Species):

        Returns:
            (ConstrainedCompositeBonds): The constrained primitive
        """
        bonds = []
        coeffs = []
        msg = "Current driving coordinates: "

        g_q = self._get_gradients_across_ics(point)

        for idx, bond in enumerate(self.bonds):
            i, j = bond.atom_indexes
            d_curr = point.distance(i, j)
            d0 = bond.final_dist

            # The target value (bond.final_dist) is estimated from product
            # and may not be correct for this GSM path. Check the gradient
            # when close to convergence and stop driving if too low

            # enforce the sign
            c = np.sign(bond.dr) * np.abs(
                point.distance(i, j) - bond.final_dist
            )

            # stop moving along almost formed/broken bonds
            if np.abs(c) > 1e-4:
                bonds.append((i, j))
                coeffs.append(c)
                msg += f"{bond} * {c:.3f}"

        coeffs = list(np.array(coeffs) / np.average(coeffs))
        logger.info(msg)
        driven_coord = ConstrainedCompositeBonds(bonds, coeffs, value=0)

        # take a step along the coordinate
        curr_val = driven_coord(point.coordinates.flatten())
        next_val = curr_val + self.step_size

        return ConstrainedCompositeBonds(bonds, coeffs, value=next_val)
