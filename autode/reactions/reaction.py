import os
import base64
import hashlib
import pickle

from typing import Union, Optional, List, Generator, TYPE_CHECKING, Tuple
from datetime import date
from autode.config import Config
from autode.solvent.solvents import get_solvent
from autode.transition_states.locate_tss import (
    find_tss,
    translate_rotate_reactant,
)
from autode.bond_rearrangement import get_bond_rearrangs
from autode.transition_states import TransitionState, TransitionStates
from autode.exceptions import UnbalancedReaction, SolventsDontMatch, NoMapping
from autode.log import logger
from autode.methods import get_hmethod, get_lmethod
from autode.species.complex import ReactantComplex, ProductComplex
from autode.species.molecule import Reactant, Product
from autode.plotting import plot_reaction_profile
from autode.values import Energy, PotentialEnergy, Enthalpy, FreeEnergy
from autode.utils import (
    work_in,
    work_in_tmp_dir,
    requires_hl_level_methods,
    checkpoint_rxn_profile_step,
)
from autode.reactions import reaction_types

if TYPE_CHECKING:
    from autode.species import Species
    from autode.bond_rearrangement import BondRearrangement


class Reaction:
    def __init__(
        self,
        *args: Union[str, "autode.species.species.Species"],
        name: str = "reaction",
        solvent_name: Optional[str] = None,
        smiles: Optional[str] = None,
        temp: float = 298.15,
    ):
        r"""
        Elementary chemical reaction formed from reactants and products.
        Number of atoms, charge and solvent must match on either side of
        the reaction. For example::

                            H                             H    H
                           /                               \  /
            H   +    H -- C -- H     --->     H--H   +      C
                          \                                 |
                           H                                H


        Arguments:
             args (autode.species.Species | str): Reactant and Product objects
                  or a SMILES string of the whole reaction.

            name (str): Name of this reaction.

            solvent_name (str | None): Name of the solvent, if None then
                                       in the gas phase (unless reactants and
                                       products are in a solvent).

            smiles (str | None): SMILES string of the reaction e.g.
                                 "C=CC=C.C=C>>C1=CCCCC1" for the [4+2]
                                 cyclization between ethene and butadiene.

            temp (float): Temperature in Kelvin.
        """
        logger.info(f"Generating a Reaction for {name}")

        self.name = name
        self.reacs, self.prods = [], []
        self._reactant_complex, self._product_complex = None, None
        self.tss = TransitionStates()

        # If there is only one string argument assume it's a SMILES
        if len(args) == 1 and type(args[0]) is str:
            smiles = args[0]

        if smiles is not None:
            self._init_from_smiles(smiles)
        else:
            self._init_from_molecules(molecules=args)

        self.type = reaction_types.classify(self.reacs, self.prods)
        self.solvent = get_solvent(solvent_name, kind="implicit")
        self.temp = float(temp)

        self._check_solvent()
        self._check_balance()
        self._check_names()

    def __str__(self):
        """Return a very short 6 character hash of the reaction, not guaranteed
        to be unique"""

        name = (
            f'{self.name}_{"+".join([r.name for r in self.reacs])}--'
            f'{"+".join([p.name for p in self.prods])}'
        )

        if hasattr(self, "solvent") and self.solvent is not None:
            name += f"_{self.solvent.name}"

        hasher = hashlib.sha1(name.encode()).digest()
        return base64.urlsafe_b64encode(hasher).decode()[:6]

    @requires_hl_level_methods
    def calculate_reaction_profile(
        self,
        units: Union["autode.units.Unit", str] = "kcal mol-1",
        with_complexes: bool = False,
        free_energy: bool = False,
        enthalpy: bool = False,
    ) -> None:
        """
        Calculate and plot a reaction profile for this elemtary reaction. Will
        search conformers, find the lowest energy TS and plot a profile.
        Calculations are performed in a new directory (self.name/)

        -----------------------------------------------------------------------
        Keyword Arguments:
            units (autode.units.Unit | str):

            with_complexes (bool): Calculate the lowest energy conformers
                                   of the reactant and product complexes

            free_energy (bool): Calculate the free energy profile (G)

            enthalpy (bool): Calculate the enthalpic profile (H)
        """
        logger.info("Calculating reaction profile")

        if not Config.allow_association_complex_G and (
            with_complexes and (free_energy or enthalpy)
        ):
            raise NotImplementedError(
                "Significant likelihood of very low "
                "frequency harmonic modes – G and H. Set"
                " Config.allow_association_complex_G to "
                "override this"
            )

        @work_in(self.name)
        def calculate(reaction):
            reaction.find_lowest_energy_conformers()
            reaction.optimise_reacs_prods()
            reaction.locate_transition_state()
            reaction.find_lowest_energy_ts_conformer()
            if with_complexes:
                reaction.calculate_complexes()
            if free_energy or enthalpy:
                reaction.calculate_thermochemical_cont()
            reaction.calculate_single_points()
            reaction.print_output()
            return None

        calculate(self)

        if not with_complexes:
            plot_reaction_profile(
                [self],
                units=units,
                name=self.name,
                free_energy=free_energy,
                enthalpy=enthalpy,
            )

        if with_complexes:
            self._plot_reaction_profile_with_complexes(
                units=units, free_energy=free_energy, enthalpy=enthalpy
            )
        return None

    def _check_balance(self) -> None:
        """Check that the number of atoms and charge balances between reactants
        and products. If they don't then raise excpetions
        """

        def total(molecules, attr):
            return sum([getattr(m, attr) for m in molecules])

        if total(self.reacs, "n_atoms") != total(self.prods, "n_atoms"):
            raise UnbalancedReaction("Number of atoms doesn't balance")

        if total(self.reacs, "charge") != total(self.prods, "charge"):
            raise UnbalancedReaction("Charge doesn't balance")

        # Ensure the number of unpaired electrons is equal on the left and
        # right-hand sides of the reaction, for now
        if total(self.reacs, "mult") - len(self.reacs) != total(
            self.prods, "mult"
        ) - len(self.prods):
            raise NotImplementedError(
                "Found a change in spin state – not " "implemented yet!"
            )

        self.charge = total(self.reacs, "charge")
        return None

    def _check_solvent(self) -> None:
        """
        Check that all the solvents are the same for reactants and products.
        If self.solvent is set then override the reactants and products
        """
        molecules = self.reacs + self.prods
        if len(molecules) == 0:
            return  # No molecules thus no solvent needs to be checked

        first_solvent = self.reacs[0].solvent

        if self.solvent is None:
            if all([mol.solvent is None for mol in molecules]):
                logger.info("Reaction is in the gas phase")
                return

            elif all([mol.solvent is not None for mol in molecules]):

                if not all(
                    [mol.solvent == first_solvent for mol in molecules]
                ):
                    raise SolventsDontMatch(
                        "Solvents in reactants and " "products do not match"
                    )
                else:
                    logger.info(f"Setting the solvent to {first_solvent}")
                    self.solvent = first_solvent

            else:
                raise SolventsDontMatch(
                    "Some species solvated and some not. "
                    "Ill-determined solvation."
                )

        if self.solvent is not None:
            logger.info(
                f"Setting solvent to {self.solvent.name} for all "
                f"molecules in the reaction"
            )
            for mol in molecules:
                mol.solvent = self.solvent

        logger.info(
            f"Set the solvent of all species in the reaction to "
            f"{self.solvent.name}"
        )
        return None

    def _check_names(self) -> None:
        """
        Ensure there is no clashing names of reactants and products, which will
        cause problems when conformers are generated and output is printed
        """
        all_names = [mol.name for mol in self.reacs + self.prods]

        if len(set(all_names)) == len(all_names):  # Everything is unique
            return

        logger.warning(
            "Names in reactants and products are not unique. "
            "Adding prefixes"
        )

        for i, reac in enumerate(self.reacs):
            reac.name = f"r{i}_{reac.name}"

        for i, prod in enumerate(self.prods):
            prod.name = f"p{i}_{prod.name}"

        return None

    def _init_from_smiles(self, reaction_smiles) -> None:
        """
        Initialise from a SMILES string of the whole reaction e.g.::

                    CC(C)=O.[C-]#N>>CC([O-])(C#N)C

        for the addition of cyanide to acetone.

        -----------------------------------------------------------------------
        Arguments:
            reaction_smiles (str):
        """
        try:
            reacs_smiles, prods_smiles = reaction_smiles.split(">>")
        except ValueError:
            raise UnbalancedReaction("Could not decompose to reacs & prods")

        # Add all the reactants and products with interpretable names
        for i, reac_smiles in enumerate(reacs_smiles.split(".")):
            reac = Reactant(smiles=reac_smiles)
            reac.name = f"r{i}_{reac.formula}"
            self.reacs.append(reac)

        for i, prod_smiles in enumerate(prods_smiles.split(".")):
            prod = Product(smiles=prod_smiles)
            prod.name = f"p{i}_{prod.formula}"
            self.prods.append(prod)

        return None

    def _init_from_molecules(self, molecules) -> None:
        """Set the reactants and products from a set of molecules"""

        self.reacs = [
            mol
            for mol in molecules
            if isinstance(mol, Reactant) or isinstance(mol, ReactantComplex)
        ]

        self.prods = [
            mol
            for mol in molecules
            if isinstance(mol, Product) or isinstance(mol, ProductComplex)
        ]

        return None

    def _components(self) -> Generator:
        """Components of this reaction"""

        for mol in (
            self.reacs
            + self.prods
            + [self.ts, self._reactant_complex, self._product_complex]
        ):
            yield mol

    def _reasonable_components_with_energy(self) -> Generator:
        """Generator for components of a reaction that have sensible geometries
        and also energies"""

        for mol in self._components():
            if mol is None:
                continue

            if mol.energy is None:
                logger.warning(f"{mol.name} energy was None")
                continue

            if not mol.has_reasonable_coordinates:
                continue

            yield mol

    def _estimated_barrierless_delta(self, e_type: str) -> Optional[Energy]:
        """
        Assume an effective free energy barrier = 4.35 kcal mol-1 calcd.
        from k = 4x10^9 at 298 K (doi: 10.1021/cr050205w). Must have a ∆G_r

        -----------------------------------------------------------------------
        Arguments:
            e_type (str): Type of energy to calculate: {'energy', 'enthalpy',
                                                        'free_energy'}
        Returns:
            (autode.values.Energy | None):
        """

        if self.delta(e_type) is None:
            logger.error(
                f"Could not estimate barrierless {e_type},"
                f" an energy was None"
            )
            return None

        # Minimum barrier is the 0 for an exothermic reaction but the reaction
        # energy for a endothermic reaction
        value = max(Energy(0.0), self.delta(e_type))

        if self.type != reaction_types.Rearrangement:
            logger.warning(
                "Have a barrierless bimolecular reaction. Assuming"
                "a diffusion limited with a rate of 4 x 10^9 s^-1"
            )

            value += Energy(0.00694, units="Ha")

        if e_type == "free_energy":
            return FreeEnergy(value, estimated=True)
        elif e_type == "enthalpy":
            return Enthalpy(value, estimated=True)
        else:
            return PotentialEnergy(value, estimated=True)

    def delta(self, delta_type: str) -> Optional[Energy]:
        """
        Energy difference for either reactants->TS or reactants -> products.
        Allows for equivelances "E‡" == "E ddagger" == "E double dagger" all
        return the potential energy barrier ∆E^‡. Can return None if energies
        of the reactants/products are None but will estimate for a TS (provided
        reactants and product energies are present). Example:

        .. code-block:: Python

            >>> import autode as ade
            >>> rxn = ade.Reaction(ade.Reactant(), ade.Product())
            >>> rxn.delta('E') is None
            True

        For reactants and products with energies:

        .. code-block:: Python

            >>> A = ade.Reactant()
            >>> A.energy = 1
            >>> B = ade.Product()
            >>> B.energy = 2
            >>>
            >>> rxn = ade.Reaction(A, B)
            >>> rxn.delta('E')
            Energy(1.0 Ha)

        Arguments:
            delta_type (str): Type of difference to calculate. Possibles:
                              {E, H, G, E‡, H‡, G‡}

        Returns:
            (autode.values.Energy | None): Difference if all energies are
                                          defined or None otherwise
        """

        def delta_type_matches(*args):
            return any(s in delta_type.lower() for s in args)

        def is_ts_delta():
            return delta_type_matches("ddagger", "‡", "double dagger")

        # Determine the species on the left and right hand sides of the eqn.
        lhs, rhs = self.reacs, [self.ts] if is_ts_delta() else self.prods

        # and the type of energy to calculate
        if delta_type_matches("h", "enthalpy"):
            e_type = "enthalpy"
        elif delta_type_matches("e", "energy") and not delta_type_matches(
            "free"
        ):
            e_type = "energy"
        elif delta_type_matches("g", "free energy", "free_energy"):
            e_type = "free_energy"
        else:
            raise ValueError(
                "Could not determine the type of energy change "
                f"to calculate from: {delta_type}"
            )

        # If there is no TS estimate the effective barrier from diffusion limit
        if is_ts_delta() and self.is_barrierless:
            return self._estimated_barrierless_delta(e_type)

        # If the electronic structure has failed to calculate the energy then
        # the difference between the left and right cannot be calculated
        if any(getattr(mol, e_type) is None for mol in lhs + rhs):
            logger.warning(
                f"Could not calculate ∆{delta_type}, an energy was " f"None"
            )
            return None

        return sum(getattr(mol, e_type).to("Ha") for mol in rhs) - sum(
            getattr(mol, e_type).to("Ha") for mol in lhs
        )

    @property
    def is_barrierless(self) -> bool:
        """
        Is this reaction barrierless? i.e. without a barrier either because
        there is no enthalpic barrier to the reaction, or because a TS cannot
        be located.

        -----------------------------------------------------------------------
        Returns:
            (bool): If this reaction has a barrier
        """
        return self.ts is None

    @property
    def reactant(self) -> ReactantComplex:
        """
        Reactant complex comprising all the reactants in this reaction

        -----------------------------------------------------------------------
        Returns:
            (autode.species.ReactantComplex): Reactant complex
        """
        if self._reactant_complex is not None:
            return self._reactant_complex

        return ReactantComplex(
            *self.reacs, name=f"{self}_reactant", do_init_translation=True
        )

    @reactant.setter
    def reactant(self, value: ReactantComplex):
        """
        Set the reactant of this reaction. If unset then will use a generated
        complex of all reactants

        -----------------------------------------------------------------------
        Arguments:
            value (autode.species.ReactantComplex):
        """
        if not isinstance(value, ReactantComplex):
            raise ValueError(
                f"Could not set the reactant of {self.name} "
                f"using {type(value)}. Must be a ReactantComplex"
            )

        self._reactant_complex = value

    @property
    def product(self) -> ProductComplex:
        """
        Product complex comprising all the products in this reaction

        -----------------------------------------------------------------------
        Returns:
            (autode.species.ProductComplex): Product complex
        """
        if self._product_complex is not None:
            return self._product_complex

        return ProductComplex(
            *self.prods, name=f"{self}_product", do_init_translation=True
        )

    @product.setter
    def product(self, value: ProductComplex):
        """
        Set the product of this reaction. If unset then will use a generated
        complex of all products

        -----------------------------------------------------------------------
        Arguments:
            value (autode.species.ProductComplex):
        """
        if not isinstance(value, ProductComplex):
            raise ValueError(
                f"Could not set the product of {self.name} "
                f"using {type(value)}. Must be a ProductComplex"
            )

        self._product_complex = value

    @property
    def ts(self) -> Optional[TransitionState]:
        """
        _The_ transition state for this reaction. If there are multiple then
        return the lowest energy but if there are no transtion states then
        return None

        -----------------------------------------------------------------------
        Returns:
            (autode.transition_states.TransitionState | None):
        """
        return self.tss.lowest_energy

    @ts.setter
    def ts(self, value: Optional[TransitionState]):
        """
        Set the TS of this reaction, will override any other transition states
        located.

        -----------------------------------------------------------------------
        Arguments:
            value (autode.transition_states.TransitionState | None):
        """
        self.tss.clear()

        if value is None:
            return

        if not isinstance(value, TransitionState):
            raise ValueError(f"TS of {self.name} must be a TransitionState")

        self.tss.append(value)

    def switch_reactants_products(self) -> None:
        """Addition reactions are hard to find the TSs for, so swap reactants
        and products and classify as dissociation. Likewise for reactions wher
        the change in the number of bonds is negative
        """
        logger.info("Swapping reactants and products")

        self.prods, self.reacs = self.reacs, self.prods

        (self._product_complex, self._reactant_complex) = (
            self._reactant_complex,
            self._product_complex,
        )
        return None

    @checkpoint_rxn_profile_step("reactant_product_conformers")
    def find_lowest_energy_conformers(self) -> None:
        """Try and locate the lowest energy conformation using simulated
        annealing, then optimise them with xtb, then optimise the unique
        (defined by an energy cut-off) conformers with an electronic structure
        method"""

        h_method = get_hmethod() if Config.hmethod_conformers else None
        for mol in self.reacs + self.prods:
            # .find_lowest_energy_conformer works in conformers/
            mol.find_lowest_energy_conformer(hmethod=h_method)

        return None

    @checkpoint_rxn_profile_step("reactants_and_products")
    @work_in("reactants_and_products")
    def optimise_reacs_prods(self) -> None:
        """Perform a geometry optimisation on all the reactants and products
        using the method"""
        h_method = get_hmethod()
        logger.info(f"Optimising reactants and products with {h_method.name}")

        for mol in self.reacs + self.prods:
            mol.optimise(h_method)

        return None

    @checkpoint_rxn_profile_step("complexes")
    @work_in("complexes")
    def calculate_complexes(self) -> None:
        """Find the lowest energy conformers of reactant and product complexes
        using optimisation and single points"""
        h_method = get_hmethod()
        conf_hmethod = h_method if Config.hmethod_conformers else None

        self._reactant_complex = ReactantComplex(
            *self.reacs, name=f"{self}_reactant", do_init_translation=True
        )

        self._product_complex = ProductComplex(
            *self.prods, name=f"{self}_product", do_init_translation=True
        )

        for species in [self._reactant_complex, self._product_complex]:
            species.find_lowest_energy_conformer(hmethod=conf_hmethod)
            species.optimise(method=h_method)

        return None

    @requires_hl_level_methods
    @checkpoint_rxn_profile_step("transition_states")
    @work_in("transition_states")
    def locate_transition_state(self) -> None:

        if self.type is None:
            raise RuntimeError(
                "Cannot invoke locate_transition_state without a reaction type"
            )

        # If there are more bonds in the product e.g. an addition reaction then
        # switch as the TS is then easier to find
        if sum(p.graph.number_of_edges() for p in self.prods) > sum(
            r.graph.number_of_edges() for r in self.reacs
        ):
            self.switch_reactants_products()
            self.tss = find_tss(self)
            self.switch_reactants_products()
        else:
            self.tss = find_tss(self)

        return None

    @work_in_tmp_dir()
    def _get_mapped_aligned_geometries(
        self, react_orient: bool
    ) -> List[tuple]:
        """
        Performs tha mapping and alignment of reactant and product
        complexes and returns the most optimal pair based on RMSD.
        If the reactant or product are complexes, then random
        orientations are generated, and then minimised at low-level.
        (For the reactant it can be over-ridden to use an orientation
        guessed from the bond rearrangment with the argument react_orient)

        Args:
            react_orient: If True, the reactant complex orientation
                          is set according to the bond rearrangement

        Returns:
            (list[tuple[Species]]): List of tuples of aligned reactant/product pairs
        """
        possible_reacs_prods = []

        # need to optimise at least at low-level for reliable geometries
        lmethod = get_lmethod()
        for mol in self.reacs + self.prods:
            mol.optimise(method=lmethod)

        # get copies so that originals are not modified
        if sum(p.graph.number_of_edges() for p in self.prods) > sum(
            r.graph.number_of_edges() for r in self.reacs
        ):
            logger.warning("Swtiching reactants and products")
            rct_complex = self.product.copy()
            prod_complex = self.reactant.copy()
        else:
            rct_complex = self.reactant.copy()
            prod_complex = self.product.copy()

        # go through all possible bond rearrangements
        bond_rearrs = get_bond_rearrangs(
            reactant=rct_complex,
            product=prod_complex,
            name=f"{self.name}_ext",
            save=False,
        )
        if bond_rearrs is None:
            raise RuntimeError("Unable to find any bond rearrangements")

        # generate conformers if more than two items and optimise
        if prod_complex.n_molecules > 1:
            prod_complex.populate_conformers()
            prod_complex.conformers.prune(
                e_tol=Energy(1e-6, "Ha"), rmsd_tol=0.1, remove_no_energy=True
            )
        else:
            prod_complex.conformers.append(prod_complex.copy())
        # for reactant, optimal orientation is probably better
        if not react_orient:
            rct_complex.populate_conformers()
            rct_complex.conformers.prune(
                e_tol=Energy(1e-6, "Ha"), rmsd_tol=0.1, remove_no_energy=True
            )

        for idx, bond_rearr in enumerate(bond_rearrs):
            assert bond_rearr.n_bbonds >= bond_rearr.n_fbonds

            if react_orient:
                # take copies as the orientations might be different
                # for different bond rearrangements
                rct_oriented = rct_complex.copy()

                # get optimal orientation
                translate_rotate_reactant(
                    reactant=rct_oriented,
                    bond_rearrangement=bond_rearr,
                    shift_factor=1.5 if rct_oriented.charge == 0 else 2.5,
                )
                # conformer is then just a copy of the chosen orientation
                rct_oriented.conformers.append(rct_oriented.copy())
            else:
                # no need to orient, just assign
                rct_oriented = rct_complex

            try:

                (
                    rct_mapped,
                    prod_mapped,
                ) = align_product_to_reactant_by_symmetry_rmsd(
                    prod_complex, rct_oriented, bond_rearr
                )

                possible_reacs_prods.append((rct_mapped, prod_mapped))

            except NoMapping:
                logger.error(
                    f"Unable to atom map for bond rearrangement"
                    f" {str(bond_rearr)}"
                )

        return possible_reacs_prods

    def print_mapped_xyz_geometries(
        self, use_reactive_orientation: bool = True
    ):
        """
        Print the atom-mapped xyz geometries that can be fed into
        other external programs, or used for double ended TS search
        like GSM. Requires at least one available low-level method.
        Will modify reactant and product geometries.

        Args:
            use_reactive_orientation (bool): In case there are multiple
                                reactants, whether to use the optimal
                                orientation of the complex estimated from
                                the bond rearrangement. If False, will
                                minimise RMSD against optimised geometries
                                from random orientations
        """
        reacs_prods = self._get_mapped_aligned_geometries(
            use_reactive_orientation
        )

        if len(reacs_prods) == 0:
            raise RuntimeError("No suitable reactant -> product graph mapping")

        for idx, pair in enumerate(reacs_prods):
            pair[0].print_xyz_file(
                filename=f"{self.name}_reactant_ext_{idx}.xyz"
            )
            pair[1].print_xyz_file(
                filename=f"{self.name}_product_ext_{idx}.xyz"
            )
        return None

    @checkpoint_rxn_profile_step("transition_state_conformers")
    @work_in("transition_states")
    def find_lowest_energy_ts_conformer(self) -> None:
        """Find the lowest energy conformer of the transition state"""
        if self.ts is None:
            logger.error("No transition state to evaluate the conformer of")
            return None

        else:
            return self.ts.find_lowest_energy_ts_conformer()

    @checkpoint_rxn_profile_step("single_points")
    @work_in("single_points")
    def calculate_single_points(self) -> None:
        """Perform a single point energy evaluations on all the reactants and
        products using the hmethod"""
        h_method = get_hmethod()
        logger.info(f"Calculating single points with {h_method.name}")

        for mol in self._reasonable_components_with_energy():
            mol.single_point(h_method)

        return None

    @work_in("output")
    def print_output(self) -> None:
        """Print the final optimised structures along with the methods used"""
        from autode.log.methods import methods

        # Print the computational methods used in this autode initialisation
        with open("methods.txt", "w") as out_file:
            print(methods, file=out_file)

        csv_file = open("energies.csv", "w")
        method = get_hmethod()
        print(
            f"Energies generated by autodE on: {date.today()}. Single point "
            f"energies at {method.keywords.sp.bstring} and optimisations at "
            f"{method.keywords.opt.bstring}",
            "Species, E_opt, G_cont, H_cont, E_sp",
            sep="\n",
            file=csv_file,
        )

        def print_energies_to_csv(_mol):
            print(
                f"{_mol.name}",
                f"{_mol.energies.first_potential}",
                f"{_mol.g_cont}",
                f"{_mol.h_cont}",
                f"{_mol.energies.last_potential}",
                sep=",",
                file=csv_file,
            )

        # Print xyz files of all the reactants and products
        for mol in self.reacs + self.prods:
            mol.print_xyz_file()
            print_energies_to_csv(mol)

        # and the reactant and product complexes if they're present
        for mol in [self._reactant_complex, self._product_complex]:
            if mol is not None and mol.energy is not None:
                mol.print_xyz_file()
                print_energies_to_csv(mol)

        # If it exists print the xyz file of the transition state
        if self.ts is not None:
            ts_title_str = ""
            imags = self.ts.imaginary_frequencies

            if self.ts.has_imaginary_frequencies and len(imags) > 0:
                ts_title_str += f". Imaginary frequency = {imags[0]:.1f} cm-1"

            if self.ts.has_imaginary_frequencies and len(imags) > 1:
                ts_title_str += (
                    f". Additional imaginary frequencies: " f"{imags[1:]} cm-1"
                )

            print_energies_to_csv(self.ts)
            self.ts.print_xyz_file(additional_title_line=ts_title_str)
            self.ts.print_imag_vector(name="TS_imag_mode")

        return None

    @checkpoint_rxn_profile_step("thermal")
    @work_in("thermal")
    def calculate_thermochemical_cont(
        self, free_energy: bool = True, enthalpy: bool = True
    ) -> None:
        """
        Calculate thermochemical contributions to the energies

        -----------------------------------------------------------------------
        Arguments
            free_energy (bool):

            enthalpy (bool):
        """
        logger.info("Calculating thermochemical contributions")

        if not (free_energy or enthalpy):
            logger.info("Nothing to be done – neither G or H requested")
            return None

        # Calculate G and H contributions for all components
        for mol in self._reasonable_components_with_energy():
            mol.calc_thermo(temp=self.temp)

        return None

    def _plot_reaction_profile_with_complexes(
        self, units: "autode.units.Unit", free_energy: bool, enthalpy: bool
    ) -> None:
        """Plot a reaction profile with the association complexes of R, P"""
        rxns = []

        if any(mol.energy is None for mol in (self.reactant, self.product)):
            raise ValueError(
                "Could not plot a reaction profile with "
                "association complexes without energies for"
                "reaction.reactant_complex or product_complex"
            )

        # If the reactant complex contains more than one molecule then
        # make a reaction that is separated reactants -> reactant complex
        if len(self.reacs) > 1:
            rxns.append(
                Reaction(
                    *self.reacs,
                    self.reactant.to_product_complex(),
                    name="reactant_complex",
                )
            )

        # The elementary reaction is then
        # reactant complex -> product complex
        reaction = Reaction(self.reactant, self.product)
        reaction.ts = self.ts
        rxns.append(reaction)

        # As with the product complex add the dissociation of the product
        # complex into it's separated components
        if len(self.prods) > 1:
            rxns.append(
                Reaction(
                    *self.prods,
                    self.product.to_reactant_complex(),
                    name="product_complex",
                )
            )

        plot_reaction_profile(
            reactions=rxns,
            units=units,
            name=self.name,
            free_energy=free_energy,
            enthalpy=enthalpy,
        )
        return None

    @property
    def atomic_symbols(self) -> List[str]:
        """
        Atomic symbols of all atoms in this reaction sorted alphabetically.
        For example:

        .. code-block::

            >>> from autode import Atom, Reactant, Product, Reaction
            >>>rxn = Reaction(Reactant(smiles='O'),
                              Product(atoms=[Atom('O'), Atom('H', x=0.9)]),
                              Product(atoms=[Atom('H')]))
            >>> rxn.atomic_symbols
            ['H', 'H', 'O']

        -----------------------------------------------------------------------
        Returns:
            (list(str)): List of all atoms in this reaction, with duplicates
        """

        all_atomic_symbols = []
        for reactant in self.reacs:
            all_atomic_symbols += reactant.atomic_symbols

        return list(sorted(all_atomic_symbols))

    def has_identical_composition_as(self, reaction: "Reaction") -> bool:
        """Does this reaction have the same chemical identity as another?"""
        return self.atomic_symbols == reaction.atomic_symbols

    def save(self, filepath: str) -> None:
        """Save the state of this reaction to a binary file that can be reloaded"""

        with open(filepath, "wb") as file:
            pickle.dump(self.__dict__, file)

    def load(self, filepath: str) -> None:
        """Load a reaction state from a binary file"""

        with open(filepath, "rb") as file:
            for attr, value in dict(pickle.load(file)).items():
                setattr(self, attr, value)

    @classmethod
    def from_checkpoint(cls, filepath: str) -> "Reaction":
        """Create a reaction from a checkpoint file"""
        logger.info(f"Loading a reaction object from {filepath}")
        rxn = cls()
        rxn.load(filepath)
        return rxn


def _get_heavy_and_active_h_atom_indices(
    reactant: "Species", bond_rearr: "BondRearrangement"
) -> List[int]:
    """
    Obtain all the heavy atoms, and only those hydrogens that are
    involved in the reaction, informed by the bond rearrangement

    Args:
        reactant (Species): The reactant species
        bond_rearr (BondRearrangement): The bond rearrangement from reactant
                                        to form product

    Returns:
        (list[int]): The indices of the heavy atoms, and the active hydrogens
    """
    selected_atoms = set()
    for idx, atom in enumerate(reactant.atoms):
        if atom.label != "H":
            selected_atoms.add(idx)

    for pairs in bond_rearr.all:
        selected_atoms.update(pairs)

    return list(selected_atoms)


def align_product_to_reactant_by_symmetry_rmsd(
    product_complex: ProductComplex,
    reactant_complex: ReactantComplex,
    bond_rearr,
    max_trials: int = 50,
) -> Tuple["Species"]:
    """
    Aligns a product complex against a reactant complex for a given
    bond rearrangement by iterating through all possible graph
    isomorphisms for the heavy atoms (and any active H), and then
    checking the RMSD on those selected atoms only. Non-active hydrogens
    are not checked due to combinatorial explosion that makes this
    difficult. Assumes that the product complex and reactant complex
    have at least one conformer (orientation of one molecule against
    another, in case there is only one molecule in complex, conformer
    must be a copy of the current geometry)

    Args:
        product_complex: The product complex (must have at least one conf)
        reactant_complex: The reactant complex (must have at least one conf)
        bond_rearr: The bond rearrangement
        max_trials: Maximum number of heavy-atom isomorphisms to check against

    Returns:

    """
    from autode.geom import calc_rmsd
    from networkx.algorithms import isomorphism
    from autode.mol_graphs import reac_graph_to_prod_graph
    from autode.exceptions import NoMapping

    # extract only heavy atoms and any hydrogens involved in reaction
    fit_atom_idxs = _get_heavy_and_active_h_atom_indices(
        reactant_complex, bond_rearr
    )

    # NOTE: multiple reaction mappings possible due to symmetry, e.g., 3 hydrogens
    # in -CH3 are symmetric/equivalent in 2D. We choose unique mappings where the
    # mapping of heavy atoms (or hydrogens involved in reaction) change, to reduce
    # the complexity of the problem.
    node_match = isomorphism.categorical_node_match("atom_label", "C")
    gm = isomorphism.GraphMatcher(
        reac_graph_to_prod_graph(reactant_complex.graph, bond_rearr),
        product_complex.graph,
        node_match=node_match,
    )
    assert len(product_complex.conformers) > 0, "Must have conformer(s)"

    unique_mappings = []

    def is_mapping_unique(full_map):
        for this_map in unique_mappings:
            if all(this_map[i] == full_map[i] for i in fit_atom_idxs):
                return False
        return True

    # collect at most <max_trials> number of unique mappings
    for mapping in gm.isomorphisms_iter():
        if is_mapping_unique(mapping):
            unique_mappings.append(mapping)
        if len(unique_mappings) > max_trials:
            break
    # todo check the logic of this function
    lowest_rmsd, aligned_rct, aligned_prod = None, None, None

    for mapping in unique_mappings:

        sorted_mapping = {i: mapping[i] for i in sorted(mapping)}

        for rct_conf in reactant_complex.conformers:
            # take copy to avoid permanent reordering
            rct_tmp = rct_conf.copy()
            rct_tmp.reorder_atoms(sorted_mapping)
            # TODO: get RMSD by chosen atoms only

            for conf in product_complex.conformers:
                rmsd = calc_rmsd(conf.coordinates, rct_tmp.coordinates)
                if lowest_rmsd is None or rmsd < lowest_rmsd:
                    lowest_rmsd, aligned_rct, aligned_prod = (
                        rmsd,
                        rct_tmp,
                        conf.copy(),
                    )

    logger.info(f"Lowest heavy-atom RMSD of fit = {lowest_rmsd}")

    if aligned_rct is None:
        raise NoMapping("Unable to obtain isomorphism mapping for heavy atom")

    # TODO: then align according to heavy atoms, and use Hungarian algorithm to
    # deal with hydrogens
    _align_species(aligned_rct, aligned_prod)

    return aligned_rct, aligned_prod


def _align_species(first_species: "Species", second_species: "Species"):
    import numpy as np
    from autode.geom import get_rot_mat_kabsch

    logger.info("Aligning species by translation and rotation")
    # first translate the molecules to the origin
    p_mat = first_species.coordinates.copy()
    p_mat -= np.average(p_mat, axis=0)
    first_species.coordinates = p_mat

    q_mat = second_species.coordinates.copy()
    q_mat -= np.average(q_mat, axis=0)
    second_species.coordinates = q_mat

    logger.info(
        "Rotating initial_species (reactant) "
        "to align with final_species (product) "
        "as much as possible"
    )
    rot_mat = get_rot_mat_kabsch(p_mat, q_mat)
    rotated_p_mat = np.dot(rot_mat, p_mat.T).T
    first_species.coordinates = rotated_p_mat
    return None
