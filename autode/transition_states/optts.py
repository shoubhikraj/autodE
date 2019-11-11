from copy import deepcopy
import numpy as np
from autode.config import Config
from autode.log import logger
from autode.transition_states.transition_state import TS
from autode.atoms import get_atomic_weight
from autode.molecule import Molecule
from autode.calculation import Calculation
from autode.mol_graphs import is_isomorphic
from autode.methods import get_hmethod
from autode.methods import get_lmethod


def get_ts(ts_guess, imag_freq_threshold=-100):
    """
    Get a transition object from a set of xyzs by running an ORCA OptTS calculation
    :param ts_guess: (object) TSguess object
    :param imag_freq_threshold: (float) Imaginary frequency threshold for a *true* TS, given as as a negative
    i.e. non-complex value
    :return:
    """

    if ts_guess is None:
        logger.warning('Cannot find a transition state; had no TS guess')
        return None

    if not ts_guess.run_orca_optts():
        return None

    imag_freqs, _, _ = ts_guess.get_imag_frequencies_xyzs_energy()

    # check all imag modes
    correct_mode = None
    for i in range(len(imag_freqs)):
        mode = i + 6
        logger.info(
            f'Checking to see if imag mode {i} has the correct vector')
        if ts_has_correct_imaginary_vector(ts_guess.optts_calc, n_atoms=len(ts_guess.xyzs), active_bonds=ts_guess.active_bonds, molecules=(ts_guess.reactant, ts_guess.product), mode_number=mode):
            correct_mode = i
            break

    if correct_mode is None:
        return None

    if len(imag_freqs) > 1:
        logger.warning(
            f'OptTS calculation returned {len(imag_freqs)} imaginary frequencies')
        if not ts_guess.do_displacements(correct_mode):
            return None

    if not ts_guess.check_optts_convergence():
        return None

    if ts_guess.optts_converged or ts_guess.optts_nearly_converged:
        imag_freqs, _, _ = ts_guess.get_imag_frequencies_xyzs_energy()
        if len(imag_freqs) > 0:

            if imag_freqs[0] > imag_freq_threshold:
                logger.warning(
                    f'Probably haven\'t found the correct TS {imag_freqs[0]} > {imag_freq_threshold} cm-1')
                return None

            if len(imag_freqs) == 1:
                logger.info('Found TS with 1 imaginary frequency')

            if ts_has_correct_imaginary_vector(ts_guess.optts_calc, n_atoms=len(ts_guess.xyzs),
                                               active_bonds=ts_guess.active_bonds, molecules=(ts_guess.reactant, ts_guess.product)):

                ts_guess.pi_bonds = ts_guess.optts_calc.get_pi_bonds()

                if ts_guess.optts_converged:
                    return TS(ts_guess)

                if ts_guess.optts_nearly_converged:
                    return TS(ts_guess, converged=False)
        else:
            logger.warning('TS has *0* imaginary frequencies')

    return None


def get_displaced_xyzs_along_imaginary_mode(calc, mode_number=7, displacement_magnitude=1.0):
    """
    Displace the geometry along the imaginary mode with mode number iterating from 0, where 0-2 are translational
    normal modes, 3-5 are rotational modes and 6 is the largest imaginary mode. To displace along the second imaginary
    mode we have mode_number=7
    :param calc: (object)
    :param mode_number: (int)
    :param displacement_magnitude: (float)
    :return: (list)
    """
    logger.info('Displacing along imaginary mode')

    current_xyzs = calc.get_final_xyzs()
    n_atoms = len(current_xyzs)
    mode_distplacement_coords = calc.get_normal_mode_displacements(
        mode_number=mode_number)

    displaced_xyzs = deepcopy(current_xyzs)
    for i in range(n_atoms):
        for j in range(3):
            displaced_xyzs[i][j+1] += displacement_magnitude * \
                mode_distplacement_coords[i][j]      # adding coord (nx3)
    #                                                                                               # to xyzs (nx4)
    return displaced_xyzs


def ts_has_correct_imaginary_vector(calc, n_atoms, active_bonds, molecules=None, threshold_contribution=0.25, mode_number=6):
    """
    For an orca output file check that the first imaginary mode (number 6) in the final frequency calculation
    contains the correct motion, i.e. contributes more than threshold_contribution in relative terms to the
    magnitude of the sum of the forces

    :param calc: (object)
    :param n_atoms: (int) number of atoms
    :param active_bonds: (list(tuples))
    :param threshold_contribution: (float) threshold contribution to the imaginary mode from the atoms in
    bond_ids_to_add
    :param mode_numer: (int) which normal mode to check
    :return:
    """

    logger.info(
        f'Checking the active atoms contribute more than {threshold_contribution} to the imag mode')

    if active_bonds is None:
        logger.info('Cannot determine whether the correct atoms contribute')
        return True

    imag_normal_mode_displacements_xyz = calc.get_normal_mode_displacements(
        mode_number=mode_number)
    if imag_normal_mode_displacements_xyz is None:
        logger.error('Have no imaginary normal mode displacements to analyse')
        return False

    final_xyzs = calc.get_final_xyzs()

    imag_mode_magnitudes = [np.linalg.norm(
        np.array(dis_xyz)) for dis_xyz in imag_normal_mode_displacements_xyz]

    weighted_imag_mode_magnitudes = []

    for index, atom in enumerate(final_xyzs):
        atom_label = atom[0]
        weighting = get_atomic_weight(atom_label) + 10
        weighted_imag_mode_magnitudes.append(
            (imag_mode_magnitudes[index] * weighting))

    should_be_active_atom_magnitudes = []
    for atom_id in range(n_atoms):
        if any([atom_id in bond_ids for bond_ids in active_bonds]):
            should_be_active_atom_magnitudes.append(
                weighted_imag_mode_magnitudes[atom_id])

    relative_contribution = np.sum(np.array(
        should_be_active_atom_magnitudes)) / np.sum(np.array(weighted_imag_mode_magnitudes))

    if molecules is not None:
        if threshold_contribution - 0.1 < relative_contribution < threshold_contribution + 0.1:
            logger.info(
                f'Unsure if significant contribution from active atoms to imag mode (contribution = {relative_contribution:.3f}). Displacing along imag modes to check')
            if check_close_imag_contribution(calc, molecules, method=get_lmethod(), mode_number=mode_number):
                logger.info(
                    'Imaginary mode links reactants and products, TS found')
                return True
            logger.info(
                'Lower level method didn\'t find link, trying higher level of theory')
            if check_close_imag_contribution(calc, molecules, method=get_hmethod(), mode_number=mode_number):
                logger.info(
                    'Imaginary mode links reactants and products, TS found')
                return True
            logger.info(
                'Imaginary mode doesn\'t link reactants and products, TS *not* found')
            return False

    if relative_contribution > threshold_contribution:
        logger.info(
            f'TS has significant contribution from the active atoms to the imag mode (contribution = {relative_contribution:.3f})')
        return True

    logger.info(
        f'TS has *no* significant contribution from the active atoms to the imag mode (contribution = {relative_contribution:.3f})')
    return False


def check_close_imag_contribution(calc, molecules, method, mode_number=6, disp_mag=0.8):
    """Displaced atoms along the imaginary mode to see if products and reactants are made

    Arguments:
        calc {calculation obj} -- 
        molecules {tuple} -- tuple containing the reactant and product objects
        method {electronic structure method} -- 


    Keyword Arguments:
        disp_mag {int} -- Distance to be displaced along the imag mode (default: {0.8})
        mode_number {int} -- normal mode number to be checked (default: {6})

    Returns:
        {bool} -- if the imag mode is correct or not
    """
    forward_displaced_xyzs = get_displaced_xyzs_along_imaginary_mode(
        calc, mode_number=mode_number, displacement_magnitude=disp_mag)
    forward_displaced_mol = Molecule(xyzs=forward_displaced_xyzs)
    forward_displaced_calc = Calculation(name=calc.name + f'_{mode_number}_forwards_displacement', molecule=forward_displaced_mol, method=method,
                                         keywords=method.opt_keywords, n_cores=Config.n_cores,
                                         max_core_mb=Config.max_core, opt=True)
    forward_displaced_calc.run()
    forward_displaced_mol.set_xyzs(forward_displaced_calc.get_final_xyzs())

    backward_displaced_xyzs = get_displaced_xyzs_along_imaginary_mode(
        calc, mode_number=mode_number, displacement_magnitude=-disp_mag)
    backward_displaced_mol = Molecule(xyzs=backward_displaced_xyzs)
    backward_displaced_calc = Calculation(name=calc.name + f'_{mode_number}_backwards_displacement', molecule=backward_displaced_mol, method=method,
                                          keywords=method.opt_keywords, n_cores=Config.n_cores,
                                          max_core_mb=Config.max_core, opt=True)
    backward_displaced_calc.run()
    backward_displaced_mol.set_xyzs(backward_displaced_calc.get_final_xyzs())

    if is_isomorphic(forward_displaced_mol.graph, molecules[0].graph):
        if is_isomorphic(backward_displaced_mol.graph, molecules[1].graph):
            return True
    if is_isomorphic(backward_displaced_mol.graph, molecules[0].graph):
        if is_isomorphic(forward_displaced_mol.graph, molecules[1].graph):
            return True
    return False
