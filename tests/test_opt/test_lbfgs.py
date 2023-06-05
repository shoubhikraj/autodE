import os
import numpy as np
from autode.opt.optimisers import LBFGSOptimiser
from autode import Molecule
from autode.opt.coordinates import CartesianCoordinates
from autode.methods import XTB
from autode.geom import calc_rmsd
from autode.values import Energy
from ..testutils import requires_with_working_xtb_install, work_in_zipped_dir

here = os.path.dirname(os.path.abspath(__file__))


def test_lbfgs_trust_update():
    init_alpha = 0.2
    mol = Molecule(smiles="N#N")
    opt = LBFGSOptimiser(
        maxiter=10,
        gtol=1e-3,
        etol=1e-4,
        init_alpha=init_alpha,
        max_e_rise=Energy(0.004, "Ha"),
    )
    opt._initialise_species_and_method(mol, XTB())
    coords = CartesianCoordinates(mol.coordinates)

    def dummy_energy_series_trust_update(energy_list):
        """
        Generate dummy coordinates with energies and put them in
        optimiser and update trust radius
        """
        opt._history.clear()
        for energy in energy_list:
            new_coords = coords.copy()
            new_coords.e = Energy(energy, "Ha")
            opt._history.append(new_coords)
            opt._update_trust_radius()

    # increase in energy larger than max_e_rise
    dummy_energy_series_trust_update([-15.116, -15.111])
    # last step should be removed
    assert len(opt._history) == 1
    assert np.isclose(opt.alpha, init_alpha / 4, rtol=1e-8)

    # reset alpha
    opt.alpha = init_alpha
    # increase in energy within max_e_rise
    dummy_energy_series_trust_update([-15.116, -15.114])
    assert len(opt._history) == 2
    assert np.isclose(opt.alpha, init_alpha / 1.5, rtol=1e-8)

    # trust radius is increased when energy goes down for 4 iterations
    # and the last step is same size as trust radius
    opt.alpha = init_alpha
    # three steps
    dummy_energy_series_trust_update([-15.111, -15.112, -15.113, -15.114])
    # dummy step of size trust radius
    opt._coords = coords.copy() - init_alpha / np.sqrt(len(coords))
    opt._coords.e = Energy(-15.115, "Ha")
    assert opt._n_e_decrease == 3
    opt._update_trust_radius()
    assert np.isclose(opt.alpha, init_alpha * 1.2, rtol=1e-8)


@work_in_zipped_dir(os.path.join(here, "data", "opt.zip"))
@requires_with_working_xtb_install
def test_lbfgs_small_molecule_opt():
    mol = Molecule("opt-test_dih_rot.xyz")
    mol2 = mol.copy()
    opt = LBFGSOptimiser(maxiter=100, gtol=5e-4, etol=1e-4, max_vecs=15)
    opt.run(mol, method=XTB())
    # optimise with xTB's internal one
    mol2.optimise(method=XTB())

    rmsd = calc_rmsd(mol.coordinates, mol2.coordinates)
    assert rmsd < 0.01
