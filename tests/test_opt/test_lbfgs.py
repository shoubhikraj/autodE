import os
from autode.opt.optimisers import LBFGSOptimiser
from autode import Molecule
from autode.methods import XTB
from autode.geom import calc_rmsd
from ..testutils import requires_with_working_xtb_install, work_in_zipped_dir

here = os.path.dirname(os.path.abspath(__file__))


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
