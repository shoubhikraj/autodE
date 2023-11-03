import numpy as np
from autode.opt.optimisers.cartesian_driving import DrivenDistances
from autode.opt.coordinates.primitives import PrimitiveDistance
from autode.opt.coordinates.internals import AnyPIC
from autode import Molecule


def test_driving_coords_gradients():
    dists = DrivenDistances([(1, 2)], [1.0])
    mol = Molecule(smiles="CCO")
    x = np.array(mol.coordinates.flatten())
    delta = 0.0001
    num_grad = np.zeros_like(x)

    for idx in range(len(x)):
        x_new = x.copy()
        x_new[idx] += delta
        num_grad[idx] = (dists(x_new) - dists(x)) / delta

    calc_grad = dists.derivative(x)

    assert np.allclose(num_grad, calc_grad, rtol=1e-3)
