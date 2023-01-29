from typing import Optional, List
import numpy as np

from autode.values import Distance
from autode.opt.coordinates import OptCoordinates
from autode.utils import work_in_tmp_dir

import autode.species.species
import autode.wrappers.methods


def _calculate_engrad_for_species(
    species: autode.species.species.Species,
    method: autode.wrappers.methods.Method,
    n_cores: int,
):
    from autode.calculations import Calculation

    engrad_calc = Calculation(
        name=f"{species.name}_engrad",
        molecule=species,
        method=method,
        keywords=method.keywords.grad,
        n_cores=n_cores,
    )
    engrad_calc.run()
    engrad_calc.clean_up(force=True, everything=True)
    return species.energy.to("Ha"), species.gradient.to("Ha/ang")


@work_in_tmp_dir()
def _calculate_hessian_for_species(
    species, method: "autode.wrappers.methods.Method", n_cores
):
    """
    Convenience function for calculating the Hessian for a
    molecule; removes all input and output files for the
    calculation
    """
    from autode.calculations import Calculation

    hess_calc = Calculation(
        name=f"{species.name}_hess",
        molecule=species,
        method=method,
        keywords=method.keywords.hess,
        n_cores=n_cores,
    )
    hess_calc.run()
    hess_calc.clean_up(force=True, everything=True)
    return species.hessian.to("ha/ang^2")


class ConstrainedImagePair:
    """
    A pair of images (reactant and product) constrined by
    distance (d = d_i) and energy (E_1 = E_2) criteria.
    The constraints are enforced by Lagrangian multipliers
    """

    def __init__(
        self,
        left_image: autode.species.species.Species,
        right_image: autode.species.species.Species,
        constrain_energy: bool = True,
    ):
        """
        Initialize the constrained image pair. Does not initialize
        methods!

        Args:
            left_image: One molecule of the pair
            right_image: Another molecule of the pair
            constrain_energy (bool): Are energy constrained applied
        """
        assert isinstance(left_image, autode.species.species.Species)
        assert isinstance(right_image, autode.species.species.Species)
        self._left_image = left_image.new_species(name="left_image")
        self._right_image = right_image.new_species(name="right_image")
        # todo sanity check that two images have same charge etc. and they are not
        # too close
        self._is_e_constr = bool(constrain_energy)

        self._engrad_method = None
        self._hess_method = None
        self._n_cores = None

        # C_E = (E_1 - E_2)
        # C_d = (d - d_i)
        self._d_i = None
        self._lambda_eng = 0  # energy constraint lagrange multiplier
        self._lambda_dist = 0  # dist. constraint lagrange multiplier

    def set_method_and_n_cores(
        self,
        engrad_method: autode.wrappers.methods.Method,
        hess_method: autode.wrappers.methods.Method,
        n_cores: int,
    ):
        if not isinstance(engrad_method, autode.wrappers.methods.Method):
            raise ValueError(
                f"The engrad_method needs to be of type autode."
                f"wrappers.method.Method, But "
                f"{type(engrad_method)} was supplied."
            )
        self._engrad_method = engrad_method
        if not isinstance(hess_method, autode.wrappers.methods.Method):
            raise ValueError(
                f"The hess_method needs to be of type autode."
                f"wrappers.method.Method, But "
                f"{type(hess_method)} was supplied."
            )
        self._hess_method = hess_method
        self._n_cores = int(n_cores)

    def update_one_side_molecular_grad(self, side: str):
        assert self._engrad_method is not None
        assert self._n_cores is not None
        if side == "left":
            image = self._left_image
        elif side == "right":
            image = self._right_image
        else:
            raise Exception

        en, grad = _calculate_engrad_for_species(
            species=image.copy(),
            method=self._engrad_method,
            n_cores=self._n_cores,
        )
        image.energy = en
        image.gradient = grad

    def update_both_side_molecular_grad(self):
        # todo parallelise
        pass

    def get_one_sided_jacobian_of_constraints(self, side: str) -> np.ndarray:
        """
        Obtains the Jacobian (first derivative) of the constraints
        on one side of the image pair (i.e. only one side considered)

        Args:
            side: 'left' or 'right'

        Returns:
            (np.ndarray): An (n_atoms * n_constraints) shaped matrix,
                          in units of Hartree/angs
        """
        if side == "left":
            img = self._left_image
            fac = 1.0
        elif side == "right":
            img = self._right_image
            fac = -1.0
        else:
            raise Exception
        # 1st column is derivatives of C_E (if exists)
        # 2nd column is derivatives of C_d
        dist_vec = np.array(
            self._left_image.coordinates.to("ang").flatten()
            - self._right_image.coordinates.to("ang").flatten()
        )
        # grad(d - d_i) = grad(d)
        # grad_1(d) = (1/d) (r_1 - r_2)
        # grad_2(d) = (1/d) (r_2 - r_1) = - (1/d) (r_1 - r_2)
        dist_constr_grad = (
            float(1 / self.euclid_dist) * fac * dist_vec.reshape(-1, 1)
        )  # column vector
        A_matrix = dist_constr_grad
        if self._is_e_constr:
            assert self._left_image.gradient is not None
            assert self._right_image.gradient is not None
            # grad(E_1-E_2) = grad(E_1) if 1 else -grad(E_2)
            eng_constr_grad = fac * (
                np.array(img.gradient.to("ha/ang")).reshape(-1, 1)
            )
            A_matrix = np.hstack((dist_constr_grad, eng_constr_grad))
        return A_matrix

    def get_combined_jacobian_of_constraints(self) -> np.ndarray:
        """
        Obtains the Jacobian (first dervatives) of the constraints
        for the combined image pair (i.e. both ends considered)

        Returns:
             (np.ndarray): An (2 n_atoms * n_constraints) shaped array,
                           in units of Hartree/Angstrom
        """
        # A_matrix is n_atoms * n_constraints matrix
        # 1st column is derivatives of C_E against coords (if present)
        # 2nd column is derivatives of C_d against coords
        A_mat_r1 = self.get_one_sided_jacobian_of_constraints("left")
        A_mat_r2 = self.get_one_sided_jacobian_of_constraints("right")
        A_mat_total = np.vstack((A_mat_r1, A_mat_r2))
        return A_mat_total

    def get_one_sided_lagrangian_gradient(self, side: str) -> np.ndarray:
        """
        Get the gradient of the Lagrangian, for one side only

        Args:
            side: 'left' or 'right'

        Returns:
            (np.ndarray): The gradient of Lagrangian (L) in a flat array
        """
        if side == "left":
            img = self._left_image
        elif side == "right":
            img = self._right_image
        else:
            raise Exception
        assert img.gradient is not None

        grad = np.array(img.gradient.to("ha/ang")).flatten().reshape(-1, 1)
        # g_con = g - A @ lambda  <= lambda is a column matrix of multipliers
        if self._is_e_constr:
            lmda_col = np.array([self._lambda_eng, self._lambda_dist]).reshape(
                -1, 1
            )
            constr_func_col = np.array([-self.C_E, -self.C_d]).reshape(-1, 1)
        else:
            lmda_col = np.array([self._lambda_dist]).reshape(-1, 1)
            constr_func_col = np.array([-self.C_d]).reshape(-1, 1)
        A_matrix = self.get_one_sided_jacobian_of_constraints(side=side)
        grad_con = grad - (A_matrix @ lmda_col)

        # grad(L) = [[grad_con, -C_E, -C_d]]
        grad_L = np.vstack((grad_con, constr_func_col))
        return grad_L

    def update_one_side_molecular_hessian(self, side: str):
        assert self._hess_method is not None
        assert self._n_cores is not None
        if side == "left":
            img = self._left_image
        elif side == "right":
            img = self._right_image
        else:
            raise Exception

        hess = _calculate_hessian_for_species(
            species=img.copy(), method=self._hess_method, n_cores=self._n_cores
        )
        img.hessian = hess

    def get_one_sided_hessian_of_constraints(
        self, side: str
    ) -> List[np.ndarray]:
        """
        Obtain the Hessian of the constraint functions C_E and C_d

        Args:
            side: 'left' or 'right'

        Returns:
            (List[np.ndarray]): If energy constraint applied, then Hessians
                                of energy and distance constraint functions,
                                otherwise only distance constraint
        """
        if side == "left":
            img = self._left_image
            fac = 1.0
        elif side == "right":
            img = self._right_image
            fac = -1.0
        else:
            raise Exception

        hess_list = []
        # hess(d - d_i) = hess(d)
        dist_vec = np.array(
            self._left_image.coordinates.to("ang").flatten()
            - self._right_image.coordinates.to("ang").flatten()
        )
        grad_d = (
            float(1 / self.euclid_dist) * fac * dist_vec.reshape(-1, 1)
        )  # column vector
        hess_d = float(1 / self.euclid_dist) * (
            np.identity(self.n_atoms) - (grad_d @ grad_d.T)
        )
        hess_list.append(hess_d)

        if self._is_e_constr:
            assert img.hessian is not None
            # hess(E_1 - E_2) = hess(E_1) or -hess(E_2)
            eng_constr_hess = fac * img.hessian
            hess_list.append(eng_constr_hess)

        return hess_list

    def get_both_sided_hessian_of_constraints(self) -> List[np.ndarray]:
        hess_list = []
        # hess(d - d_i) = hess(d)
        z = np.concatenate(
            (
                np.array(self._left_image.coordinates.to("ang").flatten()),
                np.array(self._right_image.coordinates.to("ang").flatten()),
            )
        ).reshape(-1, 1)
        ident_mat = np.identity(self.n_atoms)
        transf_mat = np.vstack(
            (
                np.hstack((ident_mat, -ident_mat)),
                np.hstack((-ident_mat, ident_mat)),
            )
        )
        grad_d = float(1 / self.euclid_dist) * (transf_mat @ z)
        hess_d = float(1 / self.euclid_dist) * (
            transf_mat - (grad_d @ grad_d.T)
        )
        hess_list.append(hess_d)
        if self._is_e_constr:
            assert self._left_image.hessian is not None
            assert self._right_image.hessian is not None
            left_hess = np.array(self._left_image.hessian.to("ha/ang^2"))
            right_hess = np.array(self._right_image.hessian.to("ha/ang^2"))
            # hess(E_1 - E_2) = hess(E_1) - hess(E_2)
            filler = np.zeros((self.n_atoms, self.n_atoms))
            eng_constr_hess = np.vstack(
                (
                    np.hstack((left_hess, filler)),
                    np.hstack((filler, -right_hess)),
                )
            )
            hess_list.append(eng_constr_hess)
        return hess_list

    def get_one_sided_lagrangian_hessian(self, side: str) -> np.ndarray:
        if side == "left":
            img = self._left_image
        elif side == "right":
            img = self._right_image
        else:
            raise Exception

        assert img.hessian is not None
        H_matrix = np.array(img.hessian.to("ha/ang^2"))
        constr_hessians = self.get_one_sided_hessian_of_constraints(side=side)
        if self._is_e_constr:
            W_matrix = H_matrix - self._lambda_eng * constr_hessians[0]
            W_matrix -= self._lambda_dist * constr_hessians[1]
        else:
            W_matrix = H_matrix - self._lambda_dist * constr_hessians[0]

        A_matrix = self.get_one_sided_jacobian_of_constraints(side=side)
        end_zero = np.zeros((self.n_constraints, self.n_constraints))
        # hess(L) = [[W, -A],[-A.T, 0]]
        lagrange_hess = np.vstack(
            (
                np.hstack((W_matrix, -A_matrix)),
                np.hstack((-A_matrix.T, end_zero)),
            )
        )

        return lagrange_hess
        # todo refactor so that code is not copied

    @property
    def n_atoms(self):
        """Number of atoms"""
        return self._left_image.n_atoms

    @property
    def n_constraints(self):
        return 2 if self._is_e_constr else 1

    @property
    def target_dist(self) -> Distance:
        return Distance(self._d_i, units="ang")

    @target_dist.setter
    def target_dist(self, value):
        if isinstance(value, Distance):
            self._d_i = value.to("ang")
        elif isinstance(value, float):
            self._d_i = Distance(value, units="ang")
        else:
            raise ValueError(
                "The value of target_dist must be either"
                " autode.values.Distance or float, but "
                f"{type(value)} was supplied"
            )

    @property
    def euclid_dist(self) -> Distance:
        """
        Returns the euclidean distance between the two images of the pair
        """
        dist_vec = np.array(
            self._left_image.coordinates.to("ang").flatten()
            - self._right_image.coordinates.to("ang").flatten()
        )
        return Distance(np.linalg.norm(dist_vec), units="ang")

    @property
    def C_E(self) -> float:
        """
        Returns the current value of the energy constraint function
        (E_1 - E_2) in Hartree
        """
        return float(
            self._left_image.energy.to("Ha")
            - self._right_image.energy.to("Ha")
        )

    @property
    def C_d(self) -> float:
        """
        Returns the current value of the dustance constraint function
        (d - d_i) in Hartree
        """
        return float(self.euclid_dist.to("ang") - self.target_dist.to("ang"))


class DHS:
    def __init__(
        self,
        initial_species: autode.species.Species,
        final_species: autode.species.Species,
        reduction_factor: float = 0.05,
    ):
        self.imgpair = ConstrainedImagePair(
            initial_species, final_species, constrain_energy=False
        )
        self._reduction_fac = reduction_factor
