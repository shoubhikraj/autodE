#ifndef ADE_EXT_IDPP_H
#define ADE_EXT_IDPP_H

#include <vector>
#include "arrayhelper.hpp"

namespace autode
{

    /* A struct containing parameters used for IDPP */
    struct IdppParams {
        double k_spr;  // spring constant
        bool sequential;  // whether sequential IDPP or not
        bool debug;  // whether to print debug messages
        double rmsgtol; // RMS gradient tolerance for path
        int maxiter; // maxiter for path
        double add_img_maxgtol; // Max gradient tol. for adding img
        double add_img_maxiter; // maxiter for each image addition
        std::vector<double> cov_rs; // covalent radii for all atoms

        void check_validity() const;
    };

    class Image {
        /* A NEB image, holding coordinates, energy and gradients */

    private:
        int n_atoms; // number of atoms

    public:
        arrx::array1d coords;  // coordinates
        arrx::array1d grad;  // gradients
        double en = 0.0;  // energy
        double k_spr;  // force constant

        Image() = default;

        explicit Image(int num_atoms, double k);

        double get_tau_k_fac(arrx::array1d& tau,
                             const Image& img_m1,
                             const Image& img_p1,
                             const bool force_lc) const;

        void update_neb_grad(const Image& img_m1,
                             const Image& img_p1,
                             bool force_lc);

        double max_g() const;
    };


    class InterpPotentialBase {
        /* Base class for any interpolation potential */

        protected:
            int n_atoms = 0;   // Total number of atoms
            int n_images = 0;  // Total number of images

        public:
            // Polymorphic base must have virtual descriptor
            virtual ~InterpPotentialBase() = default;

            virtual void calc_potential_engrad(int idx, Image& img) const = 0;
    };

    class IDPPotential : public InterpPotentialBase {
        /* The IDPP potential, using weighted interatomic distances */

    private:
        std::vector<arrx::array1d> all_target_ds; // interpolated bond distances

    public:
        IDPPotential() = default;

        explicit IDPPotential(const arrx::array1d& init_coords,
                               const arrx::array1d& final_coords,
                               const int num_images);

        void calc_potential_engrad(int idx, Image& img) const override;
    };

    class ScaledInterAtomPotential : public InterpPotentialBase {
        /* Scaled interatomic distance potential using exponential scaling */

    private:
        std::vector<arrx::array1d> all_target_qs; // interpolated scaled distances
        arrx::array1d cov_radii; // covalent radii for each atom
        const double aleph = 1.7; // exponential scaling parameter
        const double bet = 0.01;   // inverse distance scaling parameter

        // Helper function to calculate r_ij_e (sum of covalent radii)
        double get_r_e(int atom_i, int atom_j) const;

        // Helper function to calculate q_ij from r_ij and r_ij_e
        double calc_q(double r_ij, double r_e) const;

        // Helper function to calculate dq/dr_ij
        double calc_dq_dr(double r_ij, double r_e) const;

    public:
        ScaledInterAtomPotential() = default;

        explicit ScaledInterAtomPotential(const arrx::array1d& init_coords,
                                           const arrx::array1d& final_coords,
                                           const int num_images,
                                           const std::vector<double>& cov_rs);

        void calc_potential_engrad(int idx, Image& img) const override;
    };

    class NEB {
        /* A NEB calculation for interpolation, holding images and potential */

    public:
        int n_images;  // total number of images
        int n_atoms;  // total number of atoms

        struct frontier_pair {
            int left, right;
        } frontier;  // frontier image indices
        bool images_prepared = false; // are images filled in?

        std::vector<Image> images;  // Set of images
        double k_spr;  // base spring constant

        NEB(arrx::array1d init_coords,
            arrx::array1d final_coords,
            double k_spr,
            int num_images);

        void fill_linear_interp();

        void fill_sequentially(const InterpPotentialBase& pot,
                               const int add_maxiter,
                               const double add_maxgtol);

        double get_d_id() const;

        double get_k_mid() const;

        void reset_k_spr();

        void get_engrad(double& en, arrx::array1d& grad) const;

        void get_frontier_engrad(double& en, arrx::array1d& grad) const;

        void get_coords(arrx::array1d& coords) const;

        void get_frontier_coords(arrx::array1d& coords) const;

        void set_coords(const arrx::array1d& coords);

        void set_frontier_coords(const arrx::array1d& coords);

        void add_first_two_images();

        void add_image_next_to(const int idx);

    };

    /* Barzilai-Borwein algorithm for minimisation */
    class BBMinimiser {

    private:
        int iter = 0;  // current number of iterations
        int maxiter;  // maximum number of iterations
        double gtol; // gradient tolerance criteria

        const double max_trust = 0.05; // maximum trust radius
        const double min_trust = 0.001; // minimum trust radius
        const double max_step = 0.05; //max displacement

        int last_tr_upd = 0; // last iteration of trust update
        double trust = 0.01; // current trust radius

        double last_low_rms_g; // last low RMS gradient
        double last_low_rmsg_iter; // iteration of last low RMS g
        int Nmin = 0; // number of consecutive iters where RMS g decreased
        arrx::array1d s_k, y_k; // change in x and g

    public:
        arrx::array1d coords, last_coords;  // current & last coordinates
        arrx::array1d grad, last_grad;  // current & last gradients
        double en, last_en;  // current & last energies
        arrx::array1d step;

        explicit BBMinimiser(int max_iter, double tol);

        void update_trust_radius();

        void take_step();

        int min_frontier(NEB& neb,
                         const NEB::frontier_pair idxs,
                         const InterpPotentialBase& pot);

        void min_path(NEB& neb, const InterpPotentialBase& pot);
    };

    NEB calculate_neb(arrx::array1d init_coords,
                      arrx::array1d final_coords,
                      const int num_images,
                      const IdppParams& params,
                      const arrx::array1d& interm_coords,
                      const bool load_from_interm);

    std::vector<double> calculate_idpp_path(double* init_coords_ptr,
                             double* final_coords_ptr,
                             int coords_len,
                             int n_images,
                             double* all_coords_ptr,
                             const IdppParams& params);

    double get_path_length(double* init_coords_ptr,
                         double* final_coords_ptr,
                         int coords_len,
                         int n_images,
                         const IdppParams& params);

    void relax_path(double* all_coords_ptr,
                    int coords_len,
                    int n_images,
                    const IdppParams& params);
}

#endif // ADE_EXT_IDPP_H
