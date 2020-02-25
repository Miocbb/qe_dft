#ifndef _QE_DFT_SRC_QE_DFT_H_
#define _QE_DFT_SRC_QE_DFT_H_

#include <Eigen/Dense>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

/**
 * @brief top-level namespace for QE-DFT method.
 */
namespace qe_dft {

using std::shared_ptr;
using std::size_t;
using std::vector;
using Matrix = Eigen::MatrixXd;
using SharedMatrix = std::shared_ptr<Matrix>;

/**
 * @brief Electron spin label.
 */
enum Spin {
    Alpha, /*< alpha spin. */
    Beta,  /*< beta spin. */
};

/**
 * @brief QE-DFT base class.
 */
class QedftBase {
  protected:
    vector<size_t> nocc_;        /*< N system occupied orbital numbers. */
    SharedMatrix S_;             /*< AO overlap matrix. */
    vector<SharedMatrix> C_N_;   /*< N system CO coefficient matrix. */
    vector<SharedMatrix> C_N_1_; /*< N-1/N+1 system CO COefficient matrix. */
    SharedMatrix eig_N_1_;       /*< eigenvalues of N-1/N+1 system. */

    bool is_same_dimension(const Matrix &A, const Matrix &B)
    {
        return (A.rows() == B.rows() && A.cols() == B.cols());
    }

  public:
    /**
     * @brief Matched orbitals information.
     * @details The first is the corresponding orbital index
     * (starting from zero) in N-1/N+1 system. The second is the overlap between
     * the corresponding orbital in N-1/N+1 system and the reference orbital in
     * N system.
     */
    using MatchedOrbitalInfo = std::pair<size_t, double>;

    /**
     * @brief QE-DFT base constructor.
     *
     * @param [in] nocc: occupation number of alpha and beta for N system.
     * @param [in] S: AO overlap matrix with dimension [nbasis, nbasis].
     * @param [in] C_N: CO coefficient matrix for N system. Dimension
     * [nbasis, nbasis]. Each column is a CO.
     * @param [in] C_N_1: CO coefficient matrix for N-1/N+1 system. Dimension
     * [nbasis, nbasis]. Each column is a CO.
     * @param [in] eig_N_1: N-1/N+1 system eigenvalues. Dimension [2, nbasis].
     * 1st row is the alpha eigenvalues. 2nd row is the beta eigenvalues.
     * The sequence of the eigenvalues is the same as `C_N_1` stored CO order.
     */
    QedftBase(const vector<size_t> &nocc, SharedMatrix S,
              vector<SharedMatrix> C_N, vector<SharedMatrix> C_N_1,
              SharedMatrix eig_N_1);

    /**
     * @brief Given one orbital index in N system, find the corresponding
     * orbital index in the calculation system (N-1/N+1 system).
     *
     * @param [in] spin: the spin of the input orbital.
     * @param [in] index: the orbital index in N system starts from zero.
     * @return MatchedOrbitalInfo: the corresponding orbital information.
     */
    MatchedOrbitalInfo get_corresponding_orbital(Spin spin, size_t index);
};

class QedftRemovalOpenShell : public QedftBase {
  public:
    QedftRemovalOpenShell(const vector<size_t> &nocc, SharedMatrix S,
                          vector<SharedMatrix> C_N, vector<SharedMatrix> C_N_1,
                          SharedMatrix eig_N_1)
        : QedftBase(nocc, S, C_N, C_N_1, eig_N_1)
    {
        if (nocc[0] != nocc[1]) // N system should be close shell
            throw std::runtime_error(
                "Error in QedftRemovalOpenShell constructor: the N-1 system "
                "is not open shell.");
    }

    /**
     * @brief Calculate the singlet excitation from QE-DFT method
     * from N-1 system.
     * @param [in] gs_orb_idx_b: the beta orbital index (starting from zero)
     * relates to the ground state.
     * @param [in] exci_orb_idx_a: the alpha orbital index (starting from zero)
     * relates to the excited state.
     * @param [in] exci_orb_idx_b: the beta orbital index (starting from zero)
     * relates to the excited state.
     * @ return double: singlet excitation energy in a.u..
     */
    double get_singlet_excitation(size_t gs_orb_idx_b, size_t exci_orb_idx_a,
                                  size_t exci_orb_idx_b)
    {
        const double gs_orbE_b = eig_N_1_->row(1)[gs_orb_idx_b];
        const double exci_orbE_a = eig_N_1_->row(0)[exci_orb_idx_a];
        const double exci_orbE_b = eig_N_1_->row(1)[exci_orb_idx_b];
        const double Es = 2 * exci_orbE_b - exci_orbE_a - gs_orbE_b;
        return Es;
    }

    /**
     * @brief Calculate the triplet excitation from QE-DFT method
     * from N-1 system.
     * @param [in] gs_orb_idx_b: the beta orbital index (starting from zero)
     * relates to the ground state.
     * @param [in] exci_orb_idx_a: the alpha orbital index (starting from zero)
     * relates to the excited state.
     * @ return double: triplet excitation energy in a.u..
     */
    double get_triplet_excitation(size_t gs_orb_idx_b, size_t exci_orb_idx_a)
    {
        const double gs_orbE_b = eig_N_1_->row(1)[gs_orb_idx_b];
        const double exci_orbE_a = eig_N_1_->row(0)[exci_orb_idx_a];
        const double Et = exci_orbE_a - gs_orbE_b;
        return Et;
    }

    /**
     * @brief the orbital index relates to the ground state N system.
     */
    size_t gs_orbital_index() const {return nocc_[0] - 1;}
};

class QedftRemovalCloseShell : public QedftBase {
  public:
    QedftRemovalCloseShell(const vector<size_t> &nocc, SharedMatrix S,
                           vector<SharedMatrix> C_N, vector<SharedMatrix> C_N_1,
                           SharedMatrix eig_N_1)
        : QedftBase(nocc, S, C_N, C_N_1, eig_N_1)
    {
        if (nocc[0] == nocc[1]) // N system should be open shell
            throw std::runtime_error(
                "Error in QedftRemovalCloseShell constructor: the N-1 system "
                "is not close shell.");
    }

    /**
     * @brief Calculate the doublet excitation from QE-DFT method from N-1
     * system.
     * @param [in] gs_orb_idx_a: the alpha orbital index (starting from zero)
     * relates to the ground state.
     * @param [in] exci_orb_idx_a: the alpha orbital index (starting from zero,
     * alpha and beta will be the same) relates to the excited state.
     * @ return double: the doublet excitation energy.
     */
    double get_doublet_excitation(size_t gs_orb_idx_a, size_t exci_orb_idx_a)
    {
        const double gs_orb_E_a = eig_N_1_->row(0)[gs_orb_idx_a]; // alpha lumo in N-1
        const double exci_orb_E_a = eig_N_1_->row(0)[exci_orb_idx_a];
        const double Ed = exci_orb_E_a - gs_orb_E_a;
        return Ed;
    }

    /**
     * @brief the orbital index relates to the ground state N system.
     */
    size_t gs_orbital_index() const {return nocc_[0] - 1;}
};

} // namespace qe_dft

#endif // _QE_DFT_SRC_QE_DFT_H_
