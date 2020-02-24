#ifndef _QE_DFT_SRC_QE_DFT_H_
#define _QE_DFT_SRC_QE_DFT_H_

#include <Eigen/Dense>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qe_dft {

using Matrix = Eigen::MatrixXd;
/**
 * @brief Orbital index pair <alpha, beta>.
 */
using OrbIndexPair = std::pair<size_t, size_t>;
using ExciEnergyPair = std::pair<double, double>;
using SharedMatrix = std::shared_ptr<Matrix>;
using std::shared_ptr;
using std::vector;

class Qedft {
  private:
    size_t nocc_;
    SharedMatrix S_;
    vector<SharedMatrix> C_N_;
    vector<SharedMatrix> C_N_1_;
    SharedMatrix eig_N_1_;

    bool is_same_dimension(const Matrix &A, const Matrix &B)
    {
        return (A.rows() == B.rows() && A.cols() == B.cols());
    }

  public:
    Qedft(size_t nocc, SharedMatrix S, vector<SharedMatrix> C_N,
          vector<SharedMatrix> C_N_1, SharedMatrix eig_N_1);

    /**
     * @brief Given one orbital index in N system, find the corresponding
     * orbitals (alpha and beta) index in N-1 system.
     *
     * @param [in] index: the orbital index in N system starts from zero.
     * @return OrbIndexPair: the correspongding orbitals index in N-1
     * system.
     */
    OrbIndexPair get_corresponding_orbitals(size_t index);

    /**
     * @brief Given one set of orbital indice in N system, find the
     * corresponding orbitals (alpha and beta) indices in N-1 system.
     *
     * @param [in] indices: the orbital indices in N system. Indices start from
     * zero.
     * @return vector<OrbIndexPair>: the set of correspongding orbitals
     * indices in N-1 system.
     */
    vector<OrbIndexPair> get_corresponding_orbitals(vector<size_t> indices);

    /**
     * @brief Calculate the excitation energy (singlet and triplet) from
     * QE-DFT method from N-1 system.
     * @param [in] index: the orbital index (starting from zero) relates to the
     * excited state.
     * @ return ExciEnergyPair: the excitation energy.
     */
    ExciEnergyPair excitation_energies(size_t index);

    /**
     * @brief Calculate the excitation energies (singlet and triplet) from
     * QE-DFT method from N-1 system.
     * @param [in] index: the set of orbital indices (starting from zero)
     * relate to the excited states.
     * @ return vector<ExciEnergyPair>: the excitation energies.
     */
    vector<ExciEnergyPair> excitation_energies(vector<size_t> indices);
};

} // namespace qe_dft

#endif // _QE_DFT_SRC_QE_DFT_H_
