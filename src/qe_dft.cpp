#include "qe_dft.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

namespace qe_dft {

Qedft::Qedft(size_t nocc, SharedMatrix S, vector<SharedMatrix> C_N,
             vector<SharedMatrix> C_N_1, SharedMatrix eig_N_1)
    : nocc_{nocc}, S_{S}, C_N_{C_N}, C_N_1_{C_N_1}, eig_N_1_{eig_N_1}
{
    // check unrestricted calculation.
    if (C_N_.size() != 2) {
        throw std::runtime_error("miss one spin of CO coefficient matrix "
                                 "for N reference system.");
    }
    if (C_N_1_.size() != 2) {
        throw std::runtime_error("miss one spin of CO coefficient matrix "
                                 "for N-1 reference system.");
    }
    if (eig_N_1_->rows() != 2 && eig_N_1_->cols() != S_->rows()) {
        throw std::runtime_error(
            "miss one spin of eigenvalues for N-1 reference system.");
    }

    // check close shell N system.
    Matrix diff = (*C_N_[0]) - (*C_N_[1]);
    if (!diff.isZero(1e-8)) {
        throw std::runtime_error(
            "Found different CO coefficient matrix for alpha and beta spin "
            "for N referent system. N system has to be close shell.");
    }

    // check dimension mathcing
    if (!is_same_dimension(*C_N_[1], *C_N_[0])) {
        throw std::runtime_error(
            "Dimension does not match between alpha and beta spin of "
            "N-system CO coefficient matrix.");
    }
    if (!is_same_dimension(*C_N_1[1], *C_N_1[0])) {
        throw std::runtime_error(
            "Dimension does not match between alpha and beta spin of "
            "N-1 system CO coefficient matrix.");
    }
    if (!is_same_dimension(*S_, *C_N_[0])) {
        throw std::runtime_error("Dimension does not match between S and "
                                 "N-system CO coefficient matrix.");
    }
    if (!is_same_dimension(*S_, *C_N_1[0])) {
        throw std::runtime_error("Dimension does not match between S and "
                                 "N-1 system CO coefficient matrix.");
    }
}

OrbIndexPair Qedft::get_corresponding_orbitals(size_t index)
{
    const size_t nbasis = S_->rows();

    vector<size_t> rst_idx_vec;
    for (size_t s = 0; s < 2; ++s) {
        Matrix orb_overlap =
            C_N_1_[s]->transpose() * (*S_) * C_N_[s]->col(index);
        orb_overlap = orb_overlap.array().abs();
        size_t idx =
            std::max_element(orb_overlap.data(), orb_overlap.data() + nbasis) -
            orb_overlap.data();
        rst_idx_vec.push_back(idx);

#ifdef DEBUG_PRINT
        std::cout << "Orbital [" << index
                  << "] overlap with N-1 all alpha orbitals:\n"
                  << orb_overlap.transpose() << std::endl;
#endif
    }
    return OrbIndexPair(rst_idx_vec[0], rst_idx_vec[1]);
}

vector<OrbIndexPair> Qedft::get_corresponding_orbitals(vector<size_t> indices)
{
    vector<OrbIndexPair> rst;
    for (auto idx : indices) {
        rst.push_back(get_corresponding_orbitals(idx));
    }
    return rst;
}

ExciEnergyPair Qedft::excitation_energies(size_t index)
{
    if (index < nocc_) {
        throw std::runtime_error("Wrong excitation index. QEDFT only cover "
                                 "HOMO -> LUMO + (i) excitations.");
    }

    auto idx_pair = get_corresponding_orbitals(index);
    const double ref = eig_N_1_->row(1)[nocc_ - 1]; // beta lumo
    const double exci_a = eig_N_1_->row(0)[idx_pair.first];
    const double exci_b = eig_N_1_->row(1)[idx_pair.second];

    double exci_S = 2 * exci_b - exci_a - ref;
    double exci_T = exci_a - ref;
    return ExciEnergyPair(exci_S, exci_T);
}

vector<ExciEnergyPair> Qedft::excitation_energies(vector<size_t> indices)
{
    vector<ExciEnergyPair> rst;
    for (auto idx : indices) {
        rst.push_back(excitation_energies(idx));
    }
    return rst;
}


} // namespace qe_dft
