#include "qe_dft.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <utility>

namespace qe_dft {

QedftBase::QedftBase(const vector<size_t> &nocc, SharedMatrix S,
                     vector<SharedMatrix> C_N, vector<SharedMatrix> C_N_1,
                     SharedMatrix eig_N_1)
    : nocc_{nocc}, S_{S}, C_N_{C_N}, C_N_1_{C_N_1}, eig_N_1_{eig_N_1}
{
    // check electron number
    if (nocc.size() != 2) {
        throw std::runtime_error(
            "Incomplete information for alpha and beta "
            "occupied orbitals number for N-1/N+1 system.");
    }
    if (nocc[0] < nocc[1]) {
        throw std::runtime_error(
            "Alpha electron number should be greater or equal to beta electron "
            "number.");
    }
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

QedftBase::MatchedOrbitalInfo QedftBase::get_corresponding_orbital(Spin spin,
                                                                   size_t index)
{
    const size_t nbasis = S_->rows();
    if (index >= nbasis) {
        std::stringstream msg;
        msg << "Input index = " << index
            << ", which exceeds the basis dimension.";
        throw std::runtime_error(msg.str());
    }
    const size_t is = (spin == Alpha ? 0 : 1);
    Eigen::VectorXd overlap = C_N_1_[is]->transpose() * (*S_) * C_N_[is]->col(index);
    overlap = overlap.array().abs();
    size_t idx = std::max_element(overlap.data(), overlap.data() + nbasis) -
                 overlap.data();
    return std::make_pair(idx, overlap(idx));
}

} // namespace qe_dft
